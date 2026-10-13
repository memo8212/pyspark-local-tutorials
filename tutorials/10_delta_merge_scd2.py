"""10 - Delta MERGE in depth: SCD1 upsert and SCD2 history.

Run:  python tutorials/10_delta_merge_scd2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("10 - Delta MERGE: SCD1 & SCD2")
    spark = get_spark("10-delta-merge-scd")
    from pyspark.sql import functions as F, Window

    # ---- raw source batches (Bronze)
    section("Bronze: two batches (batch2 corrects Bob, adds Carol & Dave)")
    batch1 = spark.createDataFrame(
        [(1, "Alice", "NORTH", 250.0), (2, "Bob", "WEST", 90.0)],
        ["cust_id", "name", "region", "amount"],
    ).withColumn("updated_at", F.to_timestamp(F.lit("2025-01-01 10:00:00")))
    batch2 = spark.createDataFrame(
        [(2, "Bob", "SOUTH", 150.0), (3, "Carol", "EAST", 500.0), (4, "Dave", "NORTH", 40.0)],
        ["cust_id", "name", "region", "amount"],
    ).withColumn("updated_at", F.to_timestamp(F.lit("2025-01-02 09:00:00")))
    batch1.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("bronze_customers")
    batch2.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("bronze_customers")

    # latest row per key from Bronze
    latest = (spark.table("bronze_customers")
              .withColumn("rn", F.row_number().over(
                  Window.partitionBy("cust_id").orderBy(F.col("updated_at").desc())))
              .filter("rn = 1").drop("rn"))
    latest.createOrReplaceTempView("latest_customers")

    # ------------------------------------------------------------- SCD1
    section("SCD1: keep only the latest version (MERGE upsert)")
    latest.limit(0).write.format("delta").mode("overwrite").saveAsTable("silver_scd1")
    spark.sql("""
        MERGE INTO silver_scd1 AS t
        USING latest_customers AS s
        ON t.cust_id = s.cust_id
        WHEN MATCHED AND s.updated_at > t.updated_at THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    spark.table("silver_scd1").orderBy("cust_id").show()

    # ------------------------------------------------------------- SCD2
    section("SCD2: keep full history with effective dating")
    dim = "silver_dim_customer"

    # seed with batch1 as the current version
    (batch1.select("cust_id", "name", "region", "updated_at")
          .withColumn("effective_from", F.col("updated_at"))
          .withColumn("effective_to", F.lit(None).cast("timestamp"))
          .withColumn("is_current", F.lit(True))
          .withColumn("sk", F.expr("uuid()"))
          .drop("updated_at")
          .write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(dim))

    # step 1: close current rows whose tracked attributes changed
    spark.sql(f"""
        MERGE INTO {dim} AS t
        USING (SELECT DISTINCT cust_id, name, region
               FROM   bronze_customers
               WHERE  updated_at >= '2025-01-02') AS s
        ON t.cust_id = s.cust_id AND t.is_current = true
        WHEN MATCHED AND (t.name <> s.name OR t.region <> s.region)
        THEN UPDATE SET t.is_current = false, t.effective_to = current_timestamp()
    """)

    # step 2: insert a new current row for keys with no current row (changed or new)
    spark.sql(f"""
        INSERT INTO {dim}
        SELECT s.cust_id, s.name, s.region,
               s.updated_at AS effective_from,
               CAST(NULL AS TIMESTAMP) AS effective_to,
               true AS is_current, uuid() AS sk
        FROM (SELECT cust_id, name, region, updated_at,
                     row_number() OVER (PARTITION BY cust_id ORDER BY updated_at DESC) rn
              FROM bronze_customers) s
        LEFT JOIN {dim} t ON t.cust_id = s.cust_id AND t.is_current = true
        WHERE s.rn = 1 AND t.cust_id IS NULL
    """)

    spark.table(dim).orderBy("cust_id", "effective_from").show(truncate=False)

    section("Point-in-time query: who was 'current' on 2025-01-01?")
    spark.sql(f"""
        SELECT cust_id, name, region
        FROM   {dim}
        WHERE  effective_from <= '2025-01-01 23:59:59'
          AND (effective_to IS NULL OR effective_to > '2025-01-01 23:59:59')
        ORDER  BY cust_id
    """).show()

    stop_spark(spark)


if __name__ == "__main__":
    main()
