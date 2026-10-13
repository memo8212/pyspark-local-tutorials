"""03 - DataFrame transformations: select/filter/withColumn/when/cast/sort.

Run:  python tutorials/03_dataframe_transformations.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("03 - DataFrame Transformations")
    spark = get_spark("03-transformations")
    from pyspark.sql import functions as F

    df = spark.createDataFrame(
        [
            (1, "Alice", "NORTH", "2024-01-05", 120.0, 2),
            (2, "Bob", "WEST", "2024-01-07", 80.0, 1),
            (3, "Carol", "NORTH", "2024-02-01", 300.0, 5),
            (4, "Dan", "EAST", "2024-02-11", 45.0, 3),
            (5, "Erin", "WEST", "2024-03-02", 900.0, 1),
        ],
        ["id", "name", "region", "order_date", "amount", "qty"],
    )
    df.show()

    section("select / selectExpr")
    df.select("id", "region", "amount").show()
    df.selectExpr("id", "region", "amount * qty AS line_total").show()

    section("filter / where (equivalent)")
    df.filter(F.col("amount") >= 100).show()
    df.where((F.col("region") == "NORTH") & (F.col("amount") > 100)).show()
    df.filter(F.col("region").isin("NORTH", "WEST")).show()
    df.filter(F.col("amount").between(50, 300)).show()
    df.filter(F.col("name").like("A%")).show()

    section("withColumn, when/otherwise, cast")
    enriched = (
        df
        .withColumn("line_total", F.col("amount") * F.col("qty"))
        .withColumn("tier", F.when(F.col("line_total") < 300, "LOW")
                            .when(F.col("line_total") < 1000, "MED")
                            .otherwise("HIGH"))
        .withColumn("order_date", F.to_date("order_date", "yyyy-MM-dd"))
        .withColumn("qty", F.col("qty").cast("int"))
    )
    enriched.show()

    section("drop, distinct, dropDuplicates")
    print("drop a column:", enriched.drop("tier").columns)
    print("distinct regions:", [r[0] for r in df.select("region").distinct().collect()])
    dup = df.union(df)  # duplicate every row
    print("with dups:", dup.count(), "| after dropDuplicates:", dup.dropDuplicates().count())

    section("orderBy / sort, limit")
    df.orderBy(F.col("amount").desc()).show()
    df.orderBy("region", F.col("amount").desc()).limit(3).show()

    section("Handling nulls")
    with_nulls = df.withColumn("region", F.when(F.col("id") == 2, None).otherwise(F.col("region")))
    with_nulls.na.drop(subset=["region"]).show()
    with_nulls.na.fill({"region": "UNKNOWN"}).show()

    stop_spark(spark)


if __name__ == "__main__":
    main()
