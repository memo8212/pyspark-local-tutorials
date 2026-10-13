"""18 - End-to-end project: Bronze -> Silver -> Gold + ML scoring on local Delta.

Run:  python tutorials/18_end_to_end_project.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import generate_all  # noqa: E402
from common.helpers import banner, section  # noqa: E402
from common.paths import DATA_DIR, OUTPUT_DIR  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("18 - End-to-End Project (local Delta medallion)")
    generate_all()
    spark = get_spark("18-e2e")
    from pyspark.sql import functions as F, Window

    lake = OUTPUT_DIR / "lakehouse"
    bronze = str(lake / "bronze")
    silver = str(lake / "silver")
    gold = str(lake / "gold")

    # ------------------------------------------------------------- Bronze
    section("Bronze: ingest raw CSVs as Delta with audit columns")
    orders = spark.read.csv(str(DATA_DIR / "orders.csv"), header=True, inferSchema=True)
    (orders.withColumn("_ingested_at", F.current_timestamp())
           .withColumn("_source", F.lit("orders.csv"))
           .write.format("delta").mode("overwrite").save(bronze + "/orders"))
    products = spark.read.csv(str(DATA_DIR / "products.csv"), header=True, inferSchema=True)
    (products.withColumn("_ingested_at", F.current_timestamp())
             .write.format("delta").mode("overwrite").save(bronze + "/products"))
    customers = spark.read.csv(str(DATA_DIR / "customers.csv"), header=True, inferSchema=True)
    (customers.withColumn("_ingested_at", F.current_timestamp())
              .write.format("delta").mode("overwrite").save(bronze + "/customers"))
    print("bronze orders:", spark.read.format("delta").load(bronze + "/orders").count())

    # ------------------------------------------------------------- Silver
    section("Silver: join + derive + dedupe")
    o = spark.read.format("delta").load(bronze + "/orders").drop("_ingested_at", "_source")
    p = (spark.read.format("delta").load(bronze + "/products")
         .withColumnRenamed("name", "product_name"))
    c = (spark.read.format("delta").load(bronze + "/customers")
         .withColumnRenamed("name", "customer_name"))

    full = (o.join(p, "product_id", "inner")
              .join(c, "customer_id", "inner")
              .withColumn("order_date", F.to_date("order_date", "yyyy-MM-dd"))
              .withColumn("line_total", F.round(F.col("qty") * F.col("unit_price"), 2))
              .withColumn("order_month", F.date_format("order_date", "yyyy-MM"))
              .drop("_ingested_at"))
    dedup = Window.partitionBy("order_id").orderBy(F.col("order_date").desc())
    full = full.withColumn("rn", F.row_number().over(dedup)).filter("rn = 1").drop("rn")
    full.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver + "/sales")
    print("silver sales rows:", full.count())

    # ------------------------------------------------------------- Gold
    section("Gold: business aggregates")
    by_region = (full.groupBy("region")
                     .agg(F.count("order_id").alias("orders"),
                          F.round(F.sum("line_total"), 2).alias("revenue"))
                     .orderBy(F.col("revenue").desc()))
    by_region.write.format("delta").mode("overwrite").save(gold + "/revenue_by_region")

    by_category = (full.groupBy("category")
                       .agg(F.round(F.sum("line_total"), 2).alias("revenue"))
                       .orderBy(F.col("revenue").desc()))
    by_category.write.format("delta").mode("overwrite").save(gold + "/revenue_by_category")
    by_region.show()
    by_category.show()

    # ------------------------------------------------------------- ML scoring
    section("ML: predict line_total from qty and unit_price (regression)")
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.evaluation import RegressionEvaluator

    train, test = full.randomSplit([0.8, 0.2], seed=1)
    model = Pipeline(stages=[
        VectorAssembler(inputCols=["qty", "unit_price"], outputCol="features"),
        LinearRegression(featuresCol="features", labelCol="line_total"),
    ]).fit(train)
    preds = model.transform(test)
    rmse = RegressionEvaluator(labelCol="line_total", metricName="rmse").evaluate(preds)
    print(f"line_total RMSE = {rmse:.3f}")

    scored = preds.withColumn("predicted_total", F.round("prediction", 2))
    scored.select("order_id", "qty", "unit_price", "line_total", "predicted_total").show(5)
    scored.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
          .save(gold + "/scored_sales")

    # ------------------------------------------------------------- Report
    section("Report")
    print("Lakehouse written under:", lake)
    print("  bronze/orders, bronze/products, bronze/customers")
    print("  silver/sales")
    print("  gold/revenue_by_region, gold/revenue_by_category, gold/scored_sales")
    total_rev = full.agg(F.sum("line_total")).first()[0]
    print(f"total revenue: {total_rev:,.2f}")

    section("Delta history of silver/sales")
    spark.sql(f"DESCRIBE HISTORY delta.`{silver}/sales`") \
         .select("version", "operation").show(truncate=False)

    stop_spark(spark)


if __name__ == "__main__":
    main()
