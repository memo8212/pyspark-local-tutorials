"""04 - Aggregations and joins.

Run:  python tutorials/04_aggregations_and_joins.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import generate_all  # noqa: E402
from common.helpers import banner, section  # noqa: E402
from common.paths import DATA_DIR  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("04 - Aggregations & Joins")
    generate_all()
    spark = get_spark("04-agg-joins")
    from pyspark.sql import functions as F

    orders = spark.read.csv(str(DATA_DIR / "orders.csv"), header=True, inferSchema=True)
    customers = (spark.read.csv(str(DATA_DIR / "customers.csv"), header=True, inferSchema=True)
                 .withColumnRenamed("name", "customer_name"))
    products = (spark.read.csv(str(DATA_DIR / "products.csv"), header=True, inferSchema=True)
                .withColumnRenamed("name", "product_name"))

    section("groupBy + agg")
    (orders.groupBy("channel")
           .agg(F.count("*").alias("orders"),
                F.sum("qty").alias("units"),
                F.avg("qty").alias("avg_qty"),
                F.min("order_date").alias("first_order"),
                F.max("order_date").alias("last_order"))
           .orderBy(F.col("orders").desc())
           .show())

    section("countDistinct")
    orders.agg(F.countDistinct("customer_id").alias("distinct_customers")).show()

    section("Join: enrich orders with customer + product")
    enriched = (orders
                .join(customers, on="customer_id", how="inner")
                .join(products, on="product_id", how="inner")
                .withColumn("line_total", F.round(F.col("qty") * F.col("unit_price"), 2)))
    enriched.select("order_id", "customer_name", "region", "category", "line_total").show(5)

    section("Join types")
    left = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["k", "v"])
    right = spark.createDataFrame([(2, "B"), (3, "C"), (4, "D")], ["k", "w"])
    print("inner   :", sorted((r.k, r.v, r.w) for r in left.join(right, "k", "inner").collect()))
    print("left    :", sorted((r.k, r.v, r.w) for r in left.join(right, "k", "left").collect()))
    print("left_anti:", [r.k for r in left.join(right, "k", "left_anti").collect()])
    print("left_semi:", [r.k for r in left.join(right, "k", "left_semi").collect()])
    print("full    :", sorted((r.k, r.v, r.w) for r in left.join(right, "k", "full").collect()))

    section("Aggregate after join: revenue by region")
    (enriched.groupBy("region")
             .agg(F.round(F.sum("line_total"), 2).alias("revenue"),
                  F.count("order_id").alias("orders"))
             .orderBy(F.col("revenue").desc())
             .show())

    section("pivot: orders per channel per region")
    (orders.join(customers, "customer_id")
           .groupBy("region").pivot("channel").count()
           .na.fill(0).orderBy("region").show())

    section("union / unionByName")
    a = spark.createDataFrame([(1, "x")], ["id", "val"])
    b = spark.createDataFrame([(2, "y")], ["id", "val"])
    a.union(b).show()
    a.unionByName(b).show()

    stop_spark(spark)


if __name__ == "__main__":
    main()
