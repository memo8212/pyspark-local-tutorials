"""Unit tests for common DataFrame logic. Run with: pytest -q"""
from __future__ import annotations

from pyspark.sql import functions as F, Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType


def make_orders(spark):
    schema = StructType([
        StructField("order_id", IntegerType()),
        StructField("region", StringType()),
        StructField("amount", DoubleType()),
        StructField("qty", IntegerType()),
    ])
    return spark.createDataFrame(
        [(1, "NORTH", 120.0, 2), (2, "WEST", 80.0, 1), (3, "NORTH", 300.0, 5)],
        schema,
    )


def test_filter_and_count(spark):
    df = make_orders(spark)
    assert df.filter(F.col("region") == "NORTH").count() == 2
    assert df.filter(F.col("amount") >= 100).count() == 2


def test_with_column_line_total(spark):
    df = make_orders(spark).withColumn("line_total", F.col("amount") * F.col("qty"))
    rows = {r["order_id"]: r["line_total"] for r in df.select("order_id", "line_total").collect()}
    assert rows == {1: 240.0, 2: 80.0, 3: 1500.0}


def test_aggregation_by_region(spark):
    df = make_orders(spark)
    agg = {r["region"]: r["total"] for r in
           df.groupBy("region").agg(F.sum("amount").alias("total")).collect()}
    assert agg == {"NORTH": 420.0, "WEST": 80.0}


def test_window_dedup_keeps_latest(spark):
    df = spark.createDataFrame(
        [(1, "a", "2024-01-01"), (1, "b", "2024-01-05"), (2, "c", "2024-01-03")],
        ["id", "val", "ts"],
    )
    w = Window.partitionBy("id").orderBy(F.col("ts").desc())
    dedup = df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")
    vals = {r["id"]: r["val"] for r in dedup.collect()}
    assert vals == {1: "b", 2: "c"}


def test_join_types(spark):
    left = spark.createDataFrame([(1, "a"), (2, "b")], ["k", "v"])
    right = spark.createDataFrame([(2, "B"), (3, "C")], ["k", "w"])
    assert left.join(right, "k", "inner").count() == 1
    assert left.join(right, "k", "left").count() == 2
    assert sorted(r.k for r in left.join(right, "k", "left_anti").collect()) == [1]


def test_schema_is_stable(spark):
    df = make_orders(spark)
    dtypes = dict(df.dtypes)
    assert dtypes["amount"] == "double"
    assert dtypes["qty"] == "int"
    assert set(df.columns) == {"order_id", "region", "amount", "qty"}


def test_delta_roundtrip(spark, tmp_path):
    path = str(tmp_path / "delta_orders")
    df = make_orders(spark)
    df.write.format("delta").mode("overwrite").save(path)
    back = spark.read.format("delta").load(path)
    assert back.count() == df.count()
    # update + time travel
    spark.sql(f"UPDATE delta.`{path}` SET amount = amount + 1 WHERE region = 'WEST'")
    assert back.count() == 3
    v0 = spark.read.format("delta").option("versionAsOf", 0).load(path)
    west_v0 = v0.filter("region = 'WEST'").first()["amount"]
    assert west_v0 == 80.0
