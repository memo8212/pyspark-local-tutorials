"""17 - Testing PySpark code (pytest patterns + inline assertions).

Run:  python tutorials/17_testing_pyspark.py
See also: tests/test_transformations.py (run with `pytest -q`).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


# --------------------------------------------------------------------------
# Put business logic in pure functions that take/return DataFrames.
# They are then easy to unit-test without a cluster.
# --------------------------------------------------------------------------
def enrich_orders(orders):
    from pyspark.sql import functions as F
    return (
        orders
        .withColumn("order_date", F.to_date("order_date", "yyyy-MM-dd"))
        .withColumn("line_total", F.round(F.col("amount") * F.col("qty"), 2))
        .withColumn("tier", F.when(F.col("line_total") < 300, "LOW")
                            .otherwise("HIGH"))
    )


def main() -> None:
    banner("17 - Testing PySpark")
    spark = get_spark("17-testing")
    from pyspark.sql import functions as F

    section("Arrange: a tiny, deterministic input")
    orders = spark.createDataFrame(
        [(1, "2024-01-05", 120.0, 2), (2, "2024-01-07", 80.0, 1)],
        ["order_id", "order_date", "amount", "qty"],
    )

    section("Act")
    result = enrich_orders(orders)
    result.show()

    section("Assert: compare against expected rows (order-independent)")
    expected = {
        (1, "LOW"),    # 120*2=240 -> LOW
        (2, "LOW"),    # 80*1=80  -> LOW
    }
    actual = {(r["order_id"], r["tier"]) for r in result.select("order_id", "tier").collect()}
    assert actual == expected, f"tier mismatch: {actual}"
    assert result.filter("line_total IS NULL").count() == 0
    print("inline assertions passed")

    section("Schema assertion (guard against regressions)")
    dtypes = dict(result.dtypes)
    assert dtypes["order_date"] == "date"
    assert dtypes["line_total"] == "double"
    assert dtypes["tier"] == "string"
    print("schema assertions passed")

    section("Numeric assertion with tolerance")
    total = result.agg(F.sum("line_total")).first()[0]
    assert abs(total - 320.0) < 1e-9, total
    print("numeric assertion passed")

    section("What the pytest file covers")
    print("- a module-scoped SparkSession fixture (create once, reuse)")
    print("- testing transformations, aggregations and schema")
    print("- deterministic inputs (no randomness in tests)")
    print("Run them with:  pytest -q")

    stop_spark(spark)


if __name__ == "__main__":
    main()
