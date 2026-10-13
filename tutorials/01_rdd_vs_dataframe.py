"""01 - RDDs vs DataFrames, laziness, transformations vs actions.

Run:  python tutorials/01_rdd_vs_dataframe.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("01 - RDDs vs DataFrames")
    spark = get_spark("01-rdd-vs-dataframe")
    sc = spark.sparkContext

    # ---------------------------------------------------------------- RDDs
    section("RDDs: low-level, untyped, distributed collections")
    numbers = sc.parallelize(range(1, 11), numSlices=4)
    evens = numbers.filter(lambda x: x % 2 == 0)      # transformation (lazy)
    squared = evens.map(lambda x: x * x)              # transformation (lazy)
    total = squared.reduce(lambda a, b: a + b)        # action (triggers work)
    print("even squares:", squared.collect())
    print("sum of even squares:", total)

    # ----------------------------------------------------------- DataFrames
    section("DataFrames: typed, optimized (Catalyst + Tungsten)")
    df = spark.createDataFrame(
        [(1, "Alice", 250.0), (2, "Bob", 90.0), (3, "Carol", 1300.0)],
        ["id", "name", "amount"],
    )
    df.show()

    from pyspark.sql import functions as F

    high = df.filter(F.col("amount") >= 100).orderBy(F.col("amount").desc())
    print("rows with amount >= 100:")
    high.show()

    # ------------------------------------------------------------- Laziness
    section("Laziness: transformations build a plan; actions run it")
    plan = df.filter(F.col("amount") > 0).withColumn("tax", F.col("amount") * 0.13)
    print("filter+withColumn returned instantly (no job ran yet).")
    print("Now an action (.count()) triggers execution:")
    print("row count:", plan.count())

    print("\nTakeaway: prefer DataFrames (optimizer + types). Use RDDs only for "
          "unstructured byte-level work.")

    stop_spark(spark)


if __name__ == "__main__":
    main()
