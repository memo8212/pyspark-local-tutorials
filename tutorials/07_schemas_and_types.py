"""07 - Schemas & types: StructType, casting, dates, nulls, nested data.

Run:  python tutorials/07_schemas_and_types.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("07 - Schemas & Types")
    spark = get_spark("07-schemas-types")
    from datetime import date
    from pyspark.sql import functions as F
    from pyspark.sql.types import (StructType, StructField, IntegerType,
                                   StringType, DoubleType, DateType,
                                   ArrayType, MapType)

    section("Explicit schema (do this for real pipelines)")
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("signup", DateType(), True),
    ])
    df = spark.createDataFrame(
        [(1, "Alice", 250.5, date(2024, 1, 5)), (2, "Bob", 90.0, date(2024, 3, 11))],
        schema,
    )
    df.printSchema()
    df.show()

    section("Casting between types")
    df2 = spark.createDataFrame([("1", "250.50", "2024-01-05")], ["id", "amount", "signup"])
    df2.printSchema()
    casted = (df2
              .withColumn("id", F.col("id").cast("int"))
              .withColumn("amount", F.col("amount").cast("double"))
              .withColumn("signup", F.to_date("signup", "yyyy-MM-dd")))
    casted.printSchema()
    casted.show()

    section("Date/time functions")
    (df
     .withColumn("year", F.year("signup"))
     .withColumn("month", F.month("signup"))
     .withColumn("next_week", F.date_add("signup", 7))
     .withColumn("days_since", F.datediff(F.current_date(), "signup"))
     .show())

    section("Nulls: coalesce, isnull, fill, drop")
    nulls = spark.createDataFrame([(1, None), (2, "x"), (3, None)], ["id", "val"])
    nulls.withColumn("val2", F.coalesce("val", F.lit("MISSING"))).show()
    nulls.select(F.count("*").alias("rows"),
                 F.count("val").alias("non_null"),
                 F.sum(F.col("val").isNull().cast("int")).alias("nulls")).show()

    section("Nested: struct, array, map")
    nested = spark.createDataFrame(
        [
            (1, ["a", "b"], {"k1": 10, "k2": 20}),
            (2, ["c"], {"k1": 30}),
        ],
        StructType([
            StructField("id", IntegerType()),
            StructField("tags", ArrayType(StringType())),
            StructField("props", MapType(StringType(), IntegerType())),
        ]),
    )
    nested.printSchema()
    (nested
     .withColumn("first_tag", F.col("tags")[0])
     .withColumn("n_tags", F.size("tags"))
     .withColumn("k1", F.col("props")["k1"])
     .show())

    stop_spark(spark)


if __name__ == "__main__":
    main()
