"""02 - DataFrame basics: creation, schemas, inspection, columns.

Run:  python tutorials/02_dataframe_basics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("02 - DataFrame Basics")
    spark = get_spark("02-dataframe-basics")

    # ---------------------------------------------------- create from tuples
    section("Create from a list of tuples")
    df = spark.createDataFrame(
        [(1, "Alice", "NORTH", 250.0), (2, "Bob", "WEST", 90.0)],
        ["id", "name", "region", "amount"],
    )
    df.show()

    # ------------------------------------------------ create with explicit schema
    section("Create with an explicit schema (recommended for real data)")
    from pyspark.sql.types import (StructType, StructField, IntegerType,
                                   StringType, DoubleType)
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("name", StringType(), True),
        StructField("region", StringType(), True),
        StructField("amount", DoubleType(), True),
    ])
    df2 = spark.createDataFrame([(3, "Carol", "NORTH", 1300.0)], schema)
    df2.show()

    # ---------------------------------------------------- from pandas
    section("Create from a pandas DataFrame")
    import pandas as pd
    pdf = pd.DataFrame({"id": [4], "name": ["Dan"], "region": ["EAST"], "amount": [480.0]})
    spark.createDataFrame(pdf).show()

    # ---------------------------------------------------- inspect
    section("Inspect: schema, columns, dtypes, describe")
    df.printSchema()
    print("columns :", df.columns)
    print("dtypes  :", df.dtypes)
    print("count   :", df.count())
    df.describe("amount").show()

    # ---------------------------------------------------- columns
    section("Columns: select, rename, add")
    from pyspark.sql import functions as F
    (df
     .withColumnRenamed("name", "customer_name")
     .withColumn("amount_with_tax", F.round(F.col("amount") * 1.13, 2))
     .select("id", "customer_name", "amount", "amount_with_tax")
     .show())

    # ---------------------------------------------------- collect vs show
    section("show() vs collect() vs take()")
    print("first row     :", df.first())
    print("first 2 rows  :", df.take(2))
    print("all rows (small!):", df.collect())  # never collect big data

    stop_spark(spark)


if __name__ == "__main__":
    main()
