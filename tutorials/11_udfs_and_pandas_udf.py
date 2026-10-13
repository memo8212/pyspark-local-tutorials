"""11 - UDFs and pandas UDFs (and when NOT to use them).

Run:  python tutorials/11_udfs_and_pandas_udf.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("11 - UDFs & pandas UDFs")
    spark = get_spark("11-udfs")
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, DoubleType
    from pyspark.sql.functions import udf, pandas_udf

    df = spark.createDataFrame(
        [(1, "alice@example.com", 250.0), (2, "bob@test.org", 90.0)],
        ["id", "email", "amount"],
    )

    section("Prefer built-in functions (optimized by Catalyst)")
    df.withColumn("domain", F.split("email", "@")[1]).show()

    section("Python UDF (row-by-row: simple but slower)")
    @udf(StringType())
    def email_domain(email):
        return email.split("@")[-1] if email else None

    df.withColumn("domain_udf", email_domain("email")).show()

    section("Register a UDF for use in SQL")
    spark.udf.register("email_domain_sql", email_domain)
    df.createOrReplaceTempView("users")
    spark.sql("SELECT id, email, email_domain_sql(email) AS domain FROM users").show()

    section("pandas UDF (vectorized: much faster — operates on whole batches)")
    # NOTE: return a pandas Series (required by Spark 4.x)
    import pandas as pd

    @pandas_udf(DoubleType())
    def with_tax(amount: pd.Series) -> pd.Series:
        return (amount * 1.13).round(2)

    df.withColumn("amount_taxed", with_tax("amount")).show()

    section("pandas UDF: batch aggregation")
    @pandas_udf(DoubleType())
    def mean_plus_one(s: pd.Series) -> float:
        return float(s.mean() + 1.0)

    df.agg(mean_plus_one("amount").alias("mean_amount_plus1")).show()

    section("Guidance")
    print("- Built-ins first: they run in the JVM and are optimized.")
    print("- Python UDFs: flexible but row-by-row and slow (data crosses to Python).")
    print("- pandas UDFs: vectorized via Arrow; use for heavy per-row logic.")
    print("- For ML-style feature work, prefer MLlib transformers (see 13-15).")

    stop_spark(spark)


if __name__ == "__main__":
    main()
