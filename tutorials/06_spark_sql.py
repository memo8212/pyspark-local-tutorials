"""06 - Spark SQL: temp views, CTEs, window SQL, mixing with Python.

Run:  python tutorials/06_spark_sql.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import generate_all  # noqa: E402
from common.helpers import banner, section  # noqa: E402
from common.paths import DATA_DIR  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("06 - Spark SQL")
    generate_all()
    spark = get_spark("06-spark-sql")

    sales = spark.read.csv(str(DATA_DIR / "orders.csv"), header=True, inferSchema=True)
    customers = spark.read.csv(str(DATA_DIR / "customers.csv"), header=True, inferSchema=True)

    section("Register temp views and query with SQL")
    sales.createOrReplaceTempView("orders")
    customers.createOrReplaceTempView("customers")

    spark.sql("""
        SELECT channel, COUNT(*) AS orders, SUM(qty) AS units
        FROM   orders
        GROUP  BY channel
        ORDER  BY orders DESC
    """).show()

    section("CTE + join")
    spark.sql("""
        WITH o AS (SELECT customer_id, SUM(qty) AS units FROM orders GROUP BY customer_id)
        SELECT c.region, ROUND(AVG(o.units), 2) AS avg_units, COUNT(*) AS customers
        FROM   customers c JOIN o ON c.customer_id = o.customer_id
        GROUP  BY c.region
        ORDER  BY avg_units DESC
    """).show()

    section("Window functions in SQL")
    spark.sql("""
        SELECT channel, order_date, qty,
               SUM(qty) OVER (PARTITION BY channel ORDER BY order_date) AS running_units,
               RANK()     OVER (PARTITION BY channel ORDER BY qty DESC) AS rnk
        FROM   orders
        ORDER  BY channel, order_date
        LIMIT  10
    """).show()

    section("spark.sql() returns a DataFrame (mix SQL and Python)")
    from pyspark.sql import functions as F
    top = spark.sql("SELECT channel, SUM(qty) AS units FROM orders GROUP BY channel")
    top.withColumn("units_k", F.round(F.col("units") / 1000, 2)).show()

    section("Parameterized SQL")
    min_qty = 5
    spark.sql(f"SELECT COUNT(*) AS big_orders FROM orders WHERE qty >= {min_qty}").show()

    section("Persist a SQL result as a Delta table (uses local warehouse)")
    result_df = spark.sql("""
        SELECT channel, COUNT(*) AS orders, SUM(qty) AS units
        FROM   orders GROUP BY channel
    """)
    result_df.write.mode("overwrite").saveAsTable("orders_by_channel")
    spark.sql("SELECT * FROM orders_by_channel ORDER BY orders DESC").show()
    print("tables in default db:", [t.name for t in spark.catalog.listTables() if t.tableType != "TEMPORARY"])

    stop_spark(spark)


if __name__ == "__main__":
    main()
