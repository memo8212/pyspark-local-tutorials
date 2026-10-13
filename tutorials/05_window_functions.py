"""05 - Window functions: ranking, running totals, lag/lead, dedup.

Run:  python tutorials/05_window_functions.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("05 - Window Functions")
    spark = get_spark("05-window-functions")
    from pyspark.sql import functions as F, Window

    sales = spark.createDataFrame(
        [
            ("2024-01", "NORTH", "Widget", 1200.0),
            ("2024-01", "WEST",  "Widget", 800.0),
            ("2024-01", "NORTH", "Gadget", 400.0),
            ("2024-02", "NORTH", "Widget", 900.0),
            ("2024-02", "WEST",  "Widget", 1500.0),
            ("2024-02", "EAST",  "Gadget", 300.0),
            ("2024-03", "WEST",  "Gadget", 1100.0),
        ],
        ["month", "region", "product", "revenue"],
    )
    sales.show()

    by_region = Window.partitionBy("region").orderBy(F.col("revenue").desc())

    section("Ranking: row_number / rank / dense_rank / percent_rank")
    (sales
     .withColumn("row_number", F.row_number().over(by_region))
     .withColumn("rank", F.rank().over(by_region))
     .withColumn("dense_rank", F.dense_rank().over(by_region))
     .withColumn("pct_rank", F.round(F.percent_rank().over(by_region), 2))
     .orderBy("region", "row_number")
     .show())

    section("Running total and moving average (rowsBetween)")
    running = (Window.partitionBy("region").orderBy("month")
               .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    movavg = Window.partitionBy("region").orderBy("month").rowsBetween(-1, 1)
    (sales
     .withColumn("running_rev", F.round(F.sum("revenue").over(running), 2))
     .withColumn("movavg_rev", F.round(F.avg("revenue").over(movavg), 2))
     .orderBy("region", "month")
     .show())

    section("lag / lead: month-over-month change")
    lagged = (sales
              .withColumn("prev_rev", F.lag("revenue", 1).over(Window.partitionBy("product").orderBy("month")))
              .withColumn("mom_change", F.round(F.col("revenue") - F.col("prev_rev"), 2)))
    lagged.orderBy("product", "month").show()

    section("Dedup pattern: keep latest row per key")
    events = spark.createDataFrame(
        [(1, "A", "2024-01-01"), (1, "A", "2024-01-03"), (2, "B", "2024-01-02")],
        ["id", "val", "updated_at"],
    ).withColumn("updated_at", F.to_date("updated_at"))
    latest = Window.partitionBy("id").orderBy(F.col("updated_at").desc())
    events.withColumn("rn", F.row_number().over(latest)).filter("rn = 1").drop("rn").show()

    section("Global window (no partition): share of total")
    (sales
     .withColumn("share", F.round(F.col("revenue") / F.sum("revenue").over(Window.partitionBy()), 3))
     .orderBy(F.col("revenue").desc())
     .show())

    stop_spark(spark)


if __name__ == "__main__":
    main()
