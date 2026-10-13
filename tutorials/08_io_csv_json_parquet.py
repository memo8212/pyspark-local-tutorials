"""08 - I/O: CSV, JSON, Parquet; write options and partitioning.

Run:  python tutorials/08_io_csv_json_parquet.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.data import generate_all  # noqa: E402
from common.helpers import banner, section  # noqa: E402
from common.paths import DATA_DIR, OUTPUT_DIR  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def _tree(path, max_files=12):
    out = []
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(root, f), path))
            if len(out) >= max_files:
                return out
    return out


def main() -> None:
    banner("08 - I/O: CSV, JSON, Parquet")
    generate_all()
    spark = get_spark("08-io")
    out = OUTPUT_DIR / "io_demo"
    out.mkdir(parents=True, exist_ok=True)

    section("Read CSV (with header + schema inference)")
    orders = spark.read.csv(str(DATA_DIR / "orders.csv"), header=True, inferSchema=True)
    orders.printSchema()
    orders.show(3)

    section("Read NDJSON")
    events = spark.read.json(str(DATA_DIR / "events.json"))
    events.printSchema()
    events.show(3)

    section("Write & read Parquet (columnar, typed, compressed)")
    pq = out / "orders.parquet"
    orders.write.mode("overwrite").parquet(str(pq))
    back = spark.read.parquet(str(pq))
    print("parquet schema types:", dict(back.dtypes))
    back.show(3)

    section("Write partitioned by a column (folder per value)")
    part = out / "orders_by_channel"
    orders.write.mode("overwrite").partitionBy("channel").parquet(str(part))
    print("layout:", _tree(part))
    # partition pruning: only reads matching folders
    spark.read.parquet(str(part)).filter("channel = 'web'").show(3)

    section("CSV write options (delimiter, header, compression)")
    csv_out = out / "orders.csv"
    (orders.write.mode("overwrite")
          .option("header", True)
          .option("sep", "|")
          .csv(str(csv_out)))
    print("csv files:", _tree(csv_out))
    spark.read.option("header", True).option("sep", "|").csv(str(csv_out)).show(3)

    section("Coalesce to control file count")
    (orders.coalesce(1).write.mode("overwrite").parquet(str(out / "orders_single")))
    print("single-file layout:", _tree(out / "orders_single"))

    section("Bucketing vs partitioning (concept)")
    print("partitionBy = directory per value (good for low-cardinality filters)")
    print("bucketBy    = hash files (good for joins/aggregations on high-cardinality keys)")

    stop_spark(spark)


if __name__ == "__main__":
    main()
