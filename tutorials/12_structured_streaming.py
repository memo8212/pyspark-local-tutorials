"""12 - Structured Streaming: file source, checkpoint, availableNow trigger.

Run:  python tutorials/12_structured_streaming.py
"""
import json
import os
import shutil
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.paths import OUTPUT_DIR  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402

IN = OUTPUT_DIR / "stream_in"
OUT = OUTPUT_DIR / "stream_out"
CKPT = OUTPUT_DIR / "stream_ckpt"


def drop_events(path, day_offset, n, tag):
    os.makedirs(path, exist_ok=True)
    d = (date(2024, 1, 1) + timedelta(days=day_offset)).strftime("%Y-%m-%d")
    f = os.path.join(path, f"events_{tag}.json")
    with open(f, "w") as fh:
        for i in range(n):
            fh.write(json.dumps({
                "event_id": f"{tag}-{i}",
                "event_type": ["view", "add_to_cart", "purchase"][i % 3],
                "event_date": d,
                "amount": round((i + 1) * 1.5, 2),
            }) + "\n")
    print("dropped:", f)


def main() -> None:
    banner("12 - Structured Streaming")
    for p in (IN, OUT, CKPT):
        shutil.rmtree(p, ignore_errors=True)
    os.makedirs(IN, exist_ok=True)
    # First batch must exist before the stream is defined (the source path is
    # validated at definition time).
    drop_events(IN, 0, 6, "b1")

    spark = get_spark("12-streaming")
    from pyspark.sql import functions as F
    from pyspark.sql.types import (StructType, StructField, StringType, DoubleType)

    schema = StructType([
        StructField("event_id", StringType()),
        StructField("event_type", StringType()),
        StructField("event_date", StringType()),
        StructField("amount", DoubleType()),
    ])

    section("Define a streaming read over a directory (schema required)")
    stream = (
        spark.readStream.schema(schema).option("maxFilesPerTrigger", 1)
        .json(str(IN))
        .withColumn("event_date", F.to_date("event_date"))
    )
    print("isStreaming:", stream.isStreaming)

    section("Batch 1 -> append to a Delta sink with a checkpoint")
    q1 = (stream.writeStream.format("delta").outputMode("append")
          .option("checkpointLocation", str(CKPT))
          .trigger(availableNow=True)
          .start(str(OUT)))
    q1.awaitTermination()
    print("after batch 1:", spark.read.format("delta").load(str(OUT)).count(), "rows")

    section("Batch 2 -> only NEW files are processed (checkpoint remembers)")
    drop_events(IN, 1, 4, "b2")
    q2 = (stream.writeStream.format("delta").outputMode("append")
          .option("checkpointLocation", str(CKPT))
          .trigger(availableNow=True)
          .start(str(OUT)))
    q2.awaitTermination()
    total = spark.read.format("delta").load(str(OUT))
    print("after batch 2:", total.count(), "rows")
    total.orderBy("event_date", "event_id").show(20, truncate=False)

    section("Streaming aggregation (complete mode) — running totals by type")
    agg_out = OUTPUT_DIR / "stream_agg"
    agg_ckpt = OUTPUT_DIR / "stream_agg_ckpt"
    shutil.rmtree(agg_out, ignore_errors=True)
    shutil.rmtree(agg_ckpt, ignore_errors=True)
    q3 = (stream.groupBy("event_type")
          .agg(F.count("*").alias("events"), F.round(F.sum("amount"), 2).alias("amount"))
          .writeStream.format("delta").outputMode("complete")
          .option("checkpointLocation", str(agg_ckpt))
          .trigger(availableNow=True)
          .start(str(agg_out)))
    q3.awaitTermination()
    spark.read.format("delta").load(str(agg_out)).orderBy("event_type").show()

    section("Notes")
    print("- Sources: files, Kafka, Delta (change data feed), sockets, rate.")
    print("- Triggers: availableNow (batch-like), processingTime, once, continuous.")
    print("- Output modes: append, update, complete (aggregations).")
    print("- Always give each stream a UNIQUE checkpoint location.")

    stop_spark(spark)


if __name__ == "__main__":
    main()
