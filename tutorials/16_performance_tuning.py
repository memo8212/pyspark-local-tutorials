"""16 - Performance tuning: partitions, caching, broadcast joins, plans.

Run:  python tutorials/16_performance_tuning.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("16 - Performance Tuning")
    spark = get_spark("16-performance")
    from pyspark.sql import functions as F
    from pyspark.sql.functions import broadcast

    df = spark.range(0, 2_000_000).withColumn("g", (F.col("id") % 100).cast("int"))

    section("Partitions: how many, and why it matters")
    print("default parallelism :", spark.sparkContext.defaultParallelism)
    print("partitions (initial) :", df.rdd.getNumPartitions())
    df8 = df.repartition(8)
    print("after repartition(8) :", df8.rdd.getNumPartitions())
    dfc = df8.coalesce(2)
    print("after coalesce(2)    :", dfc.rdd.getNumPartitions())
    print("repartition = full shuffle (can increase); coalesce = merge (decrease only)")

    section("Partition by key for downstream grouping")
    by_key = df.repartition(16, "g")
    print("repartition(16, 'g') partitions:", by_key.rdd.getNumPartitions())

    section("Caching: reuse an expensive DataFrame")
    cached = df.filter("g < 10").select("id").cache()
    cached.count()                      # materialize
    print("cached RDDs:", spark.sparkContext._jsc.sc().getPersistentRDDs().size())
    cached.agg(F.count("*")).show()     # served from cache
    cached.unpersist()

    section("Broadcast join: small table joined to a big table")
    small = spark.createDataFrame([(i, f"name{i}") for i in range(100)], ["g", "label"])
    joined = df.join(broadcast(small), on="g", how="left")
    print("plan (look for BroadcastHashJoin):")
    joined.explain()
    joined.filter("g = 5").select("id", "label").show(3)

    section("Adaptive Query Execution (AQE) is on by default")
    print("spark.sql.adaptive.enabled =", spark.conf.get("spark.sql.adaptive.enabled"))
    print("AQE coalesces shuffle partitions, handles skew, and switches join strategies.")

    section("Predicate pushdown: filter early, scan less")
    df.write.mode("overwrite").parquet("/tmp/pyspark_perf_demo")
    scanned = spark.read.parquet("/tmp/pyspark_perf_demo").filter("g = 5")
    scanned.explain()   # PartitionFilters/PushedFilters appear in the scan
    print("filtered rows:", scanned.count())

    section("Avoid collect(); use explain() and aggregates")
    print("collect() pulls all data to the driver — never on big data.")
    print("Use df.explain() to inspect the physical plan instead.")

    section("Rules of thumb")
    for rule in [
        "Filter/project early; let Catalyst push predicates.",
        "Cache only DataFrames reused 2+ times.",
        "Broadcast tables < ~10 MB (or set spark.sql.autoBroadcastJoinThreshold).",
        "Target ~128 MB per partition; avoid millions of tiny files.",
        "Prefer built-ins/pandas UDFs over Python UDFs.",
        "Watch the Spark UI (enable spark.ui) for skew and spills.",
    ]:
        print(" -", rule)

    stop_spark(spark)


if __name__ == "__main__":
    main()
