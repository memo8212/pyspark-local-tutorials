"""Create a local SparkSession with Delta Lake enabled.

Everything in this repo runs against this session — no cluster, no cloud.
Every setting below is explained (what it does / how it helps) in
docs/setup-explained.md.
"""
from __future__ import annotations

import os
import sys

from common.paths import WAREHOUSE_DIR


def get_spark(app_name: str = "pyspark-local", shuffle_partitions: int = 8):
    """Return a local SparkSession with the Delta Lake extension configured.

    Safe to call repeatedly: Spark reuses an existing session in the same process.
    """
    # Point Spark's worker processes at *this* interpreter (the venv running the
    # script) rather than whatever `python3` is on PATH. Without this, workers
    # can't import pyarrow/numpy/pandas, so pandas UDFs and toPandas() break.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # `configure_spark_with_delta_pip` attaches the Delta jars for the installed
    # delta-spark version (so we don't hand-manage --packages).
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        # Run one in-JVM Spark using all local cores as executor threads:
        # real multi-threaded Spark with zero cluster setup.
        .master("local[*]")

        # --- Delta Lake: register the SQL extension + use the Delta catalog ---
        # Enables format("delta"), MERGE/UPDATE/DELETE, DESCRIBE HISTORY,
        # time travel and OPTIMIZE (tutorials 09, 10, 18).
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")

        # --- Performance for small local data ---
        # Spark defaults to 200 post-shuffle partitions; that's 200 tiny tasks on
        # a few thousand rows. 8 matches a laptop and keeps runs fast.
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        # Adaptive Query Execution: coalesces small partitions, picks join
        # strategies and handles skew at runtime. Keeps code fast at any size.
        .config("spark.sql.adaptive.enabled", "true")

        # --- SQL catalog so saveAsTable()/DDL work locally ---
        # Managed tables land under ./spark-warehouse; a local (Derby) Hive
        # metastore provides the catalog for `spark.sql("SELECT ...")`.
        .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
        .config("spark.sql.catalogImplementation", "hive")

        # --- Clean terminal output ---
        # Don't start the web UI (turn on + open :4040 when inspecting jobs).
        .config("spark.ui.enabled", "false")
        # Allow schema inference for streaming file sources while learning
        # (tutorial 12); production streams should pass an explicit schema.
        .config("spark.sql.streaming.schemaInference", "true")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    # Silence INFO logs so tutorial output stays readable.
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark(spark) -> None:
    try:
        spark.stop()
    except Exception:
        pass
