"""Create a local SparkSession with Delta Lake enabled.

Everything in this repo runs against this session — no cluster, no cloud.
"""
from __future__ import annotations

import os
import sys

from common.paths import WAREHOUSE_DIR


def get_spark(app_name: str = "pyspark-local", shuffle_partitions: int = 8):
    """Return a local SparkSession with the Delta Lake extension configured.

    Safe to call repeatedly: Spark reuses an existing session in the same process.
    """
    # Make Spark workers use *this* interpreter so pandas UDFs see pyarrow/numpy.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        # Delta Lake
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Small local data -> fewer shuffle tasks
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
        .config("spark.sql.catalogImplementation", "hive")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.streaming.schemaInference", "true")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def stop_spark(spark) -> None:
    try:
        spark.stop()
    except Exception:
        pass
