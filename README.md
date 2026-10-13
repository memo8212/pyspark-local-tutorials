# PySpark Local Tutorials

Learn **Apache Spark with PySpark entirely on your laptop** — no Databricks, no
cloud, no cluster. Every tutorial is a plain, runnable Python script covering
DataFrames, Spark SQL, Delta Lake, MLlib, structured streaming, performance
tuning and testing, using one shared local Spark + Delta session.

```
pip install -r requirements.txt
python run_all.py            # runs every tutorial in order
```

## Why this repo

- **Runs locally.** `pip install` + `python tutorials/02_....py`. Nothing else.
- **Complete.** Core Spark → SQL → Delta → MLlib → streaming → tuning → tests.
- **Self-contained.** One shared session (`common/session.py`), deterministic
  sample data (`common/data.py`), no external datasets.
- **Readable.** Each script prints section headers and shows results as it goes.

## Contents

| # | Script | Topic |
|---|---|---|
| 01 | `01_rdd_vs_dataframe.py` | RDDs vs DataFrames, laziness, actions vs transformations |
| 02 | `02_dataframe_basics.py` | create DataFrames, schemas, inspect, columns |
| 03 | `03_dataframe_transformations.py` | select/filter/withColumn/when/cast/sort/distinct |
| 04 | `04_aggregations_and_joins.py` | groupBy/agg, join types, unions, pivot |
| 05 | `05_window_functions.py` | ranking, running totals, lag/lead, dedup |
| 06 | `06_spark_sql.py` | temp views, SQL, CTEs, window SQL, mixing with Python |
| 07 | `07_schemas_and_types.py` | explicit schemas, StructType, casting, dates, nulls |
| 08 | `08_io_csv_json_parquet.py` | read/write CSV/JSON/Parquet, partitioning, options |
| 09 | `09_delta_basics.py` | Delta create/read/update/delete, time travel, history, optimize |
| 10 | `10_delta_merge_scd2.py` | MERGE upsert (SCD1), SCD2 history with effective dating |
| 11 | `11_udfs_and_pandas_udf.py` | Python UDFs, pandas UDFs (vectorized), when to avoid UDFs |
| 12 | `12_structured_streaming.py` | file-source stream, checkpoint, availableNow, aggregation |
| 13 | `13_ml_classification.py` | MLlib pipeline, logistic/GBT, evaluators, class imbalance |
| 14 | `14_ml_regression.py` | linear regression, feature engineering, RMSE/R2 |
| 15 | `15_ml_clustering_als.py` | KMeans clustering + ALS recommender |
| 16 | `16_performance_tuning.py` | partitions, caching, broadcast joins, explain, AQE |
| 17 | `17_testing_pyspark.py` | unit-testing Spark code with pytest (see also `tests/`) |
| 18 | `18_end_to_end_project.py` | Bronze→Silver→Gold + ML scoring on local Delta |

Extras: `tests/` (pytest), `common/` (session, data, helpers), `run_all.py`.

## Setup

### 1. Python + dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Java

Spark needs a JVM (Java 17 recommended for Spark 4, Java 11/17 for Spark 3.5):

```bash
java -version     # if missing, install a JDK (e.g. `sudo apt install openjdk-17-jdk`)
```

### 3. Run

```bash
python common/data.py                       # generate sample data (also done by run_all)
python tutorials/02_dataframe_basics.py     # run one tutorial
python run_all.py                           # run them all
python run_all.py 09 10 13                  # run a subset by prefix
pytest -q                                   # unit tests
make all                                    # data + all tutorials (if you have make)
```

### What each step does (and why)

Not sure why a step exists? [`docs/setup-explained.md`](docs/setup-explained.md)
walks through **every** setup step and **every** setting in
`common/session.py` — what it does and how it helps. Read it once if you want to
understand the machine under the hood.

## Version compatibility

`delta-spark` must match your PySpark major version, otherwise the Delta
extension fails to load:

| PySpark | delta-spark | Notes |
|---|---|---|
| 3.5.x | 3.x | Java 8/11/17 |
| 4.0–4.2 | 4.x | Java 17+ |

If Delta misbehaves, pin both, e.g. `pip install "pyspark==3.5.1" "delta-spark==3.2.0"`.

## How the shared session works

`common/session.py` builds a `local[*]` session with the Delta extension and a
local Hive metastore (so `saveAsTable` / `spark.sql` DDL work), then sets
`PYSPARK_PYTHON` so pandas UDFs find pyarrow. Every tutorial does:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.session import get_spark
spark = get_spark("tutorial-name")
```

## Layout

```
common/      session.py, data.py, paths.py, helpers.py
tutorials/   01..18 runnable scripts
tests/       pytest unit tests
docs/        setup-explained.md (what each setup step does + why)
data/        generated sample CSVs/JSON (git-ignored)
output/      generated outputs, e.g. Delta tables (git-ignored)
run_all.py   run every tutorial in order
```

## Notes & troubleshooting

- **`metastore_db`/`derby.log`** appear after runs; they're git-ignored. Delete
  them (and `spark-warehouse/`) to reset tables: `make clean`.
- **Too verbose?** Sessions set log level to WARN. Adjust in `common/session.py`.
- **Memory/threads:** `local[*]` uses all cores. Lower it in `session.py` if
  needed (`master("local[2]")`).
- **First Delta write is slow** (Ivy resolves the Delta jars once); subsequent
  runs are fast.

## License

MIT — see `LICENSE`.
