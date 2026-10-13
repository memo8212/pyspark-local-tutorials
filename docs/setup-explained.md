# Setup, Explained

This page explains **what every setup step does and how it helps**, so the repo
isn't a black box. It covers the environment (Python, dependencies, Java, data)
and every configuration line in `common/session.py`.

Read it once; after that, `make setup && make all` is all you need.

---

## 0. Why a "setup" at all

Spark is a JVM application with a Python front-end. To run it **locally** we must
give it three things:

1. **A Python environment** with PySpark + the Delta extension + data libs.
2. **A Java runtime** (Spark runs on the JVM; PySpark is a client over a socket).
3. **A session configuration** that makes Delta, SQL tables and streaming work
   predictably on a laptop.

Each step below maps to one of those needs.

---

## 1. Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**What it does:** creates an isolated folder (`.venv/`) with its own Python and
site-packages.

**How it helps:**
- Keeps PySpark/Delta versions pinned to *this* project, so they can't clash with
  other tools on your machine.
- Makes the repo reproducible: delete `.venv/` and rebuild from
  `requirements.txt`.
- `PYSPARK_PYTHON` later points workers at this interpreter, so UDFs and
  libraries resolve consistently.

---

## 2. Dependencies (`requirements.txt`)

```bash
pip install -U pip
pip install -r requirements.txt
```

| Package | What it does | How it helps here |
|---|---|---|
| **pyspark** | Spark's Python API + bundled Spark runtime | The engine everything runs on |
| **delta-spark** | Delta Lake extension for Spark | ACID tables, `MERGE`, time travel (tutorials 09–10, 18) |
| **pandas** | Tabular Python library | `createDataFrame(pdf)`, pandas UDFs, driver-side maths |
| **numpy** | Array maths | Synthetic data + fast numeric examples |
| **pyarrow** | Columnar exchange format | Required for pandas UDFs and fast `toPandas()` |
| **pytest** | Test runner | Runs `tests/` and the patterns in tutorial 17 |

**How it helps:** one command installs a complete local lakehouse stack. No
cloud account, no cluster, no external data.

> `pyarrow` matters more than it looks: Spark moves pandas UDF data over Arrow,
> so without it vectorized UDFs fail with `No module named 'pyarrow'`.

---

## 3. Java (the JVM)

```bash
java -version   # install a JDK if missing
```

**What it does:** Spark itself is Scala/Java; PySpark starts a JVM and talks to
it over a local socket (Py4J). No JVM → no Spark.

**How it helps:** the JVM is where query planning (Catalyst), execution
(Tungsten) and the Delta transaction log actually run. Your Python code is a
client of it.

**Version note:** Spark 4.x needs **Java 17**; Spark 3.5 works with Java 11/17.
`common/session.py` doesn't pin this — the JVM is provided by your machine.

---

## 4. Version compatibility (read this before mixing versions)

`delta-spark` must match the PySpark **major** version, or the extension won't
load:

| PySpark | delta-spark | Java |
|---|---|---|
| 3.5.x | 3.x | 11 or 17 |
| 4.0–4.2 | 4.x | 17+ |

**How it helps:** a matching pair avoids cryptic `NoSuchMethod`/class-not-found
errors when Spark loads the Delta extension. If in doubt, pin both:
`pip install "pyspark==3.5.1" "delta-spark==3.2.0"`.

---

## 5. Generating sample data

```bash
python common/data.py
```

**What it does:** writes small, **deterministic** CSVs/JSON into `data/`
(`customers`, `products`, `orders`, `events`). `run_all.py` calls it first.

**How it helps:**
- **Deterministic** (fixed seeds) → every run and every tutorial produces the
  same numbers, so examples are stable and testable.
- **Self-contained** → no downloads, no licensing, works offline.
- **Small** → the whole repo runs in seconds and fits in memory, so you focus on
  *concepts*, not data wrangling.

---

## 6. The shared session (`common/session.py`)

Every tutorial calls `get_spark()`, which builds one configured `local[*]`
session. Here is **every** setting and why it's there.

### 6.1 `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON`

```python
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
```

**What it does:** tells Spark to launch worker processes with *this* interpreter
(the one running the script), not whatever `python3` is on `PATH`.

**How it helps:** workers then see the same `pyarrow`, `numpy`, `pandas` as your
venv. Without it, pandas UDFs (tutorial 11) and `toPandas()` can fail with
`ModuleNotFoundError` because the worker used a different Python.

### 6.2 `master("local[*]")`

**What it does:** runs Spark in a single JVM using all CPU cores as local
executor threads.

**How it helps:** you get real multi-threaded Spark with zero setup — no
standalone/worker processes to start. Lower it to `local[2]` if you want to
leave cores free (see `get_spark(shuffle_partitions=...)` and the README).

### 6.3 Delta extension + catalog

```python
.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
.config("spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog")
```

**What it does:** plugs Delta into Spark's SQL layer and replaces the default
table catalog with Delta's.

**How it helps:** enables `format("delta")`, `MERGE`, `UPDATE`/`DELETE`,
`DESCRIBE HISTORY`, time travel and `OPTIMIZE` — i.e. tutorials 09, 10, 18 work
like they do on a managed platform.

### 6.4 `spark.sql.shuffle.partitions = 8`

**What it does:** sets the number of partitions used after a shuffle
(joins/aggregations). Spark's default is **200**.

**How it helps:** 200 partitions on a few thousand rows means 200 tiny tasks and
lots of scheduling overhead — locally that's slow. `8` matches a laptop and the
small datasets. On a real cluster you'd leave it high (or let AQE handle it).

### 6.5 `spark.sql.adaptive.enabled = true`

**What it does:** turns on Adaptive Query Execution (AQE), which re-plans based
on runtime statistics.

**How it helps:** AQE **coalesces** small shuffle partitions, **switches** join
strategies (e.g. to broadcast), and **handles skew** automatically — so the same
code stays reasonably fast whether you have 1k or 1M rows. It's on by default in
modern Spark; setting it explicitly documents the intent.

### 6.6 `spark.sql.warehouse.dir` + Hive metastore

```python
.config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
.config("spark.sql.catalogImplementation", "hive")
```

**What it does:** points managed-table storage at `./spark-warehouse/` and uses a
local (Derby) Hive metastore so `saveAsTable(...)` and SQL DDL have a catalog.

**How it helps:** `df.write.saveAsTable("orders_by_channel")` and
`spark.sql("SELECT * FROM orders_by_channel")` work locally (tutorials 06, 09,
10). The metastore lives in `./metastore_db` + `derby.log`; delete them (and
`spark-warehouse/`) with `make clean` to reset tables.

### 6.7 `spark.ui.enabled = false`

**What it does:** doesn't start the Spark web UI.

**How it helps:** avoids a background HTTP server and port usage for scripts you
run in a terminal. Turn it back on (`true`) and open `localhost:4040` when you
want to inspect jobs, stages and skew.

### 6.8 `spark.sql.streaming.schemaInference = true`

**What it does:** allows inferring schema for streaming file sources when no
schema is given.

**How it helps:** convenient while learning structured streaming (tutorial 12).
In production you'd usually pass an **explicit** schema for stability — the
tutorial shows both.

### 6.9 `sparkContext.setLogLevel("WARN")`

**What it does:** suppresses Spark's INFO chatter.

**How it helps:** tutorial output stays readable; only warnings/errors show.

---

## 7. Running the tutorials

```bash
python tutorials/02_dataframe_basics.py   # one tutorial
python run_all.py                         # all, each in its own process
python run_all.py 09 10 13                # a subset by prefix
pytest -q                                 # the test suite
make all                                  # data + all tutorials
```

**`run_all.py` runs each script in a **separate process**.** How it helps: each
tutorial gets a clean JVM/session, so one script's cached state or stopped
session can't affect the next, and a failure is isolated and easy to spot.

**`pytest`** reuses one session for the whole suite (via `conftest.py`), which is
faster than starting Spark per test. `make clean` removes `spark-warehouse/`,
`metastore_db/`, `derby.log`, `output/` and caches.

---

## 8. How all of this helps (summary)

| Setup step | Enables |
|---|---|
| venv + requirements | reproducible local Spark + Delta stack |
| Java/JDK | the engine Spark actually runs on |
| matching versions | Delta extension loads cleanly |
| `common/data.py` | deterministic, offline, tiny datasets |
| `PYSPARK_PYTHON` | pandas UDFs & `toPandas()` work |
| Delta extension/catalog | ACID tables, MERGE, time travel |
| shuffle=8 + AQE | fast, sane performance on a laptop |
| Hive metastore + warehouse | SQL DDL and `saveAsTable` |
| UI off + WARN logs | clean, quiet terminal output |
| `run_all.py` process isolation | reliable, debuggable runs |

---

## 9. Troubleshooting map

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named 'pyarrow'` in a UDF | worker used a different Python | ensure `PYSPARK_PYTHON=sys.executable` (already set in `session.py`) |
| Delta class/`NoSuchMethod` errors | `delta-spark` ↔ PySpark mismatch | pin matching versions (section 4) |
| `UnsupportedOperationException` on Windows / missing `winutils` | no Hadoop winutils | set `HADOOP_HOME` + `winutils.exe`, or use WSL |
| `Path does not exist` for a stream | source dir not created before `readStream` | create the dir / drop a file first (see tutorial 12) |
| `Table ... does not support REPLACE TABLE AS SELECT` | local catalog limitation | write a DataFrame with `saveAsTable` instead of CTAS (see tutorial 06) |
| Slow first Delta write | Ivy resolves Delta jars once | subsequent runs are fast |
| Stale tables/errors after experiments | old metastore | `make clean` |
| `Java gateway process exited` | no/incompatible JVM | install Java 17 (Spark 4) |

---

## 10. Where to look next

- `common/session.py` — the configuration described above, with inline comments.
- `common/data.py` — the deterministic data generator.
- `README.md` — quick start and the tutorial index.
