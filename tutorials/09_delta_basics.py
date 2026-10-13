"""09 - Delta Lake basics: ACID writes, DML, time travel, history.

Run:  python tutorials/09_delta_basics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("09 - Delta Lake Basics")
    spark = get_spark("09-delta-basics")

    table = "delta_demo"

    section("Create a Delta table (managed, in the local warehouse)")
    data = [(1, "Alice", "NORTH", 250.0), (2, "Bob", "WEST", 90.0), (3, "Carol", "NORTH", 1300.0)]
    df = spark.createDataFrame(data, ["id", "name", "region", "amount"])
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    spark.table(table).show()

    section("DML: UPDATE / DELETE / INSERT")
    spark.sql(f"UPDATE {table} SET amount = amount * 1.10 WHERE region = 'NORTH'")
    spark.sql(f"DELETE FROM {table} WHERE amount < 100")
    spark.sql(f"INSERT INTO {table} VALUES (4, 'Dan', 'EAST', 480.0)")
    spark.table(table).orderBy("id").show()

    section("MERGE (upsert)")
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW updates AS
        SELECT * FROM VALUES (2, 'Bob', 'WEST', 150.0), (5, 'Erin', 'WEST', 2600.0)
        AS t(id, name, region, amount)
    """)
    spark.sql(f"""
        MERGE INTO {table} AS t
        USING updates AS s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    spark.table(table).orderBy("id").show()

    section("History & time travel")
    spark.sql(f"DESCRIBE HISTORY {table}").select("version", "timestamp", "operation").show(truncate=False)
    v0 = spark.read.format("delta").option("versionAsOf", 0).table(table)
    print("version 0 (original rows):")
    v0.show()

    section("Schema evolution with mergeSchema")
    more = spark.createDataFrame([(6, "Fay", "EAST", 700.0, "GOLD")],
                                 ["id", "name", "region", "amount", "tier"])
    (more.write.format("delta").mode("append")
         .option("mergeSchema", "true").saveAsTable(table))
    spark.table(table).printSchema()

    section("OPTIMIZE (compact small files)")
    spark.sql(f"OPTIMIZE {table}")
    print("OPTIMIZE complete.")

    section("Table details")
    spark.sql(f"DESCRIBE DETAIL {table}").select("format", "numFiles", "sizeInBytes").show()

    section("Path-based Delta (no catalog) also works")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "output", "delta_by_path")
    df.write.format("delta").mode("overwrite").save(path)
    print("wrote path table:", path)
    spark.read.format("delta").load(path).show()

    stop_spark(spark)


if __name__ == "__main__":
    main()
