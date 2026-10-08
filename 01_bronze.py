# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze ingestion
# MAGIC
# MAGIC Loads the 9 Olist CSV files from the Unity Catalog Volume into bronze Delta tables.
# MAGIC - Every column is read as **string** (schema-on-read): no row is lost to a failed cast; typing happens in silver.
# MAGIC - Two audit columns are added: `_ingest_ts` (load time) and `_source_file` (origin file).
# MAGIC - The load is **idempotent**: `overwrite` replaces each table, so running it twice never duplicates rows.

# COMMAND ----------

from pyspark.sql import functions as F

CATALOG = "workspace"
BASE_PATH = f"/Volumes/{CATALOG}/landing/olist_raw"

# Bronze table name -> source file in the Volume
FILES = {
    "customers":            "olist_customers_dataset.csv",
    "orders":               "olist_orders_dataset.csv",
    "order_items":          "olist_order_items_dataset.csv",
    "order_payments":       "olist_order_payments_dataset.csv",
    "order_reviews":        "olist_order_reviews_dataset.csv",
    "products":             "olist_products_dataset.csv",
    "sellers":              "olist_sellers_dataset.csv",
    "geolocation":          "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")

# COMMAND ----------

# Fail fast if a source file is missing from the Volume
available = {f.name for f in dbutils.fs.ls(BASE_PATH)}
missing = [f for f in FILES.values() if f not in available]
assert not missing, f"Missing files in {BASE_PATH}: {missing}"
print(f"All {len(FILES)} source files found")

# COMMAND ----------

for table, file in FILES.items():
    (spark.read
          .option("header", True)          # first row = column names
          .option("multiLine", True)       # review comments contain line breaks
          .option("escape", '"')           # quotes inside text fields
          .csv(f"{BASE_PATH}/{file}")      # no inferSchema: everything stays as string
          .withColumn("_ingest_ts", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path"))
          .write.mode("overwrite")
          .saveAsTable(f"{CATALOG}.bronze.{table}"))
    print(f"Loaded {CATALOG}.bronze.{table}")

# COMMAND ----------

# Control totals: every bronze table must have rows
counts = {t: spark.table(f"{CATALOG}.bronze.{t}").count() for t in FILES}
for t, n in counts.items():
    print(f"{t:22} {n:>10,}")

empty = [t for t, n in counts.items() if n == 0]
assert not empty, f"Empty bronze tables: {empty}"
