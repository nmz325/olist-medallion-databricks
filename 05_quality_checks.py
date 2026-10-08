# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Data quality checks
# MAGIC
# MAGIC Runs after gold and before the dashboard refresh and the ML notebook.
# MAGIC - Each check runs a SQL query that returns one value and evaluates it with a rule.
# MAGIC - All results are appended to `workspace.gold.dq_results` to track quality over time.
# MAGIC - If any check fails, the notebook raises an error **after** running all checks, so the Job stops and downstream tasks do not run.

# COMMAND ----------

from datetime import datetime

results = []

def check(name, sql, ok):
    """Runs a SQL query that returns one value and evaluates it with a rule."""
    value = spark.sql(sql).first()[0]
    passed = bool(ok(value))
    results.append((name, float(value) if value is not None else None, passed))
    print(f"{'OK  ' if passed else 'FAIL'} | {name}: {value}")

# COMMAND ----------

# --- Volume ---
check("silver.orders is not empty",
      "SELECT count(*) FROM workspace.silver.orders",
      lambda v: v > 0)

# --- Uniqueness ---
check("order_id is unique in silver.orders",
      "SELECT count(*) - count(DISTINCT order_id) FROM workspace.silver.orders",
      lambda v: v == 0)

check("one review per order in silver.order_reviews",
      "SELECT count(*) - count(DISTINCT order_id) FROM workspace.silver.order_reviews",
      lambda v: v == 0)

# --- Referential integrity ---
check("order items without an order",
      """SELECT count(*) FROM workspace.silver.order_items i
         LEFT JOIN workspace.silver.orders o ON i.order_id = o.order_id
         WHERE o.order_id IS NULL""",
      lambda v: v == 0)

check("order items without a product",
      """SELECT count(*) FROM workspace.silver.order_items i
         LEFT JOIN workspace.silver.products p ON i.product_id = p.product_id
         WHERE p.product_id IS NULL""",
      lambda v: v == 0)

check("order items without a seller",
      """SELECT count(*) FROM workspace.silver.order_items i
         LEFT JOIN workspace.silver.sellers s ON i.seller_id = s.seller_id
         WHERE s.seller_id IS NULL""",
      lambda v: v == 0)

# --- Validity ---
check("review scores outside 1-5",
      "SELECT count_if(review_score NOT BETWEEN 1 AND 5) FROM workspace.silver.order_reviews",
      lambda v: v == 0)

check("negative prices",
      "SELECT count_if(price < 0) FROM workspace.silver.order_items",
      lambda v: v == 0)

# --- Completeness ---
check("% customers with coordinates",
      """SELECT 100 * avg(CASE WHEN g.zip_prefix IS NOT NULL THEN 1 ELSE 0 END)
         FROM workspace.silver.customers c
         LEFT JOIN workspace.silver.geolocation g ON c.zip_prefix = g.zip_prefix""",
      lambda v: v >= 99)

# --- Consistency: same GMV from two fact tables ---
check("GMV difference fact_sales vs fact_orders",
      """SELECT abs((SELECT sum(total_value) FROM workspace.gold.fact_sales)
                  - (SELECT sum(order_value) FROM workspace.gold.fact_orders))""",
      lambda v: v < 0.01)

# --- Reconciliation: the GMV -> payments bridge closes to the cent ---
check("GMV to payments bridge gap",
      """WITH o AS (SELECT * FROM workspace.gold.fact_orders)
         SELECT abs(
             (SELECT sum(order_value) FROM o)
           + (SELECT coalesce(sum(paid_value), 0) FROM o WHERE items_qty IS NULL)
           + (SELECT coalesce(sum(paid_value - order_value), 0) FROM o WHERE items_qty IS NOT NULL)
           - (SELECT coalesce(sum(order_value), 0) FROM o WHERE items_qty IS NOT NULL AND paid_value IS NULL)
           - (SELECT sum(payment_value) FROM workspace.gold.fact_payments))""",
      lambda v: v < 0.01)

# COMMAND ----------

# Save this run's results to track data quality over time
run_ts = datetime.now()
(spark.createDataFrame([(run_ts, n, v, p) for n, v, p in results],
                       "run_ts timestamp, check_name string, value double, passed boolean")
      .write.mode("append")
      .saveAsTable("workspace.gold.dq_results"))

failed = [n for n, _, p in results if not p]
assert not failed, f"Data quality checks failed: {failed}"
print(f"All {len(results)} data quality checks passed")
