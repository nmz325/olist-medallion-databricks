# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: star schema and KPIs
# MAGIC
# MAGIC Builds the business layer as a star schema:
# MAGIC - **Dimensions:** `dim_date`, `dim_customer` (real customer = `customer_unique_id`), `dim_seller`, `dim_product`.
# MAGIC - **Facts, one grain each:** `fact_sales` (one row per item), `fact_orders` (one row per order), `fact_payments` (one row per payment).
# MAGIC - **KPI tables** for the dashboard: `kpi_ventas_mensuales`, `kpi_sellers`.
# MAGIC - **Informational PK/FK constraints**, added at the end because `CREATE OR REPLACE` drops them.
# MAGIC
# MAGIC Automated checks and the GMV-to-payments reconciliation run in `05_quality_checks`.

# COMMAND ----------

# Drop foreign keys first: a dimension cannot be replaced while a fact table points to it
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.gold")

fks = {
    "fact_sales":  ["fk_sales_customer", "fk_sales_seller", "fk_sales_product", "fk_sales_date"],
    "fact_orders": ["fk_orders_customer", "fk_orders_date"],
}
for table, names in fks.items():
    if spark.catalog.tableExists(f"workspace.gold.{table}"):
        for name in names:
            spark.sql(f"ALTER TABLE workspace.gold.{table} DROP CONSTRAINT IF EXISTS {name}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Calendar dimension: one row per day, even days without sales
# MAGIC CREATE OR REPLACE TABLE workspace.gold.dim_date AS
# MAGIC SELECT d AS date_key, year(d) AS year, quarter(d) AS quarter, month(d) AS month,
# MAGIC        date_format(d, 'yyyy-MM') AS year_month, date_format(d, 'MMMM') AS month_name,
# MAGIC        dayofweek(d) AS day_of_week, date_format(d, 'EEEE') AS day_name,
# MAGIC        dayofweek(d) IN (1, 7) AS is_weekend
# MAGIC FROM (SELECT explode(sequence(DATE'2016-01-01', DATE'2018-12-31', INTERVAL 1 DAY)) AS d);
# MAGIC
# MAGIC -- Real customer (customer_unique_id) with the latest known address
# MAGIC CREATE OR REPLACE TABLE workspace.gold.dim_customer AS
# MAGIC WITH base AS (
# MAGIC   SELECT c.customer_unique_id, c.zip_prefix, c.city, c.state,
# MAGIC          min(o.purchase_ts) OVER (PARTITION BY c.customer_unique_id) AS first_purchase_ts,
# MAGIC          count(*)           OVER (PARTITION BY c.customer_unique_id) AS total_orders,
# MAGIC          row_number()       OVER (PARTITION BY c.customer_unique_id ORDER BY o.purchase_ts DESC) AS rn
# MAGIC   FROM workspace.silver.customers c
# MAGIC   JOIN workspace.silver.orders o ON c.customer_id = o.customer_id)
# MAGIC SELECT b.customer_unique_id, b.zip_prefix, b.city, b.state, g.lat, g.lng,
# MAGIC        b.first_purchase_ts, b.total_orders, b.total_orders > 1 AS is_repeat_customer
# MAGIC FROM base b
# MAGIC LEFT JOIN workspace.silver.geolocation g ON b.zip_prefix = g.zip_prefix
# MAGIC WHERE b.rn = 1;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.gold.dim_seller AS
# MAGIC SELECT s.seller_id, s.zip_prefix, s.city, s.state, g.lat, g.lng
# MAGIC FROM workspace.silver.sellers s
# MAGIC LEFT JOIN workspace.silver.geolocation g ON s.zip_prefix = g.zip_prefix;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.gold.dim_product AS
# MAGIC SELECT *, length_cm * height_cm * width_cm AS volume_cm3
# MAGIC FROM workspace.silver.products;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Fact: one row per item sold
# MAGIC CREATE OR REPLACE TABLE workspace.gold.fact_sales AS
# MAGIC SELECT i.order_id, i.order_item_id, c.customer_unique_id, i.seller_id, i.product_id,
# MAGIC        CAST(o.purchase_ts AS DATE) AS date_key, o.order_status,
# MAGIC        i.price, i.freight_value, i.price + i.freight_value AS total_value
# MAGIC FROM workspace.silver.order_items i
# MAGIC JOIN workspace.silver.orders o    ON i.order_id = o.order_id
# MAGIC JOIN workspace.silver.customers c ON o.customer_id = c.customer_id;
# MAGIC
# MAGIC -- Fact: one row per order (value, delivery, review)
# MAGIC CREATE OR REPLACE TABLE workspace.gold.fact_orders AS
# MAGIC WITH items AS (
# MAGIC   SELECT order_id, count(*) AS items_qty, count(DISTINCT seller_id) AS sellers_qty,
# MAGIC          sum(price) AS items_value, sum(freight_value) AS freight_value
# MAGIC   FROM workspace.silver.order_items GROUP BY order_id),
# MAGIC pay AS (
# MAGIC   SELECT order_id, sum(payment_value) AS paid_value, max(installments) AS max_installments,
# MAGIC          max_by(payment_type, payment_value) AS main_payment_type, count(*) AS payments_qty
# MAGIC   FROM workspace.silver.order_payments GROUP BY order_id)
# MAGIC SELECT o.order_id, c.customer_unique_id, CAST(o.purchase_ts AS DATE) AS date_key, o.order_status,
# MAGIC        i.items_qty, i.sellers_qty, i.items_value, i.freight_value,
# MAGIC        i.items_value + i.freight_value AS order_value,
# MAGIC        p.paid_value, p.main_payment_type, p.max_installments, p.payments_qty,
# MAGIC        datediff(o.delivered_ts, o.purchase_ts) AS delivery_days,
# MAGIC        o.delivery_delay_days,
# MAGIC        o.delivery_delay_days > 0 AS is_late,
# MAGIC        r.review_score,
# MAGIC        r.review_score <= 2 AS is_bad_review
# MAGIC FROM workspace.silver.orders o
# MAGIC JOIN workspace.silver.customers c ON o.customer_id = c.customer_id
# MAGIC LEFT JOIN items i ON o.order_id = i.order_id
# MAGIC LEFT JOIN pay p   ON o.order_id = p.order_id
# MAGIC LEFT JOIN workspace.silver.order_reviews r ON o.order_id = r.order_id;
# MAGIC
# MAGIC -- Fact: one row per payment
# MAGIC CREATE OR REPLACE TABLE workspace.gold.fact_payments AS
# MAGIC SELECT p.order_id, p.payment_sequential, c.customer_unique_id,
# MAGIC        CAST(o.purchase_ts AS DATE) AS date_key,
# MAGIC        p.payment_type, p.installments, p.payment_value,
# MAGIC        round(p.payment_value / p.installments, 2) AS installment_value
# MAGIC FROM workspace.silver.order_payments p
# MAGIC JOIN workspace.silver.orders o    ON p.order_id = o.order_id
# MAGIC JOIN workspace.silver.customers c ON o.customer_id = c.customer_id;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- KPI tables for the dashboard (canceled and unavailable orders excluded)
# MAGIC CREATE OR REPLACE TABLE workspace.gold.kpi_ventas_mensuales AS
# MAGIC SELECT CAST(date_trunc('MONTH', date_key) AS DATE) AS month,
# MAGIC        count(*)                                  AS orders,
# MAGIC        count(DISTINCT customer_unique_id)        AS customers,
# MAGIC        sum(order_value)                          AS gmv,
# MAGIC        round(avg(order_value), 2)                AS aov,
# MAGIC        round(100 * avg(CAST(is_late AS INT)), 2) AS pct_late,
# MAGIC        round(avg(review_score), 2)               AS avg_review
# MAGIC FROM workspace.gold.fact_orders
# MAGIC WHERE order_status NOT IN ('canceled', 'unavailable')
# MAGIC GROUP BY 1;
# MAGIC
# MAGIC CREATE OR REPLACE TABLE workspace.gold.kpi_sellers AS
# MAGIC SELECT s.seller_id, d.state,
# MAGIC        count(DISTINCT s.order_id)            AS orders,
# MAGIC        sum(s.total_value)                    AS gmv,
# MAGIC        round(avg(o.delivery_delay_days), 1)  AS avg_delay_days,
# MAGIC        round(avg(o.review_score), 2)         AS avg_review
# MAGIC FROM workspace.gold.fact_sales s
# MAGIC JOIN workspace.gold.fact_orders o ON s.order_id = o.order_id
# MAGIC JOIN workspace.gold.dim_seller d  ON s.seller_id = d.seller_id
# MAGIC GROUP BY 1, 2;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Informational primary and foreign keys (documentation for BI tools and Genie)
# MAGIC ALTER TABLE workspace.gold.dim_customer ALTER COLUMN customer_unique_id SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_customer ADD CONSTRAINT pk_customer PRIMARY KEY (customer_unique_id);
# MAGIC ALTER TABLE workspace.gold.dim_seller   ALTER COLUMN seller_id SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_seller   ADD CONSTRAINT pk_seller PRIMARY KEY (seller_id);
# MAGIC ALTER TABLE workspace.gold.dim_product  ALTER COLUMN product_id SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_product  ADD CONSTRAINT pk_product PRIMARY KEY (product_id);
# MAGIC ALTER TABLE workspace.gold.dim_date     ALTER COLUMN date_key SET NOT NULL;
# MAGIC ALTER TABLE workspace.gold.dim_date     ADD CONSTRAINT pk_date PRIMARY KEY (date_key);
# MAGIC
# MAGIC ALTER TABLE workspace.gold.fact_sales  ADD CONSTRAINT fk_sales_customer  FOREIGN KEY (customer_unique_id) REFERENCES workspace.gold.dim_customer;
# MAGIC ALTER TABLE workspace.gold.fact_sales  ADD CONSTRAINT fk_sales_seller    FOREIGN KEY (seller_id)  REFERENCES workspace.gold.dim_seller;
# MAGIC ALTER TABLE workspace.gold.fact_sales  ADD CONSTRAINT fk_sales_product   FOREIGN KEY (product_id) REFERENCES workspace.gold.dim_product;
# MAGIC ALTER TABLE workspace.gold.fact_sales  ADD CONSTRAINT fk_sales_date      FOREIGN KEY (date_key)   REFERENCES workspace.gold.dim_date;
# MAGIC ALTER TABLE workspace.gold.fact_orders ADD CONSTRAINT fk_orders_customer FOREIGN KEY (customer_unique_id) REFERENCES workspace.gold.dim_customer;
# MAGIC ALTER TABLE workspace.gold.fact_orders ADD CONSTRAINT fk_orders_date     FOREIGN KEY (date_key) REFERENCES workspace.gold.dim_date;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reconciliation (exploration)
# MAGIC The same money computed three ways. Expected: both GMVs equal (15,843,553.24) and total paid 16,008,872.12.
# MAGIC The gap is explained by payments on orders without items (canceled or unavailable) and installment interest.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   (SELECT sum(total_value)   FROM workspace.gold.fact_sales)    AS gmv_items,
# MAGIC   (SELECT sum(order_value)   FROM workspace.gold.fact_orders)   AS gmv_orders,
# MAGIC   (SELECT sum(payment_value) FROM workspace.gold.fact_payments) AS total_paid,
# MAGIC   (SELECT count(*) FROM workspace.gold.fact_orders WHERE items_qty IS NULL) AS orders_without_items;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Where does the gap come from?
# MAGIC SELECT
# MAGIC   round(sum(CASE WHEN items_qty IS NULL THEN paid_value END), 2)                  AS paid_on_orders_without_items,
# MAGIC   round(sum(CASE WHEN items_qty IS NOT NULL THEN paid_value - order_value END), 2) AS net_gap_on_orders_with_items,
# MAGIC   count_if(items_qty IS NOT NULL AND paid_value > order_value + 0.01)             AS overpaid_orders,
# MAGIC   count_if(paid_value IS NULL)                                                     AS orders_without_payment
# MAGIC FROM workspace.gold.fact_orders;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Do overpaid orders use more installments? (supports the interest hypothesis)
# MAGIC SELECT paid_value > order_value + 0.01 AS overpaid,
# MAGIC        count(*) AS orders, round(avg(max_installments), 1) AS avg_installments
# MAGIC FROM workspace.gold.fact_orders
# MAGIC WHERE items_qty IS NOT NULL AND paid_value IS NOT NULL
# MAGIC GROUP BY 1;
