# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: clean and standardize
# MAGIC
# MAGIC Turns the raw bronze copy into reliable tables, one per entity:
# MAGIC - Correct data types (timestamps, `DECIMAL` for money, integers).
# MAGIC - One row per key, deduplicated with `QUALIFY ROW_NUMBER()` (latest load wins).
# MAGIC - Normalized text (trim, case, accents) and standardized keys (zip codes padded to 5 digits).
# MAGIC - Explicit cleaning rules: no `not_defined` payments, installments >= 1, review scores 1-5, prices >= 0.
# MAGIC
# MAGIC Data quality checks for this layer run in `05_quality_checks`.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS workspace.silver;
# MAGIC
# MAGIC -- Reusable rule to normalize city names (lowercase, no spaces at the ends, no accents)
# MAGIC CREATE OR REPLACE FUNCTION workspace.silver.normalize_city(s STRING)
# MAGIC RETURNS STRING
# MAGIC RETURN translate(lower(trim(s)), 'áàâãäéèêëíìîïóòôõöúùûüç', 'aaaaaeeeeiiiiooooouuuuc');

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Orders: typed timestamps and delivery delay vs the promised date
# MAGIC CREATE OR REPLACE TABLE workspace.silver.orders AS
# MAGIC SELECT order_id, customer_id,
# MAGIC        lower(trim(order_status))                     AS order_status,
# MAGIC        to_timestamp(order_purchase_timestamp)        AS purchase_ts,
# MAGIC        to_timestamp(order_approved_at)               AS approved_ts,
# MAGIC        to_timestamp(order_delivered_customer_date)   AS delivered_ts,
# MAGIC        to_timestamp(order_estimated_delivery_date)   AS estimated_delivery_ts,
# MAGIC        datediff(to_timestamp(order_delivered_customer_date),
# MAGIC                 to_timestamp(order_estimated_delivery_date)) AS delivery_delay_days
# MAGIC FROM workspace.bronze.orders
# MAGIC WHERE order_id IS NOT NULL
# MAGIC QUALIFY row_number() OVER (PARTITION BY order_id ORDER BY _ingest_ts DESC) = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Customers: customer_id changes with every order; customer_unique_id is the real customer
# MAGIC CREATE OR REPLACE TABLE workspace.silver.customers AS
# MAGIC SELECT customer_id,
# MAGIC        customer_unique_id,
# MAGIC        lpad(trim(customer_zip_code_prefix), 5, '0')      AS zip_prefix,
# MAGIC        workspace.silver.normalize_city(customer_city)    AS city,
# MAGIC        upper(trim(customer_state))                       AS state
# MAGIC FROM workspace.bronze.customers
# MAGIC WHERE customer_id IS NOT NULL
# MAGIC QUALIFY row_number() OVER (PARTITION BY customer_id ORDER BY _ingest_ts DESC) = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.silver.sellers AS
# MAGIC SELECT seller_id,
# MAGIC        lpad(trim(seller_zip_code_prefix), 5, '0')     AS zip_prefix,
# MAGIC        workspace.silver.normalize_city(seller_city)   AS city,
# MAGIC        upper(trim(seller_state))                      AS state
# MAGIC FROM workspace.bronze.sellers
# MAGIC WHERE seller_id IS NOT NULL
# MAGIC QUALIFY row_number() OVER (PARTITION BY seller_id ORDER BY _ingest_ts DESC) = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE workspace.silver.order_items AS
# MAGIC SELECT order_id,
# MAGIC        try_cast(order_item_id AS INT)            AS order_item_id,
# MAGIC        product_id, seller_id,
# MAGIC        to_timestamp(shipping_limit_date)         AS shipping_limit_ts,
# MAGIC        try_cast(price AS DECIMAL(12,2))          AS price,
# MAGIC        try_cast(freight_value AS DECIMAL(12,2))  AS freight_value
# MAGIC FROM workspace.bronze.order_items
# MAGIC WHERE order_id IS NOT NULL
# MAGIC   AND try_cast(price AS DECIMAL(12,2)) >= 0
# MAGIC QUALIFY row_number() OVER (PARTITION BY order_id, order_item_id ORDER BY _ingest_ts DESC) = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Payments: drop 'not_defined' payments and force at least 1 installment
# MAGIC CREATE OR REPLACE TABLE workspace.silver.order_payments AS
# MAGIC SELECT order_id,
# MAGIC        try_cast(payment_sequential AS INT)                  AS payment_sequential,
# MAGIC        lower(trim(payment_type))                            AS payment_type,
# MAGIC        greatest(try_cast(payment_installments AS INT), 1)   AS installments,
# MAGIC        try_cast(payment_value AS DECIMAL(12,2))             AS payment_value
# MAGIC FROM workspace.bronze.order_payments
# MAGIC WHERE order_id IS NOT NULL
# MAGIC   AND lower(trim(payment_type)) <> 'not_defined'
# MAGIC QUALIFY row_number() OVER (PARTITION BY order_id, payment_sequential ORDER BY _ingest_ts DESC) = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Reviews: valid scores only, one-line comments, one review per order (latest answer)
# MAGIC CREATE OR REPLACE TABLE workspace.silver.order_reviews AS
# MAGIC SELECT review_id, order_id,
# MAGIC        try_cast(review_score AS INT)                         AS review_score,
# MAGIC        nullif(trim(review_comment_title), '')                AS comment_title,
# MAGIC        nullif(trim(regexp_replace(review_comment_message, '[\\r\\n]+', ' ')), '') AS comment_message,
# MAGIC        to_timestamp(review_creation_date)                    AS created_ts,
# MAGIC        to_timestamp(review_answer_timestamp)                 AS answered_ts
# MAGIC FROM workspace.bronze.order_reviews
# MAGIC WHERE order_id IS NOT NULL
# MAGIC   AND try_cast(review_score AS INT) BETWEEN 1 AND 5
# MAGIC QUALIFY row_number() OVER (PARTITION BY order_id
# MAGIC                            ORDER BY to_timestamp(review_answer_timestamp) DESC) = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Products: fix column typos and translate categories to English
# MAGIC CREATE OR REPLACE TABLE workspace.silver.products AS
# MAGIC SELECT p.product_id,
# MAGIC        coalesce(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
# MAGIC        try_cast(p.product_name_lenght AS INT)         AS name_length,
# MAGIC        try_cast(p.product_description_lenght AS INT)  AS description_length,
# MAGIC        try_cast(p.product_photos_qty AS INT)          AS photos_qty,
# MAGIC        try_cast(p.product_weight_g AS INT)            AS weight_g,
# MAGIC        try_cast(p.product_length_cm AS INT)           AS length_cm,
# MAGIC        try_cast(p.product_height_cm AS INT)           AS height_cm,
# MAGIC        try_cast(p.product_width_cm AS INT)            AS width_cm
# MAGIC FROM workspace.bronze.products p
# MAGIC LEFT JOIN workspace.bronze.category_translation t
# MAGIC        ON trim(p.product_category_name) = trim(t.product_category_name)
# MAGIC WHERE p.product_id IS NOT NULL
# MAGIC QUALIFY row_number() OVER (PARTITION BY p.product_id ORDER BY p._ingest_ts DESC) = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Geolocation: from ~1M rows to one row per zip prefix, coordinates inside Brazil only
# MAGIC CREATE OR REPLACE TABLE workspace.silver.geolocation AS
# MAGIC SELECT lpad(trim(geolocation_zip_code_prefix), 5, '0')                  AS zip_prefix,
# MAGIC        avg(try_cast(geolocation_lat AS DOUBLE))                         AS lat,
# MAGIC        avg(try_cast(geolocation_lng AS DOUBLE))                         AS lng,
# MAGIC        mode(workspace.silver.normalize_city(geolocation_city))          AS city,
# MAGIC        mode(upper(trim(geolocation_state)))                             AS state
# MAGIC FROM workspace.bronze.geolocation
# MAGIC WHERE try_cast(geolocation_lat AS DOUBLE) BETWEEN -34 AND 6
# MAGIC   AND try_cast(geolocation_lng AS DOUBLE) BETWEEN -74 AND -34
# MAGIC GROUP BY 1;
