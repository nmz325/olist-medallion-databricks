# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Visual insights
# MAGIC
# MAGIC Six static charts built with **matplotlib** on top of the silver and gold layers. They complement the AI/BI dashboard with views a BI tool does not do well: a heatmap, a map built from raw coordinates, a review-score distribution, a concentration curve and a category scatter.
# MAGIC
# MAGIC - Aggregations run in **Spark SQL**; only small result sets are converted to pandas for plotting.
# MAGIC - Every chart is shown inline and saved as a PNG to the Unity Catalog volume `workspace.gold.charts`.
# MAGIC - Titles describe the chart; subtitles are computed from the data, so they stay correct if the data changes.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# Charts are saved to a Unity Catalog volume, downloadable from Catalog Explorer
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.gold.charts")
OUT = "/Volumes/workspace/gold/charts"

# One small palette for every chart: blue for data, gray for context, red-gray-blue for review scores
BLUE, BLUE_LIGHT, GRAY, INK, INK_2 = "#2a78d6", "#86b6ef", "#a3a29d", "#0b0b0b", "#52514e"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SCORE_COLORS = {1: "#e34948", 2: "#f0a09f", 3: "#d9d8d4", 4: "#86b6ef", 5: "#2a78d6"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ebeae6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "legend.frameon": False,
})

def add_titles(ax, title, subtitle):
    """Left-aligned title and subtitle above the plot area."""
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 34), textcoords="offset points",
                fontsize=15, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 14), textcoords="offset points",
                fontsize=10.5, color=INK_2, ha="left", va="bottom")

def save(fig, name):
    """Saves the chart to the volume and shows it in the notebook."""
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor="white")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Orders over time
# MAGIC How did demand evolve, and how big was the Black Friday peak compared with a normal day?

# COMMAND ----------

daily = spark.sql("""
    SELECT date_key, count(*) AS orders
    FROM workspace.gold.fact_orders
    WHERE order_status NOT IN ('canceled', 'unavailable')
      AND date_key BETWEEN '2017-01-01' AND '2018-08-31'
    GROUP BY date_key
""").toPandas()

daily["date_key"] = pd.to_datetime(daily["date_key"])
daily = daily.set_index("date_key").sort_index().asfreq("D", fill_value=0)
daily["avg_7d"] = daily["orders"].rolling(7, center=True).mean()

peak_day, peak = daily["orders"].idxmax(), int(daily["orders"].max())
typical = daily.loc["2017-10-01":"2017-11-15", "orders"].median()
is_black_friday = peak_day.month == 11 and 22 <= peak_day.day <= 28
peak_label = "Black Friday" if is_black_friday else "Peak day"

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(daily.index, daily["orders"], color=BLUE_LIGHT, lw=1, label="Daily orders")
ax.plot(daily.index, daily["avg_7d"], color=BLUE, lw=2, label="7-day average")
ax.scatter([peak_day], [peak], s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
ax.annotate(f"{peak_label}, {peak_day:%d %b %Y}\n{peak:,} orders ({peak / typical:.1f}x a typical day)",
            xy=(peak_day, peak), xytext=(-18, -4), textcoords="offset points",
            ha="right", va="top", fontsize=9.5, color=INK)
ax.set_ylim(0, peak * 1.12)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_ylabel("Orders per day")
ax.legend(loc="upper left")
add_titles(ax, "Daily orders, Jan 2017 – Aug 2018",
           f"{int(daily['orders'].sum()):,} orders · canceled and unavailable excluded")
save(fig, "Orders_over_time")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. When do customers buy?
# MAGIC Orders by weekday and hour of purchase. Useful to schedule campaigns, customer service shifts and system maintenance windows.

# COMMAND ----------

slots = spark.sql("""
    SELECT dayofweek(purchase_ts) AS dow, hour(purchase_ts) AS hour, count(*) AS orders
    FROM workspace.silver.orders
    WHERE purchase_ts IS NOT NULL
    GROUP BY 1, 2
""").toPandas()

# Spark: 1 = Sunday ... 7 = Saturday  ->  0 = Monday ... 6 = Sunday
slots["dow"] = (slots["dow"] + 5) % 7
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
grid = (slots.pivot(index="dow", columns="hour", values="orders")
             .reindex(index=range(7), columns=range(24)).fillna(0))

top_dow, top_hour = np.unravel_index(grid.values.argmax(), grid.shape)
weekend_share = grid.loc[[5, 6]].values.sum() / grid.values.sum()
night_share = grid.loc[:, 0:5].values.sum() / grid.values.sum()

fig, ax = plt.subplots(figsize=(12, 4.2))
mesh = ax.pcolormesh(grid.values, cmap=SEQ_BLUE, edgecolors="white", linewidth=1.5)
ax.invert_yaxis()
ax.grid(False)
ax.spines[["left", "bottom"]].set_visible(False)
ax.set_yticks(np.arange(7) + 0.5, DAYS)
ax.set_xticks(np.arange(0, 24, 2) + 0.5, [f"{h:02d}h" for h in range(0, 24, 2)])
ax.tick_params(length=0)
ax.set_xlabel("Hour of purchase")
cbar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.03)
cbar.set_label("Orders", color=INK_2)
cbar.outline.set_visible(False)
add_titles(ax, "Orders by weekday and hour",
           f"Busiest slot: {DAYS[top_dow]} {top_hour:02d}:00–{top_hour + 1:02d}:00 · "
           f"weekends {weekend_share:.0%} of orders · 00–06h {night_share:.0%}")
save(fig, "Orders_heatmap")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Delivery time across Brazil
# MAGIC A map built only from customer coordinates (silver geolocation, ~99.7% coverage), no shapefiles needed. Each hexagon shows the average delivery time of the orders inside it.

# COMMAND ----------

geo = spark.sql("""
    SELECT c.lat, c.lng, o.delivery_days
    FROM workspace.gold.fact_orders o
    JOIN workspace.gold.dim_customer c ON o.customer_unique_id = c.customer_unique_id
    WHERE o.order_status = 'delivered'
      AND o.delivery_days IS NOT NULL
      AND c.lat IS NOT NULL
""").toPandas()

CITIES = {"São Paulo": (-46.63, -23.55), "Rio de Janeiro": (-43.20, -22.91), "Brasília": (-47.88, -15.79),
          "Manaus": (-60.02, -3.10), "Fortaleza": (-38.54, -3.73), "Porto Alegre": (-51.23, -30.03)}

fig, ax = plt.subplots(figsize=(8.5, 8.5))
hb = ax.hexbin(geo["lng"], geo["lat"], C=geo["delivery_days"], reduce_C_function=np.mean,
               gridsize=70, mincnt=3, cmap=SEQ_BLUE, vmin=5, vmax=30,
               edgecolors="white", linewidths=0.3)
for city, (lng, lat) in CITIES.items():
    ax.scatter(lng, lat, s=14, color=INK, zorder=3)
    ax.annotate(city, (lng, lat), xytext=(5, 3), textcoords="offset points", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
ax.set_aspect("equal")
ax.axis("off")
cbar = fig.colorbar(hb, ax=ax, shrink=0.55, pad=0.01, extend="max")
cbar.set_label("Average delivery time (days)", color=INK_2)
cbar.outline.set_visible(False)
add_titles(ax, "Delivery time across Brazil",
           f"{len(geo):,} delivered orders · median {geo['delivery_days'].median():.0f} days · "
           "color capped at 30 days")
save(fig, "Delivery_map")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Review scores by delivery delay
# MAGIC The full distribution of 1–5 star reviews, not just the average: how fast does satisfaction collapse once an order is late?

# COMMAND ----------

rv = spark.sql("""
    SELECT o.delivery_delay_days, o.review_score
    FROM workspace.gold.fact_orders o
    WHERE o.order_status = 'delivered'
      AND o.delivery_delay_days IS NOT NULL
      AND o.review_score IS NOT NULL
""").toPandas()

BUCKETS = ["7+ days early", "1–6 days early", "On the promised day",
           "1–3 days late", "4–7 days late", "8–14 days late", "15+ days late"]
rv["bucket"] = pd.cut(rv["delivery_delay_days"], bins=[-np.inf, -7, -1, 0, 3, 7, 14, np.inf], labels=BUCKETS)
share = pd.crosstab(rv["bucket"], rv["review_score"], normalize="index").reindex(index=BUCKETS, columns=range(1, 6)).fillna(0)
counts = rv["bucket"].value_counts().reindex(BUCKETS)

bad_on_time = share.loc[BUCKETS[:3], [1, 2]].sum(axis=1).mul(counts[BUCKETS[:3]]).sum() / counts[BUCKETS[:3]].sum()
bad_very_late = share.loc["15+ days late", [1, 2]].sum()

fig, ax = plt.subplots(figsize=(12, 5))
y = np.arange(len(BUCKETS))
left = np.zeros(len(BUCKETS))
for score in range(1, 6):
    vals = share[score].values
    ax.barh(y, vals, left=left, height=0.68, color=SCORE_COLORS[score], edgecolor="white", linewidth=1.5)
    for yi, (l, v) in enumerate(zip(left, vals)):
        if v >= 0.06:
            ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if score in (1, 5) else INK)
    left += vals
for yi, n in enumerate(counts.values):
    ax.text(1.01, yi, f"n = {n:,}", va="center", fontsize=9, color=INK_2)
ax.set_yticks(y, BUCKETS)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.grid(False)
ax.spines[["left", "bottom"]].set_visible(False)
ax.tick_params(length=0)
ax.legend(handles=[Patch(color=SCORE_COLORS[s], label=f"{s} ★") for s in range(1, 6)],
          ncol=5, loc="lower left", bbox_to_anchor=(0, -0.16), handlelength=1.2)
add_titles(ax, "Review scores by delivery delay",
           f"1–2 star reviews: {bad_on_time:.0%} when on time or early vs {bad_very_late:.0%} when 15+ days late")
save(fig, "Reviews_by_delay")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Seller concentration
# MAGIC A Pareto (Lorenz-style) curve: what share of GMV do the biggest sellers generate? High concentration means the marketplace depends on a few key accounts.

# COMMAND ----------

sellers = spark.sql("""
    SELECT CAST(gmv AS DOUBLE) AS gmv
    FROM workspace.gold.kpi_sellers
    WHERE gmv > 0
""").toPandas()

g = np.sort(sellers["gmv"].values)[::-1]
pct_sellers = np.arange(1, len(g) + 1) / len(g)
pct_gmv = g.cumsum() / g.sum()
share_at = lambda p: pct_gmv[int(np.ceil(p * len(g))) - 1]

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
ax.text(0.62, 0.55, "Every seller equal", color=INK_2, fontsize=9, rotation=37)
ax.plot(pct_sellers, pct_gmv, color=BLUE, lw=2)
for p in (0.01, 0.10, 0.20):
    s = share_at(p)
    ax.scatter([p], [s], s=50, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"Top {p:.0%} of sellers → {s:.0%} of GMV", (p, s), xytext=(10, -14),
                textcoords="offset points", fontsize=9.5, color=INK)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel("Sellers, ranked by GMV")
ax.set_ylabel("Cumulative share of GMV")
add_titles(ax, "Seller concentration",
           f"{len(g):,} sellers · half of GMV comes from the top {pct_sellers[np.searchsorted(pct_gmv, 0.5)]:.0%}")
save(fig, "Seller_concentration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Product categories: late deliveries vs reviews
# MAGIC Each bubble is a category with at least 500 delivered orders. Orders are counted once per category (an order with two items of the same category counts once), so averages are not inflated by multi-item orders.

# COMMAND ----------

cats = spark.sql("""
    WITH order_category AS (
        SELECT DISTINCT s.order_id, p.category
        FROM workspace.gold.fact_sales s
        JOIN workspace.gold.dim_product p ON s.product_id = p.product_id
    ),
    quality AS (
        SELECT oc.category,
               count(*)                          AS orders,
               100 * avg(CAST(o.is_late AS INT)) AS pct_late,
               avg(o.review_score)               AS avg_review
        FROM order_category oc
        JOIN workspace.gold.fact_orders o ON oc.order_id = o.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY oc.category
    ),
    gmv AS (
        SELECT p.category, CAST(sum(s.total_value) AS DOUBLE) AS gmv
        FROM workspace.gold.fact_sales s
        JOIN workspace.gold.dim_product p ON s.product_id = p.product_id
        WHERE s.order_status = 'delivered'
        GROUP BY p.category
    )
    SELECT q.category, q.orders, q.pct_late, q.avg_review, g.gmv
    FROM quality q JOIN gmv g ON q.category = g.category
    WHERE q.orders >= 500 AND q.category <> 'unknown'
""").toPandas()

cats["label"] = cats["category"].str.replace("_", " ")
r = cats["pct_late"].corr(cats["avg_review"])
# Label only a few categories (biggest, worst and best rated) to keep the chart readable
to_label = set(cats.nlargest(4, "gmv").index) | set(cats.nsmallest(2, "avg_review").index) \
         | set(cats.nlargest(1, "avg_review").index)

fig, ax = plt.subplots(figsize=(11, 7))
ax.axvline(cats["pct_late"].median(), color=GRAY, lw=1, ls="--")
ax.axhline(cats["avg_review"].median(), color=GRAY, lw=1, ls="--")
ax.scatter(cats["pct_late"], cats["avg_review"], s=40 + 1400 * cats["gmv"] / cats["gmv"].max(),
           color=BLUE, alpha=0.7, edgecolor="white", linewidth=1.5)
for i in to_label:
    row = cats.loc[i]
    ax.annotate(row["label"], (row["pct_late"], row["avg_review"]), xytext=(7, 5),
                textcoords="offset points", fontsize=9, color=INK)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
ax.set_xlabel("Orders delivered late")
ax.set_ylabel("Average review score (1–5)")
add_titles(ax, "Late deliveries vs review score by product category",
           f"{len(cats)} categories with 500+ delivered orders · bubble size = GMV · "
           f"dashed lines = medians · correlation r = {r:.2f}")
save(fig, "Categories_late_vs_review")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the charts
# MAGIC The PNGs are in **Catalog → workspace → gold → Volumes → charts**. Select a file and use **Download**.

# COMMAND ----------

display(dbutils.fs.ls(OUT))
