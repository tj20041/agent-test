# Databricks Notebook - Test Case 2
# FIXED: Guarded all division operations against zero divisors to avoid
# SparkArithmeticException [DIVIDE_BY_ZERO] under ANSI SQL mode.

from pyspark.sql.functions import col, sum, avg, row_number, rank, lag, lead, when, lit
from pyspark.sql.window import Window

# Create sales data with zero values
sales_data = [
    (1, "Store_A", "2024-01-01", 5, 1000.00),
    (2, "Store_A", "2024-01-02", 3, 500.00),
    (3, "Store_A", "2024-01-03", 0, 0.00),      # Zero quantity and amount
    (4, "Store_B", "2024-01-01", 8, 2000.00),
    (5, "Store_B", "2024-01-02", 0, 0.00),      # Zero quantity and amount
    (6, "Store_B", "2024-01-03", 4, 800.00),
    (7, "Store_C", "2024-01-01", 2, 300.00),
    (8, "Store_C", "2024-01-02", 1, 100.00),
    (9, "Store_C", "2024-01-03", 3, 600.00)
]

df = spark.createDataFrame(sales_data, ["id", "store", "date", "quantity", "amount"])

# Create window specification
window_spec = Window.partitionBy("store").orderBy("date")

# FIXED: Guarded division in window function calculations - avoid divide-by-zero
# by checking the denominator with when()/otherwise() before dividing.
df_with_metrics = df.withColumn(
    "running_avg_quantity",
    avg("quantity").over(window_spec)
).withColumn(
    "running_avg_amount",
    avg("amount").over(window_spec)
).withColumn(
    # FIXED: null-safe ratio_to_avg - returns NULL instead of raising when
    # running_avg_amount is 0
    "ratio_to_avg",
    when(col("running_avg_amount") != 0, col("amount") / col("running_avg_amount")).otherwise(None)
).withColumn(
    # FIXED: null-safe quantity_ratio - returns NULL instead of raising when
    # running_avg_quantity is 0
    "quantity_ratio",
    when(col("running_avg_quantity") != 0, col("quantity") / col("running_avg_quantity")).otherwise(None)
)

# Additional calculation - already null-safe using when()/otherwise()
df_with_metrics = df_with_metrics.withColumn(
    "avg_price_per_unit",
    when(col("quantity") > 0, col("amount") / col("quantity"))
    .otherwise(0)
)

# FIXED: growth_rate now guards against a zero (or null) previous amount value
prev_amount = lag("amount", 1).over(window_spec)
df_with_metrics = df_with_metrics.withColumn(
    "growth_rate",
    when(
        prev_amount.isNotNull() & (prev_amount != 0),
        (col("amount") - prev_amount) / prev_amount
    ).otherwise(None)
)

# Force execution - no longer throws ArithmeticException
df_with_metrics.show()

# Group by with null-safe division to avoid divide-by-zero for stores with zero quantity
result = df_with_metrics.groupBy("store").agg(
    sum("amount").alias("total_amount"),
    sum("quantity").alias("total_quantity")
).withColumn(
    "avg_price",
    when(col("total_quantity") != 0, col("total_amount") / col("total_quantity")).otherwise(None)
)

result.show()
