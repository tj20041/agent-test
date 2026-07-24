# Databricks Notebook - Test Case 2
# ERROR: ArithmeticException when dividing by zero in window calculations
# Expected log error: "java.lang.ArithmeticException: / by zero" or "Division by zero"

from pyspark.sql.functions import col, sum, avg, row_number, rank, lag, lead, when
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

# ERROR: Division by zero in window function
df_with_metrics = df.withColumn(
    "running_avg_quantity",
    avg("quantity").over(window_spec)
).withColumn(
    "running_avg_amount",
    avg("amount").over(window_spec)
).withColumn(
    # This will cause division by zero when amount is 0
    "ratio_to_avg",
    col("amount") / col("running_avg_amount")  # Division by zero
).withColumn(
    # Another division by zero
    "quantity_ratio",
    col("quantity") / col("running_avg_quantity")  # Division by zero
)

# Additional calculation that will also fail
df_with_metrics = df_with_metrics.withColumn(
    "avg_price_per_unit",
    when(col("quantity") > 0, col("amount") / col("quantity"))
    .otherwise(0)  # This returns 0 for division by zero, but the window function above already fails
)

# Add more window functions that cause division issues
df_with_metrics = df_with_metrics.withColumn(
    "growth_rate",
    (col("amount") - lag("amount", 1).over(window_spec)) / lag("amount", 1).over(window_spec)
)

# Force execution - will throw ArithmeticException
df_with_metrics.show()

# Group by with division that will also fail
result = df_with_metrics.groupBy("store").agg(
    sum("amount").alias("total_amount"),
    sum("quantity").alias("total_quantity")
).withColumn(
    "avg_price",
    col("total_amount") / col("total_quantity")  # Division by zero for stores with zero quantity
)

result.show()
