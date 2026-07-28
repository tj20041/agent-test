# Databricks Notebook - Customer Sales Analysis
# This notebook analyzes customer sales data from a CSV file

from pyspark.sql.functions import col, sum, avg, count, max, min, round
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType

# Define the schema for the sales data
sales_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("customer_name", StringType(), True),
    StructField("purchase_amount", DoubleType(), True),
    StructField("purchase_date", StringType(), True),
    StructField("product_category", StringType(), True),
    StructField("store_location", StringType(), True)
])

# Define the path to the CSV file
file_path = "/Volumes/default/customer_data/sales_records.csv"

# Read the CSV file
sales_df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .schema(sales_schema) \
    .csv(file_path)

# Cache the data for faster processing
sales_df.cache()

# Show the first few records
print("Sample of sales data:")
sales_df.show(10, truncate=False)

# Display data schema
print("Data Schema:")
sales_df.printSchema()

# Basic statistics
print("Basic Statistics:")
sales_df.describe().show()

# Total sales by customer
print("Total Sales by Customer:")
customer_sales = sales_df.groupBy("customer_id", "customer_name").agg(
    sum("purchase_amount").alias("total_purchases"),
    count("purchase_amount").alias("purchase_count"),
    avg("purchase_amount").alias("avg_purchase"),
    max("purchase_amount").alias("max_purchase"),
    min("purchase_amount").alias("min_purchase")
).orderBy(col("total_purchases").desc())

customer_sales.show(20, truncate=False)

# Total sales by product category
print("Sales by Product Category:")
category_sales = sales_df.groupBy("product_category").agg(
    sum("purchase_amount").alias("category_total"),
    count("purchase_amount").alias("transaction_count"),
    avg("purchase_amount").alias("avg_transaction")
).orderBy(col("category_total").desc())

category_sales.show(truncate=False)

# Monthly sales trend
from pyspark.sql.functions import substring

sales_df = sales_df.withColumn(
    "month",
    substring(col("purchase_date"), 1, 7)  # Extract YYYY-MM
)

print("Monthly Sales Trend:")
monthly_sales = sales_df.groupBy("month").agg(
    sum("purchase_amount").alias("monthly_total"),
    count("purchase_amount").alias("transaction_count")
).orderBy("month")

monthly_sales.show(truncate=False)

# Store performance analysis
print("Store Performance:")
store_performance = sales_df.groupBy("store_location").agg(
    sum("purchase_amount").alias("store_revenue"),
    count("purchase_amount").alias("store_transactions"),
    avg("purchase_amount").alias("avg_transaction_value")
).orderBy(col("store_revenue").desc())

store_performance.show(truncate=False)

# Customer segmentation based on purchase behavior
from pyspark.sql.functions import when

customer_segments = customer_sales.withColumn(
    "segment",
    when(col("total_purchases") > 10000, "Premium")
    .when(col("total_purchases") > 5000, "Gold")
    .when(col("total_purchases") > 1000, "Silver")
    .otherwise("Bronze")
)

print("Customer Segmentation:")
customer_segments.groupBy("segment").agg(
    count("customer_id").alias("customer_count"),
    sum("total_purchases").alias("segment_revenue"),
    avg("total_purchases").alias("avg_customer_value")
).show(truncate=False)

# Calculate retention metrics
from pyspark.sql.functions import datediff, current_date

# This will fail if the file is not found
total_revenue = sales_df.agg(sum("purchase_amount")).collect()[0][0]
total_customers = sales_df.select("customer_id").distinct().count()
total_transactions = sales_df.count()

print(f"Total Revenue: ${total_revenue:,.2f}")
print(f"Total Customers: {total_customers}")
print(f"Total Transactions: {total_transactions}")
print(f"Average Transaction Value: ${total_revenue/total_transactions:,.2f}")

# Save aggregated results
final_results = customer_sales.join(
    category_sales.withColumnRenamed("category_total", "sales_by_category"),
    customer_sales.customer_id == category_sales.product_category,
    "full"
)

final_results.show(truncate=False)

# Additional analysis - top spending customers by category
top_customers = sales_df.groupBy("customer_id", "customer_name", "product_category").agg(
    sum("purchase_amount").alias("spend_by_category")
).orderBy(col("spend_by_category").desc())

print("Top Customers by Category Spend:")
top_customers.show(20, truncate=False)

print("Analysis Complete!")
