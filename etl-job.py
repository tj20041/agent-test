from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp

spark = SparkSession.builder.appName("SalesETLJob").getOrCreate()

# Sample input data
data = [
    (1, "Laptop", 70000),
    (2, "Mobile", 30000),
    (3, "Keyboard", 2000),
    (4, "Monitor", 15000),
]

columns = ["id", "product", "price"]

df = spark.createDataFrame(data, columns)

print("Raw Data")
df.show()

# INTENTIONAL ERROR:
# 'product_price' does not exist.
# The actual column name is 'price'.
processed_df = (
    df
    .filter(col("product_price") > 5000)
    .withColumn("processed_at", current_timestamp())
)

print("Processed Data")
processed_df.show()

processed_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.processed_sales")

print("Databricks job completed successfully")
