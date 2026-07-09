from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# 1. Define the original valid schema and data
schema_v1 = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("purchase_amount", DoubleType(), False)
])

data_v1 = [
    (1, "Alice", 250.00),
    (2, "Bob", 150.50)
]

# Create initial DataFrame
df_v1 = spark.createDataFrame(data_v1, schema=schema_v1)

# Write the valid data to a temporary Delta table to establish the target schema
target_table_path = "dbfs:/tmp/agentic_dataops_schema_test"
df_v1.write.format("delta").mode("overwrite").save(target_table_path)
print("Step 1: Initial valid data successfully written to Delta table.")

# 2. Simulate incoming bad source data 
# (Missing the 'purchase_amount' column, adding an unexpected 'region' column)
schema_v2_bad = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("name", StringType(), False),
    StructField("region", StringType(), False)  # Schema mismatch!
])

data_v2_bad = [
    (3, "Charlie", "East"),
    (4, "Dave", "West")
]

# Create bad DataFrame
df_v2_bad = spark.createDataFrame(data_v2_bad, schema=schema_v2_bad)
print("Step 2: Bad source data detected. Attempting to ingest into pipeline...")

# 3. Attempt to append the bad data to the existing Delta table
# This will now succeed due to schema evolution (mergeSchema="true")
df_v2_bad.write.format("delta").mode("append").option("mergeSchema", "true").save(target_table_path)

print("Step 3: Bad source data successfully appended with schema merge.")
