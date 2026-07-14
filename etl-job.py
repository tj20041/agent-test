from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

# 1. Define the original valid schema and data
schema_v1 = StructType([
    StructField("customer_id",     IntegerType(), False),
    StructField("name",            StringType(),  False),
    StructField("email",           StringType(),  False),  # PII
    StructField("phone",           StringType(),  False),  # PII
    StructField("purchase_amount", DoubleType(),  False)
])

data_v1 = [
    (1, "Alice", "alice@company.com", "555-019-8372", 250.00),
    (2, "Bob",   "bob@company.com",   "555-887-2341", 150.50)
]

# Create initial DataFrame
df_v1 = spark.createDataFrame(data_v1, schema=schema_v1)

# Write the valid data to a temporary Delta table to establish the target schema
target_table_path = "dbfs:/tmp/agentic_dataops_schema_test"
df_v1.write.format("delta").mode("overwrite").save(target_table_path)
print("Step 1: Initial valid data successfully written to Delta table.")


# 2. Simulate incoming bad source data
# (Missing 'email', 'phone', 'purchase_amount' — adding unexpected 'region' column)
schema_v2_bad = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("name",        StringType(),  False),
    StructField("region",      StringType(),  False)  # Schema mismatch!
])

data_v2_bad = [
    (3, "Charlie", "East"),
    (4, "Dave",    "West")
]

# Create bad DataFrame
df_v2_bad = spark.createDataFrame(data_v2_bad, schema=schema_v2_bad)
print("Step 2: Bad source data detected. Attempting to ingest into pipeline...")


# 3. Attempt to append the bad data to the existing Delta table
# This will intentionally fail and throw DELTA_SCHEMA_MISMATCH / AnalysisException
# The error diff will show the target schema containing 'email' and 'phone' columns
df_v2_bad.write.format("delta").mode("append").save(target_table_path)

print("SUCCESS: If you see this, the test failed to break. The pipeline should crash before this line.")
