import os
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.utils import AnalysisException

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

# 2. Define valid incoming source data conforming to schema_v1
data_v2_good = [
    (3, "Charlie", 300.75),
    (4, "Dave", 420.00)
]

# Create valid DataFrame using the established schema
df_v2_good = spark.createDataFrame(data_v2_good, schema=schema_v1)
print("Step 2: Valid source data prepared for ingestion into pipeline...")

# 3. Append the valid data to the existing Delta table
# Wrapped in try/except to handle any unexpected schema or write errors gracefully
# without terminating the entire Spark session
try:
    df_v2_good.write.format("delta").mode("append").save(target_table_path)
    print("Step 3: Valid data successfully appended to Delta table.")
except AnalysisException as e:
    print(f"ERROR: Schema mismatch or analysis error during Delta write: {e}")
    raise
except Exception as e:
    print(f"ERROR: Unexpected error during Delta write: {e}")
    raise

# 4. (Optional) Intentional failure test — only enabled via explicit environment flag
# Gate the schema mismatch test behind an environment variable so it cannot
# accidentally execute in production or staging pipelines.
if os.getenv("ENABLE_FAILURE_TEST") == "true":
    print("WARN: ENABLE_FAILURE_TEST is set. Running intentional schema mismatch test...")

    # Simulate incoming bad source data
    # (Missing the 'purchase_amount' column, adding an unexpected 'region' column)
    schema_v2_bad = StructType([
        StructField("customer_id", IntegerType(), False),
        StructField("name", StringType(), False),
        StructField("region", StringType(), False)  # Schema mismatch!
    ])

    data_v2_bad = [
        (5, "Eve", "East"),
        (6, "Frank", "West")
    ]

    # Create bad DataFrame
    df_v2_bad = spark.createDataFrame(data_v2_bad, schema=schema_v2_bad)
    print("Intentional test: Bad source data created. Attempting mismatched append...")

    # Attempt to append the bad data — expected to raise DeltaAnalysisException
    # Caught here to allow downstream validation logic to run without crashing the session
    try:
        df_v2_bad.write.format("delta").mode("append").save(target_table_path)
        print("WARNING: Schema mismatch write succeeded unexpectedly — test did not trigger expected failure.")
    except AnalysisException as e:
        print(f"Schema mismatch caught as expected (test passed): {e}")
        # Emit metric, alert, or log structured event here for monitoring validation
    except Exception as e:
        print(f"Unexpected exception during failure test: {e}")
else:
    print("Step 4: Intentional failure test skipped (ENABLE_FAILURE_TEST not set).")

print("Pipeline completed successfully.")
