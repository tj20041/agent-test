from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException
from delta.tables import DeltaTable

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
quarantine_table_path = "dbfs:/tmp/agentic_dataops_schema_test_quarantine"

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

# 3. Pre-write schema compatibility check
# Compare incoming DataFrame schema against the existing Delta table schema
# before attempting the append operation.
target_schema = DeltaTable.forPath(spark, target_table_path).toDF().schema
incoming_schema = df_v2_bad.schema

target_field_names = {f.name for f in target_schema}
incoming_field_names = {f.name for f in incoming_schema}

missing_columns = target_field_names - incoming_field_names
extra_columns = incoming_field_names - target_field_names
schema_compatible = (missing_columns == set() and extra_columns == set())

if not schema_compatible:
    print(
        f"Step 3: Schema mismatch detected before write.\n"
        f"  Target schema fields   : {sorted(target_field_names)}\n"
        f"  Incoming schema fields : {sorted(incoming_field_names)}\n"
        f"  Missing columns        : {sorted(missing_columns)}\n"
        f"  Extra/unexpected cols  : {sorted(extra_columns)}"
    )

    # Route rejected records to a quarantine Delta table with a rejection reason column
    rejection_reason = (
        f"Schema mismatch — missing: {sorted(missing_columns)}, "
        f"unexpected: {sorted(extra_columns)}"
    )
    df_quarantine = df_v2_bad.withColumn(
        "_rejection_reason", F.lit(rejection_reason)
    )
    df_quarantine.write.format("delta").mode("append").save(quarantine_table_path)
    print(
        f"Step 3a: Rejected records written to quarantine path: {quarantine_table_path}"
    )
else:
    # 4. Attempt to append the data to the existing Delta table only when schemas are compatible.
    # Wrapped in try/except to catch any residual Delta schema enforcement errors.
    try:
        df_v2_bad.write.format("delta").mode("append").save(target_table_path)
        print("Step 4: Data successfully appended to Delta table.")
    except AnalysisException as exc:
        error_msg = str(exc)
        if "DELTA_SCHEMA_MISMATCH" in error_msg or "schema mismatch" in error_msg.lower():
            print(
                f"Step 4 ERROR: Delta schema mismatch caught at write time.\n"
                f"  Target path    : {target_table_path}\n"
                f"  Expected schema: {target_schema}\n"
                f"  Received schema: {incoming_schema}\n"
                f"  Exception      : {error_msg}"
            )
            # Route to quarantine on unexpected write-time schema error
            rejection_reason = f"Write-time DELTA_SCHEMA_MISMATCH: {error_msg[:200]}"
            df_quarantine = df_v2_bad.withColumn(
                "_rejection_reason", F.lit(rejection_reason)
            )
            df_quarantine.write.format("delta").mode("append").save(quarantine_table_path)
            print(
                f"Step 4a: Rejected records written to quarantine path: {quarantine_table_path}"
            )
        else:
            # Re-raise unexpected AnalysisExceptions that are not schema-related
            raise

print("Pipeline run complete. Check quarantine table for any rejected records.")
