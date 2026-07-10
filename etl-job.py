from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.utils import AnalysisException

# ---------------------------------------------------------------------------
# Helper: validate that the incoming DataFrame schema is compatible with the
# target Delta table schema before attempting a write.
# ---------------------------------------------------------------------------
def validate_schema(incoming_df, target_schema):
    """
    Compare the incoming DataFrame schema against the expected target schema.

    Returns a tuple (is_valid: bool, issues: list[str]) so the caller can
    decide how to handle a mismatch (abort, log, transform, etc.).
    """
    target_field_names = {f.name for f in target_schema.fields}
    incoming_field_names = {f.name for f in incoming_df.schema.fields}

    missing_fields = target_field_names - incoming_field_names
    extra_fields = incoming_field_names - target_field_names

    issues = []
    if missing_fields:
        issues.append(f"Missing required columns in incoming data: {sorted(missing_fields)}")
    if extra_fields:
        issues.append(f"Unexpected extra columns in incoming data: {sorted(extra_fields)}")

    # Also check data-type compatibility for columns that exist in both
    target_field_map = {f.name: f.dataType for f in target_schema.fields}
    for field in incoming_df.schema.fields:
        if field.name in target_field_map:
            if field.dataType != target_field_map[field.name]:
                issues.append(
                    f"Type mismatch on column '{field.name}': "
                    f"expected {target_field_map[field.name]}, got {field.dataType}"
                )

    return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# 1. Define the original valid schema and data
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. Simulate incoming bad source data
#    (Missing 'purchase_amount' column, adding unexpected 'region' column)
# ---------------------------------------------------------------------------
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
print("Step 2: Bad source data detected. Validating schema before ingestion...")

# ---------------------------------------------------------------------------
# 3. Validate incoming schema against the established target schema BEFORE
#    attempting the write, and raise a descriptive error on mismatch so the
#    pipeline fails fast with a clear, actionable message.
# ---------------------------------------------------------------------------
is_valid, schema_issues = validate_schema(df_v2_bad, schema_v1)

if not is_valid:
    error_msg = (
        "Schema validation failed — incoming data does not match the target "
        "Delta table schema at '{}'.\n".format(target_table_path)
        + "\n".join(f"  - {issue}" for issue in schema_issues)
        + "\nExpected schema : {}".format(
            [(f.name, str(f.dataType)) for f in schema_v1.fields]
        )
        + "\nIncoming schema : {}".format(
            [(f.name, str(f.dataType)) for f in df_v2_bad.schema.fields]
        )
    )
    print(f"ERROR: {error_msg}")
    raise ValueError(error_msg)

# ---------------------------------------------------------------------------
# 4. Only reach this block when the schema is valid.
#    The option mergeSchema='true' is kept as an additional safety net to
#    allow controlled schema evolution for additive (non-breaking) changes.
# ---------------------------------------------------------------------------
try:
    df_v2_bad.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .save(target_table_path)
    print("Step 3: Data successfully appended to Delta table.")
except AnalysisException as exc:
    # Catch DeltaAnalysisException specifically and surface schema details
    print(
        "ERROR: DeltaAnalysisException while writing to Delta table '{}'. "
        "Expected schema: {}. Incoming schema: {}. "
        "Original error: {}".format(
            target_table_path,
            [(f.name, str(f.dataType)) for f in schema_v1.fields],
            [(f.name, str(f.dataType)) for f in df_v2_bad.schema.fields],
            str(exc)
        )
    )
    raise
