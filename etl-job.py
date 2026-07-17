import logging
import sys
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# ==========================================
# 0. DATABRICKS-SAFE LOGGING SETUP
# ==========================================
logger = logging.getLogger("CustomerDemographicsETL")
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.propagate = False

logger.info("Initializing Customer Demographics ETL Job...")

# ==========================================
# PRE-FLIGHT: Validate required DBFS reference files exist
# before any Spark analysis begins.
# ==========================================
REFERENCE_FILE_PATH = "dbfs:/mnt/reference_data/regional_tax_rates_2026.csv"

try:
    dbutils.fs.ls(REFERENCE_FILE_PATH)
    logger.info("Pre-flight check passed: reference file found at %s", REFERENCE_FILE_PATH)
except Exception:
    raise FileNotFoundError(
        f"Pre-flight check FAILED: Reference file not found at expected DBFS path '{REFERENCE_FILE_PATH}'. "
        "Ensure the upstream file-delivery job has completed successfully before running this ETL. "
        "You can upload the file via: databricks fs cp local_file.csv dbfs:/mnt/reference_data/regional_tax_rates_2026.csv"
    )

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")
    schema1 = StructType([
        StructField("id", IntegerType(), True),
        StructField("email", StringType(), True),
        StructField("age", IntegerType(), True)
    ])

    schema2 = StructType([
        StructField("email", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("id", IntegerType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting data into Bronze layer...")
    df1_bronze = spark.createDataFrame([
        (1, "alice.smith@example.com", 25),
        (2, "test.user_99@domain.net", -15),
        (3, "charlie.brown@company.org", 30)
    ], schema=schema1)

    df2_bronze = spark.createDataFrame([
        ("dave.miller@startup.io", 22, 4),
        ("bot_account_x@spam.com", -5, 5),
        ("eve.adams@enterprise.co", 28, 6)
    ], schema=schema2)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # ==========================================
    # 4. PRE-UNION SCHEMA COMPATIBILITY VALIDATION
    # ==========================================
    logger.info("Validating schema compatibility before Gold layer union...")
    df1_fields = {field.name: field.dataType for field in df1_silver.schema.fields}
    df2_fields = {field.name: field.dataType for field in df2_silver.schema.fields}

    schema_errors = []
    for col_name, col_type in df1_fields.items():
        if col_name not in df2_fields:
            schema_errors.append(f"Column '{col_name}' present in df1_silver but missing from df2_silver.")
        elif df2_fields[col_name] != col_type:
            schema_errors.append(
                f"Column '{col_name}' type mismatch: df1_silver has {col_type}, df2_silver has {df2_fields[col_name]}."
            )
    for col_name in df2_fields:
        if col_name not in df1_fields:
            schema_errors.append(f"Column '{col_name}' present in df2_silver but missing from df1_silver.")

    if schema_errors:
        raise ValueError(
            "Pre-union schema validation FAILED. Incompatible schemas detected between Silver DataFrames:\n"
            + "\n".join(schema_errors)
        )
    logger.info("Schema compatibility validation passed.")

    # ==========================================
    # 5. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # FIX: Use unionByName() instead of union() to resolve columns by name
    # rather than position. This prevents silent data corruption caused by
    # positional misalignment when the two Silver DataFrames have different
    # column orderings (schema1: id, email, age vs schema2: email, age, id).
    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
