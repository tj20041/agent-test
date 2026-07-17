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
# PRE-FLIGHT: Reference Data Validation
# ==========================================
REFERENCE_PATH = "dbfs:/mnt/reference_data/regional_tax_rates_2026.csv"

try:
    dbutils.fs.ls(REFERENCE_PATH)
    logger.info("Pre-flight check passed: reference file found at %s", REFERENCE_PATH)
except Exception:
    logger.error(
        "Pre-flight check FAILED: reference file missing at %s. "
        "Ensure the file has been uploaded to DBFS before running this job.",
        REFERENCE_PATH
    )
    raise FileNotFoundError(
        f"Required reference file not found: {REFERENCE_PATH}. "
        "Upload the file or update the path in the pipeline configuration."
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

    # Pre-union schema compatibility assertion
    assert set(df1_silver.columns) == set(df2_silver.columns), (
        f"Schema mismatch before Gold union: "
        f"df1_silver columns={df1_silver.columns}, "
        f"df2_silver columns={df2_silver.columns}"
    )

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # FIX: Use unionByName() to align columns by name rather than ordinal
    # position, preventing the 'Positional Pitfall' silent data-corruption
    # bug that would occur when schema column orders differ across frames.
    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
