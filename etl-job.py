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

# Canonical column ordering — single source of truth for all layers
GOLD_SCHEMA_COLUMNS = ["id", "email", "age"]

logger.info("Initializing Customer Demographics ETL Job...")

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
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # Re-order df2_silver columns to match df1_silver's canonical schema: (id, email, age)
    # Spark's union() is position-based, so both DataFrames must share identical column
    # positions and types before the union is performed.
    df2_silver_aligned = df2_silver.select(*GOLD_SCHEMA_COLUMNS)

    # Pre-union schema equality guard: raises AssertionError at the driver level
    # before any distributed tasks are launched, preventing wasted executor retries
    # on guaranteed-to-fail structural data errors.
    assert df1_silver.schema == df2_silver_aligned.schema, (
        f"Schema mismatch before union: df1={df1_silver.schema}, df2={df2_silver_aligned.schema}"
    )

    # Now union is safe: both DataFrames share identical column positions and types
    df_gold = df1_silver.union(df2_silver_aligned)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
