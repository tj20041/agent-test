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

    # Schema assertions to catch column-order drift at the earliest stage
    assert df1_bronze.columns == ['id', 'email', 'age'], \
        f'df1_bronze schema mismatch: {df1_bronze.columns}'
    assert df2_bronze.columns == ['email', 'age', 'id'], \
        f'df2_bronze schema mismatch: {df2_bronze.columns}'
    logger.info("Bronze schema assertions passed.")

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

    # FIX: Use unionByName() to align columns by name rather than position,
    # preventing SparkNumberFormatException (CAST_INVALID_INPUT / SQLSTATE: 22018)
    # that occurred when the bare union() positionally mapped schema2's
    # StringType 'email' column against schema1's IntegerType 'id' column.
    df_gold = df1_silver.unionByName(df2_silver)

    # Data quality assertions on the Gold output
    gold_count = df_gold.count()
    assert gold_count > 0, "Gold layer is empty after union"
    null_ids = df_gold.filter(df_gold.id.isNull()).count()
    assert null_ids == 0, \
        f"{null_ids} NULL id values detected in Gold layer — possible positional misalignment"
    logger.info("Gold layer data quality checks passed. Row count: %d", gold_count)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
