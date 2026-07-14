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

    # Canonical column blueprint that the Gold layer expects.
    CANONICAL_COLUMNS = ["id", "email", "age"]

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

    # Defensive schema-validation gate: fail fast if the two DataFrames do
    # not expose the same set of columns before we attempt to merge them.
    df1_cols = set(df1_silver.columns)
    df2_cols = set(df2_silver.columns)
    if df1_cols != df2_cols:
        raise ValueError(
            "Schema mismatch between Silver DataFrames. "
            "df1 columns: %s, df2 columns: %s" % (sorted(df1_cols), sorted(df2_cols))
        )

    # Re-order both DataFrames to the canonical blueprint so downstream Gold
    # consumers always see a stable (id, email, age) column order.
    df1_aligned = df1_silver.select(*CANONICAL_COLUMNS)
    df2_aligned = df2_silver.select(*CANONICAL_COLUMNS)

    # Use unionByName so columns are matched by NAME rather than by position.
    # schema1 order is (id, email, age) while schema2 order is (email, age, id);
    # a positional union() would place email strings into the IntegerType 'id'
    # column and trigger a CAST_INVALID_INPUT (SQLSTATE 22018) failure.
    df_gold = df1_aligned.unionByName(df2_aligned)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
