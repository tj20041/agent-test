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

    # Fixed: schema2 column order now matches schema1 (id, email, age)
    # to ensure positional alignment is consistent from the Bronze layer
    # and prevent silent type mismatches if union() is ever used.
    schema2 = StructType([
        StructField("id", IntegerType(), True),
        StructField("email", StringType(), True),
        StructField("age", IntegerType(), True)
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

    # Fixed: row tuples reordered to match canonical (id, email, age) schema2
    df2_bronze = spark.createDataFrame([
        (4, "dave.miller@startup.io", 22),
        (5, "bot_account_x@spam.com", -5),
        (6, "eve.adams@enterprise.co", 28)
    ], schema=schema2)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")

    # Data quality check: warn and quarantine negative-age records at Bronze
    # before filtering so upstream degradation is visible in the logs.
    df1_bad = df1_bronze.filter("age < 0")
    df2_bad = df2_bronze.filter("age < 0")
    bad_count_1 = df1_bad.count()
    bad_count_2 = df2_bad.count()
    if bad_count_1 > 0:
        logger.warning(
            "Bronze layer df1 contains %d record(s) with negative age — quarantining.",
            bad_count_1
        )
    if bad_count_2 > 0:
        logger.warning(
            "Bronze layer df2 contains %d record(s) with negative age — quarantining.",
            bad_count_2
        )

    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # Schema compatibility guard: surface mismatches at driver side (plan time)
    # before launching any distributed tasks, giving a clear error message
    # instead of a cryptic executor-level CAST_INVALID_INPUT failure.
    assert df1_silver.schema == df2_silver.schema, (
        f"Schema mismatch before union: {df1_silver.schema} vs {df2_silver.schema}"
    )

    # Fixed: replaced positional union() with unionByName() so Spark resolves
    # columns by name rather than position, eliminating the STRING->BIGINT
    # CAST_INVALID_INPUT / SparkNumberFormatException (SQLSTATE 22018) that
    # was caused by aligning df2's 'email' STRING column against df1's 'id'
    # BIGINT column during positional merging.
    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
