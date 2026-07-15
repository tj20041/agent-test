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

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # Log schemas after transformation for observability and future schema-drift detection
    logger.info("df1_silver schema: %s", df1_silver.schema.simpleString())
    logger.info("df2_silver schema: %s", df2_silver.schema.simpleString())

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # Define the canonical column order matching schema1
    canonical_columns = ["id", "email", "age"]

    # Realign both DataFrames to the canonical column order before union.
    # df2_silver was ingested with column order (email, age, id), which is
    # incompatible with df1_silver's order (id, email, age). Using an explicit
    # select() ensures Spark aligns columns by name rather than by position,
    # preventing the CAST_INVALID_INPUT / SparkNumberFormatException that
    # occurred when union() mapped df2_silver's StringType 'email' column
    # into df1_silver's IntegerType 'id' column.
    df1_aligned = df1_silver.select(canonical_columns)
    df2_aligned = df2_silver.select(canonical_columns)

    # Guard: assert schema compatibility before union to surface any future
    # schema drift as an immediate, descriptive failure rather than a silent
    # runtime data corruption after retries are exhausted.
    assert df1_aligned.schema == df2_aligned.schema, (
        f"Schema mismatch before union: {df1_aligned.schema} vs {df2_aligned.schema}"
    )

    # unionByName() is used as the Databricks-idiomatic, name-based union.
    # On Databricks Runtime (Spark 3.1+), unionByName() aligns columns by
    # name and raises an AnalysisException at query-planning time if column
    # names do not match, rather than silently misaligning at execution time.
    df_gold = df1_aligned.unionByName(df2_aligned)

    logger.info("Pipeline completed successfully. Gold record count: %d", df_gold.count())
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
