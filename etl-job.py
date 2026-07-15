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


def assert_schema_compatible(df_a, df_b, context="union"):
    """Assert that two DataFrames have compatible schemas (same column names and types).

    Raises ValueError with a descriptive message if any column name or type
    mismatch is detected. This check runs on the Databricks driver before any
    distributed tasks are launched, surfacing schema errors cheaply and clearly.
    """
    cols_a = {f.name: f.dataType for f in df_a.schema.fields}
    cols_b = {f.name: f.dataType for f in df_b.schema.fields}
    mismatches = [
        f"Column '{c}': {cols_a.get(c)} vs {cols_b.get(c)}"
        for c in set(cols_a) | set(cols_b)
        if cols_a.get(c) != cols_b.get(c)
    ]
    if mismatches:
        raise ValueError(f"Schema mismatch before {context}: {mismatches}")


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

    # FIX 1: Harmonised schema2 column order to match schema1 (id, email, age).
    # Previously schema2 was defined as (email, age, id), causing Spark's
    # positional union() to align email (StringType) against id (IntegerType/BIGINT),
    # producing CAST_INVALID_INPUT / SparkNumberFormatException on every row.
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

    # FIX 1 (continued): Updated df2_bronze literals to match the corrected
    # schema2 column order (id, email, age). Previously literals were ordered
    # (email, age, id) to match the old mismatched schema2 definition.
    df2_bronze = spark.createDataFrame([
        (4, "dave.miller@startup.io", 22),
        (5, "bot_account_x@spam.com", -5),
        (6, "eve.adams@enterprise.co", 28)
    ], schema=schema2)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")

    pre_count_1 = df1_bronze.count()
    df1_silver = df1_bronze.filter("age >= 0")
    post_count_1 = df1_silver.count()
    logger.info(
        "df1 Silver filter: %d rows in, %d rows out, %d dropped",
        pre_count_1, post_count_1, pre_count_1 - post_count_1
    )

    pre_count_2 = df2_bronze.count()
    df2_silver = df2_bronze.filter("age >= 0")
    post_count_2 = df2_silver.count()
    logger.info(
        "df2 Silver filter: %d rows in, %d rows out, %d dropped",
        pre_count_2, post_count_2, pre_count_2 - post_count_2
    )

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # FIX 2: Added explicit name-based column reorder guard before union.
    # Declaring gold_columns and using .select() forces Spark to resolve columns
    # by name at plan-analysis time rather than by position at execution time.
    # Any future schema drift will raise a clear AnalysisException during query
    # planning on the driver, rather than a silent runtime CAST failure across
    # thousands of distributed tasks.
    gold_columns = ["id", "email", "age"]

    # Assert schema compatibility before union so failures surface on the driver
    # with a descriptive ValueError before any Spark tasks are launched.
    assert_schema_compatible(df1_silver, df2_silver, context="Gold union")

    df1_aligned = df1_silver.select(gold_columns)
    df2_aligned = df2_silver.select(gold_columns)
    df_gold = df1_aligned.union(df2_aligned)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
