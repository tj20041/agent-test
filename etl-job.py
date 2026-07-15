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

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # ------------------------------------------
    # Pre-merge schema-consistency guard.
    # df1_silver schema: (id, email, age)
    # df2_silver schema: (email, age, id) -- same columns, different order.
    # DataFrame.union() aligns columns POSITIONALLY, which would force
    # STRING email values into the BIGINT 'id' column and raise
    # SparkNumberFormatException [CAST_INVALID_INPUT] SQLSTATE 22018.
    # We validate that both DataFrames expose the same set of
    # (name, type) fields before merging, then merge BY NAME.
    # ------------------------------------------
    df1_fields = {(f.name, f.dataType.simpleString()) for f in df1_silver.schema.fields}
    df2_fields = {(f.name, f.dataType.simpleString()) for f in df2_silver.schema.fields}

    if df1_fields != df2_fields:
        only_in_df1 = df1_fields - df2_fields
        only_in_df2 = df2_fields - df1_fields
        raise ValueError(
            "Schema mismatch between df1_silver and df2_silver before Gold merge. "
            f"Fields only in df1_silver: {sorted(only_in_df1)}. "
            f"Fields only in df2_silver: {sorted(only_in_df2)}."
        )

    logger.info("Schema-consistency check passed; merging by column name...")

    # Use unionByName so columns are matched by NAME, not position.
    # This is the self-documenting, Databricks/Spark-preferred fix and
    # resolves the [CAST_INVALID_INPUT] SparkNumberFormatException.
    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
