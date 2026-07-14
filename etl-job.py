import logging
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# ==========================================
# 0. LOCAL-SAFE LOGGING SETUP
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
# 0.1 LOCAL SPARK SESSION INITIALIZATION
# ==========================================
# On the LOCAL target platform there is no ambient 'spark' object,
# so we must build one explicitly with a local master.
spark = (
    SparkSession.builder
    .appName("CustomerDemographicsETL")
    .master("local[*]")
    .getOrCreate()
)


def assert_union_compatible(left_df, right_df):
    """Validate that two DataFrames share identical column names and types.

    This runs before any union so schema drift surfaces as a clear,
    actionable error instead of an obscure cast failure downstream.
    """
    left_fields = {f.name: f.dataType for f in left_df.schema.fields}
    right_fields = {f.name: f.dataType for f in right_df.schema.fields}

    if set(left_fields.keys()) != set(right_fields.keys()):
        raise ValueError(
            "Union column-name mismatch. "
            f"left={sorted(left_fields.keys())} right={sorted(right_fields.keys())}"
        )

    for name, left_type in left_fields.items():
        right_type = right_fields[name]
        if left_type != right_type:
            raise ValueError(
                f"Union column type mismatch for '{name}': "
                f"left={left_type} right={right_type}"
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
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # Validate schema compatibility before merging so any future
    # column drift fails fast with a clear message.
    assert_union_compatible(df1_silver, df2_silver)

    # FIX: Use unionByName so columns align by NAME, not position.
    # df1_silver has (id, email, age) while df2_silver has (email, age, id).
    # A positional union() would stack the email String onto the id BIGINT
    # column, triggering CAST_INVALID_INPUT (SQLSTATE 22018).
    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    # FIX: display() is Databricks-only; use portable show() for LOCAL execution.
    df_gold.show(truncate=False)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
finally:
    try:
        spark.stop()
    except Exception:
        pass
