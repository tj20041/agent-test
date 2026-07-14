import logging
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import udf, col

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("PII_ETL_Validator")

# ── Test case selector ────────────────────────────────────────────────────────
TEST_CASE = "pii_violation"   # change to "success" to test the happy path

# ── Shared schema and data ────────────────────────────────────────────────────
schema = StructType([
    StructField("customer_id", IntegerType(), False),
    StructField("email",       StringType(),  True),
    StructField("phone",       StringType(),  True),
    StructField("amount",      DoubleType(),  False),
])

data_good = [
    (1, "alice@company.com",   "555-010-0000",  250.00),
    (2, "bob@company.com",     "555-010-0001",  150.50),
]

data_with_pii_violation = data_good + [
    (3, "jane.doe@gmail.com",  "555-019-8372", -100.00),   # negative amount + PII
    (4, "john.smith@gmail.com","555-911-0001", -999.99),   # second bad PII row
]

# ── Happy-path test ───────────────────────────────────────────────────────────
if TEST_CASE == "success":

    logger.info("TEST_CASE=success — running clean ingestion")

    df = spark.createDataFrame(data_good, schema=schema)
    logger.info("DataFrame created with %d rows", df.count())

    df.write.format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save("dbfs:/tmp/pii_etl_demo/clean")

    logger.info("Write completed successfully — no PII violations detected")

# ── PII-in-error test ─────────────────────────────────────────────────────────
elif TEST_CASE == "pii_violation":

    logger.info("TEST_CASE=pii_violation — loading dataset that contains bad rows")

    df = spark.createDataFrame(data_with_pii_violation, schema=schema)
    logger.info("DataFrame created with %d rows (includes bad PII rows)", df.count())

    # Validation UDF — raises ValueError with PII baked into the message.
    # Python's logging module routes this to stderr which Databricks
    # captures reliably in the "Recent log files" UI tab.
    @udf(returnType=DoubleType())
    def validate_amount(amount, email, phone):
        if amount < 0:
            raise ValueError(
                f"DATA QUALITY VIOLATION: negative amount={amount} "
                f"for customer email={email} phone={phone}"
            )
        return amount

    logger.info("Applying validation UDF — crash expected on bad rows")

    df_validated = df.withColumn(
        "amount",
        validate_amount(col("amount"), col("email"), col("phone"))
    )

    try:
        df_validated.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save("dbfs:/tmp/pii_etl_demo/validated")

        logger.info("Write succeeded — this line should NOT appear")

    except Exception as exc:
        # Log the full exception (including the PII ValueError) via the
        # Python logging module so it lands in stderr / the UI log tab.
        logger.error("Pipeline failed — data quality violation: %s", exc)
        raise   # re-raise so Databricks marks the job as FAILED
