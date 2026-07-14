import logging
import os
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import col, when, lit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("PII_ETL_Validator")

# ── Test case selector ────────────────────────────────────────────────────────
# Drive via environment variable so production runs always default to "success".
# Set ETL_TEST_CASE=pii_violation in a Databricks job parameter / widget to
# exercise the quarantine path in a test environment.
TEST_CASE = os.environ.get("ETL_TEST_CASE", "success")

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
    logger.info("DataFrame created with %d rows (includes bad rows)", df.count())

    # ── DataFrame-native validation: flag bad rows without a UDF ─────────────
    # Using a filter+quarantine pattern instead of a UDF that raises exceptions
    # avoids Spark task retries for expected bad data and prevents PII from
    # ever appearing in exception messages or job logs.
    df_flagged = df.withColumn(
        "_validation_error",
        when(col("amount") < 0, lit("negative_amount")).otherwise(lit(None))
    )

    df_valid = df_flagged.filter(col("_validation_error").isNull()) \
        .drop("_validation_error")

    # Quarantine log contains ONLY customer_id and error reason — no PII fields.
    df_invalid = df_flagged.filter(col("_validation_error").isNotNull()) \
        .select("customer_id", "_validation_error")

    try:
        # Write bad rows to quarantine path first (no PII columns included).
        df_invalid.write.format("delta").mode("append") \
            .save("dbfs:/tmp/pii_etl_demo/quarantine")

        # Write only valid rows to the main output path.
        df_valid.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save("dbfs:/tmp/pii_etl_demo/validated")

        valid_count = df_valid.count()
        invalid_count = df_invalid.count()
        logger.info(
            "Write completed — valid rows: %d, quarantined rows: %d",
            valid_count,
            invalid_count,
        )
        if invalid_count > 0:
            logger.warning(
                "Data quality issues detected — %d row(s) quarantined; "
                "see dbfs:/tmp/pii_etl_demo/quarantine for details",
                invalid_count,
            )

    except Exception as exc:
        # Log only the exception type and a safe summary.
        # Never interpolate the raw exception object (exc) into the log message
        # because it may carry PII from earlier in the call chain.
        logger.error(
            "Pipeline failed — unhandled exception of type %s",
            type(exc).__name__,
            exc_info=False,
        )
        raise   # re-raise so Databricks marks the job as FAILED
