import logging
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, BooleanType
from pyspark.sql.functions import udf, col, lit, current_timestamp

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
    logger.info("DataFrame created with %d rows (includes bad rows)", df.count())

    # Validation UDF — returns True/False without raising exceptions and
    # without embedding PII in any message. This avoids Spark task retries
    # caused by exception-based validation and prevents PII leakage into logs.
    @udf(returnType=BooleanType())
    def is_valid_amount(amount):
        """Return True if amount satisfies the non-negative business rule."""
        if amount is None:
            return False
        return amount >= 0

    logger.info("Applying filter-based validation — invalid rows will be quarantined")

    # Tag every row with a validity flag; no PII is referenced here.
    df_with_validity = df.withColumn("is_valid", is_valid_amount(col("amount")))

    df_valid   = df_with_validity.filter(col("is_valid") == True).drop("is_valid")
    df_invalid = df_with_validity.filter(col("is_valid") == False).drop("is_valid")

    # Log counts using only non-sensitive aggregate metrics — no PII exposed.
    valid_count   = df_valid.count()
    invalid_count = df_invalid.count()
    logger.info("Validation complete — valid rows: %d, invalid rows: %d",
                valid_count, invalid_count)

    if invalid_count > 0:
        # Log only non-sensitive identifiers (customer_id) for quarantine audit.
        bad_ids = [row.customer_id for row in df_invalid.select("customer_id").collect()]
        logger.warning(
            "DATA QUALITY VIOLATION: %d row(s) with negative amount — "
            "customer_ids=%s — rows quarantined, no PII in this message",
            invalid_count, bad_ids
        )

    try:
        # Write valid rows to the primary Delta table.
        df_valid.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save("dbfs:/tmp/pii_etl_demo/validated")
        logger.info("Validated write succeeded — %d rows written", valid_count)

        # Write invalid rows to a quarantine table for data-steward review.
        # Only non-sensitive fields (customer_id, amount, rejection metadata)
        # are preserved here; email and phone remain stored but access-controlled.
        if invalid_count > 0:
            df_quarantine = df_invalid.select(
                col("customer_id"),
                col("amount"),
                lit("negative_amount").alias("rejection_reason_code"),
                current_timestamp().alias("quarantined_at")
            )
            df_quarantine.write.format("delta").mode("append") \
                .option("mergeSchema", "true") \
                .save("dbfs:/tmp/pii_etl_demo/quarantine")
            logger.info("Quarantine write succeeded — %d invalid row(s) stored "
                        "with non-PII metadata only", invalid_count)

    except Exception as exc:
        # Log only a generic failure message — do NOT include exc directly
        # as it may contain PII from upstream stack frames.
        logger.error(
            "Pipeline failed during Delta write — check cluster logs for "
            "storage/network details. Row counts: valid=%d, invalid=%d",
            valid_count, invalid_count
        )
        raise   # re-raise so Databricks marks the job as FAILED
