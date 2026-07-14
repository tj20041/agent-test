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

    # Validation UDF — checks amount validity using only the non-PII customer_id
    # in any log/error output, never exposing email or phone in messages.
    # Returns a validation status string so bad rows can be filtered out
    # rather than crashing the entire Spark task.
    @udf(returnType=StringType())
    def validate_amount_safe(amount, customer_id):
        if amount < 0:
            # Only customer_id (non-PII surrogate) is referenced in the message.
            # Email and phone are intentionally omitted to prevent PII exposure
            # in Databricks job logs and stderr.
            raise ValueError(
                f"DATA QUALITY VIOLATION: customer_id={customer_id} "
                f"has negative amount"
            )
        return "VALID"

    logger.info("Applying pre-write validation — isolating bad rows before Delta write")

    # Add a validation status column using only customer_id (non-PII).
    df_with_status = df.withColumn(
        "validation_status",
        validate_amount_safe(col("amount"), col("customer_id"))
    )

    try:
        # Materialise the validation pass to separate clean from quarantined rows.
        # cache() avoids re-computing the UDF twice.
        df_with_status.cache()

        # ── Quarantine: capture bad rows (do NOT log PII fields) ──────────────
        df_invalid = df_with_status.filter(col("validation_status") != "VALID")
        invalid_count = df_invalid.count()

        if invalid_count > 0:
            logger.warning(
                "Data quality check found %d record(s) with negative amounts — "
                "routing to quarantine table. customer_ids: %s",
                invalid_count,
                # Log only non-PII identifiers.
                [row.customer_id for row in df_invalid.select("customer_id").collect()]
            )

            # Write quarantine records (schema preserved; no PII logged to stdout).
            df_invalid.write.format("delta").mode("append") \
                .option("overwriteSchema", "true") \
                .save("dbfs:/tmp/pii_etl_demo/quarantine")

            logger.info("Quarantine write completed for %d invalid record(s)", invalid_count)

        # ── Clean path: write only valid rows to the target Delta table ───────
        df_clean = df_with_status.filter(col("validation_status") == "VALID") \
                                  .drop("validation_status")
        clean_count = df_clean.count()

        if clean_count == 0:
            logger.error(
                "Pipeline halted — no valid records remain after data quality filtering. "
                "Total violations: %d. Investigate upstream data source.",
                invalid_count
            )
            raise RuntimeError(
                f"Pipeline aborted: all {invalid_count} record(s) failed validation "
                "(negative amounts). Check quarantine table for details."
            )

        df_clean.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .save("dbfs:/tmp/pii_etl_demo/validated")

        logger.info(
            "Write completed successfully — %d clean record(s) written, "
            "%d record(s) quarantined.",
            clean_count,
            invalid_count
        )

    except RuntimeError:
        # Re-raise pipeline-abort errors so Databricks marks the job FAILED.
        raise

    except Exception as exc:
        # Log a generic failure message — deliberately omit exc details that
        # may carry PII from the UDF ValueError message.
        logger.error(
            "Pipeline failed — unexpected error during validation or write. "
            "Error type: %s. Check Spark driver logs for stack trace.",
            type(exc).__name__
        )
        raise

    finally:
        # Always unpersist the cached DataFrame to free cluster memory.
        try:
            df_with_status.unpersist()
        except Exception:
            pass
