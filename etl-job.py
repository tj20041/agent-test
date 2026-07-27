# Databricks Notebook - Test Case 3 (FIXED)
# Original error: org.apache.spark.sql.AnalysisException: [AMBIGUOUS_REFERENCE] Reference `value` is ambiguous, could be: [`value`, `value`]. SQLSTATE: 42704
# Root cause: df1 and df2 were both derived from the same source DataFrame 'df' with only
# the join key renamed ('key1'/'key2'). All other columns ('value', 'count', 'price',
# 'group', 'items') remained identically named on both sides of the join, so unqualified
# references such as col('value') after the join could not be resolved to a single column.
# The same bug pattern existed in the result2 self-join (result1 joined with a renamed
# copy of itself, still sharing 'key1', 'group', etc.). This caused the query to abort
# during analysis (before any shuffle/OOM stage was reached), and the secondary mlflow
# ReplAwareSparkDataSourceListener stack trace was a downstream side effect of that
# aborted SQL execution, not an independent root cause.
#
# Fixes applied:
#  1. Disambiguated every overlapping column between df1/df2 before joining by aliasing
#     all non-key columns with '1'/'2' suffixes, then re-aliasing back to clean names
#     in the subsequent select() so downstream logic is unaffected.
#  2. Fixed the equivalent bug in the result2 self-join by building the renamed
#     DataFrame once (result1_renamed) with uniquely aliased columns instead of calling
#     withColumnRenamed() twice inline inside the join condition.
#  3. Added the missing imports (sum, avg, count, collect_list, row_number, udf, Window)
#     required by the aggregation/window/UDF stages further down the pipeline.
#  4. Enabled Adaptive Query Execution skew-join handling since the dataset is
#     intentionally skewed (90% of rows under 10 keys), reducing the risk of shuffle
#     memory/OOM failures now that the ambiguous-reference bug is fixed.
#  5. Wrapped the Spark actions in try/except blocks that log the specific exception
#     message before re-raising, so future regressions surface clearly instead of only
#     as a generic mlflow listener stack trace.

from pyspark.sql.functions import (
    col, explode, split, concat, lit, when, rand, struct,
    sum, avg, count, collect_list, row_number, udf
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from pyspark.sql.window import Window
import random

# Enable Adaptive Query Execution and skew-join handling to mitigate the intentional
# data skew (90% of rows under 10 keys) used in this workload.
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")


# Generate large dataset with skewed data
def generate_skewed_data(spark, num_records=500000):
    data = []
    for i in range(num_records):
        # Create data skew - 90% of data goes to 10% of keys
        if i < 450000:  # 90% of data
            key = random.randint(1, 10)  # Only 10 keys get 90% of data
        else:
            key = random.randint(11, 1000)  # Remaining keys get 10% of data

        data.append((
            key,
            f"value_{i}",
            random.randint(1, 100),
            round(random.uniform(10, 1000), 2),
            f"group_{key % 50}",
            [f"item_{j}" for j in range(random.randint(1, 100))]  # Variable size arrays
        ))
    return spark.createDataFrame(data, ["key", "value", "count", "price", "group", "items"])


df = generate_skewed_data(spark, 300000)

try:
    df.cache().count()
except Exception as e:
    print(f"Failed while caching/counting base dataframe 'df': {e}")
    raise

# Multiple joins with skewed data
# NOTE: df1 and df2 are both derived from the same base DataFrame 'df'. To avoid the
# AMBIGUOUS_REFERENCE error, every non-key column is explicitly aliased with a '1'/'2'
# suffix before the join so Spark can always resolve which side a column came from.
df1 = df.select(
    col("key").alias("key1"),
    col("value").alias("value1"),
    col("count").alias("count1"),
    col("price").alias("price1"),
    col("group").alias("group1"),
    col("items").alias("items1")
)

df2 = df.select(
    col("key").alias("key2"),
    col("value").alias("value2"),
    col("count").alias("count2"),
    col("price").alias("price2"),
    col("group").alias("group2"),
    col("items").alias("items2")
)

# Join on skewed key - will cause data skew in shuffle (mitigated by AQE skew join above)
joined_df = df1.join(df2, df1.key1 == df2.key2, "inner")

# Explode the arrays - multiplies data
# Every column referenced below is now unambiguous because it was uniquely aliased above.
exploded_df = joined_df.select(
    col("key1"),
    col("value1").alias("value"),
    col("count1").alias("count"),
    col("price1").alias("price"),
    col("group1").alias("group"),
    explode(col("items1")).alias("item")
)

# Multiple aggregations on skewed data
result1 = exploded_df.groupBy("key1", "group").agg(
    sum("count").alias("total_count"),
    sum("price").alias("total_price"),
    avg("price").alias("avg_price"),
    count("item").alias("item_count"),
    collect_list("item").alias("items_list")
)

# Another join with itself
# NOTE: build the renamed copy of result1 ONCE, with every non-join column uniquely
# aliased, rather than calling withColumnRenamed() twice inline inside the join
# condition (which previously created two different unaliased DataFrame instances and
# reintroduced the AMBIGUOUS_REFERENCE bug).
result1_renamed = result1.select(
    col("key1").alias("key1_2"),
    col("group").alias("group_2"),
    col("total_count").alias("total_count2"),
    col("total_price").alias("total_price2"),
    col("avg_price").alias("avg_price2"),
    col("item_count").alias("item_count2"),
    col("items_list").alias("items_list2")
)

result2 = result1.join(
    result1_renamed,
    result1.key1 == result1_renamed.key1_2,
    "inner"
)

# Window functions on skewed data
# 'key1' and 'total_price' are unambiguous single columns coming from result1 in result2.
window_spec = Window.partitionBy("key1").orderBy(col("total_price").desc())
result3 = result2.withColumn(
    "rank_by_price",
    row_number().over(window_spec)
).withColumn(
    "cumulative_sum",
    sum("total_price").over(Window.partitionBy("key1").orderBy("total_price"))
)

# Another repartition causing more shuffling
result4 = result3.repartition(1000, "key1")


# Complex UDF that creates more memory pressure
@udf(returnType=ArrayType(StringType()))
def process_items(items):
    # This will create large intermediate arrays
    result = []
    if items is None:
        return result
    for item in items:
        for i in range(10):  # Multiply data even more
            result.append(f"{item}_{i}")
    return result


result5 = result4.withColumn(
    "processed_items",
    process_items(col("items_list2"))
)

# Explode again
final_df = result5.select(
    col("key1"),
    col("group"),
    col("total_price"),
    explode(col("processed_items")).alias("processed_item")
)

# Force execution
try:
    final_df.count()
except Exception as e:
    print(f"Failed while counting 'final_df': {e}")
    raise

# Collect results (bounded workload; consider limiting rows in production to avoid driver OOM)
try:
    all_data = final_df.collect()
except Exception as e:
    print(f"Failed while collecting 'final_df' to the driver: {e}")
    raise
