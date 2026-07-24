# Databricks Notebook - Test Case 3
# ERROR: SparkException with "java.lang.OutOfMemoryError" or "Shuffle memory limit exceeded"
# Expected log error: "SparkException: Job aborted due to stage failure" and "OutOfMemoryError"

from pyspark.sql.functions import col, explode, split, concat, lit, when, rand, struct, sum, avg, count, collect_list, udf, row_number
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from pyspark.sql.window import Window
import random

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
df.cache().count()

# ERROR: Massive shuffling due to skewed data
# This will cause shuffle memory issues

# Multiple joins with skewed data
df1 = df.withColumnRenamed("key", "key1")
# Renamed conflicting columns in df2 to avoid ambiguity after join
df2 = df.withColumnRenamed("key", "key2") \
        .withColumnRenamed("value", "value_df2") \
        .withColumnRenamed("count", "count_df2") \
        .withColumnRenamed("price", "price_df2") \
        .withColumnRenamed("group", "group_df2") \
        .withColumnRenamed("items", "items_df2")

# Join on skewed key - will cause data skew in shuffle
joined_df = df1.join(df2, df1.key1 == df2.key2, "inner")

# Explode the arrays - multiplies data
# 'value', 'count', 'price', 'group', 'items' are now unambiguously from df1
exploded_df = joined_df.select(
    col("key1"),
    col("value"),
    col("count"),
    col("price"),
    col("group"),
    explode(col("items")).alias("item")
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
result2 = result1.join(
    result1.withColumnRenamed("total_count", "total_count2"),
    result1.key1 == result1.withColumnRenamed("total_count", "total_count2").key1,
    "inner"
)

# Window functions on skewed data
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
    for item in items:
        for i in range(10):  # Multiply data even more
            result.append(f"{item}_{i}")
    return result

result5 = result4.withColumn(
    "processed_items",
    process_items(col("items_list"))
)

# Explode again
final_df = result5.select(
    col("key1"),
    col("group"),
    col("total_price"),
    explode(col("processed_items")).alias("processed_item")
)

# Force execution - will cause OutOfMemoryError
final_df.count()

# Additional collect will cause driver OOM
all_data = final_df.collect()
