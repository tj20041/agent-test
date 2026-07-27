# Databricks Notebook - Test Case 5
# ERROR: SparkException with "Python worker failed to connect back" or "Python exception" 
# Expected log error: "PicklingError" or "PythonException" or "ValueError"

from pyspark.sql.functions import udf, col, struct, array
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import pickle
import sys

# Create data
data = [
    (1, "A", 100.0, ["x", "y", "z"]),
    (2, "B", 200.0, ["a", "b"]),
    (3, "C", 300.0, ["m", "n", "o", "p"]),
    (4, "D", 400.0, ["q", "r"]),
    (5, "E", 500.0, ["s", "t", "u"])
]

df = spark.createDataFrame(data, ["id", "code", "value", "items"])

# ERROR 1: UDF that tries to use unpickleable objects
class CustomProcessor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.cache = {}
    
    def process(self, value):
        # This object can't be pickled
        return value * self.threshold

processor = CustomProcessor(1.5)

@udf(returnType=DoubleType())
def process_value(value):
    # This will fail when trying to serialize processor
    return processor.process(value)  # processor can't be pickled

df1 = df.withColumn(
    "processed_value",
    process_value(col("value"))
)

# ERROR 2: UDF with nested functions and closures
@udf(returnType=ArrayType(StringType()))
def process_items(items):
    # Nested function with closure
    def inner_process(item):
        # This creates complex closure that can't be pickled
        return item.upper() + "_" + str(len(item))
    
    # Lambda with complex closure
    result = list(map(lambda x: inner_process(x), items))
    return result

df2 = df1.withColumn(
    "processed_items",
    process_items(col("items"))
)

# ERROR 3: UDF using non-serializable module-level variables
MODULE_CONFIG = {
    "mode": "strict",
    "threshold": 0.75,
    "mapping": {"A": 1, "B": 2, "C": 3, "D": 4}
}

@udf(returnType=StringType())
def apply_config(code, value):
    # Uses module-level dict that might not pickle properly
    mapping = MODULE_CONFIG["mapping"]
    mode = MODULE_CONFIG["mode"]
    threshold = MODULE_CONFIG["threshold"]
    
    # Complex logic using external data
    if code in mapping:
        mapped_value = mapping[code]
        if value > threshold * 100:
            return f"{mode}_{mapped_value}_{code}"
        else:
            return f"{mode}_low_{mapped_value}"
    else:
        return "unknown"

df3 = df2.withColumn(
    "config_result",
    apply_config(col("code"), col("processed_value"))
)

# ERROR 4: UDF with generator functions
@udf(returnType=ArrayType(DoubleType()))
def generate_sequence(value):
    # Generator that yields values
    def gen():
        for i in range(10):
            yield value * i
    
    # Converting generator to list - may fail in distributed context
    return list(gen())

df4 = df3.withColumn(
    "sequence",
    generate_sequence(col("processed_value"))
)

# ERROR 5: UDF with external library imports inside nested scope
@udf(returnType=DoubleType())
def calculate_statistics(items):
    # Import inside function might fail in distributed execution
    import statistics
    import numpy as np  # numpy might not be available
    
    try:
        # Complex statistics that might fail
        mean = statistics.mean(items)
        # This might fail if numpy not available
        std = np.std(items)
        return mean / std
    except:
        return None

df5 = df4.withColumn(
    "stats_result",
    calculate_statistics(col("processed_items"))
)

# ERROR 6: UDF with multiple return types
@udf(returnType=StringType())
def complex_processing(value, code):
    # This returns different types - will cause serialization issues
    if value > 200:
        return {"status": "high", "value": value}  # Returns dict
    elif code == "B":
        return [value, code, value * 2]  # Returns list
    else:
        return value  # Returns float

df6 = df5.withColumn(
    "complex_result",
    complex_processing(col("processed_value"), col("code"))
)

# ERROR 7: UDF with recursive function that has external reference
external_list = [1, 2, 3, 4, 5]

@udf(returnType=DoubleType())
def recursive_sum(n):
    def rec_helper(x):
        # References external_list - will fail to pickle
        if x == 0:
            return sum(external_list)
        return x + rec_helper(x - 1)
    return rec_helper(int(n))

df7 = df6.withColumn(
    "recursive_sum_result",
    recursive_sum(col("value"))
)

# ERROR 8: UDF trying to access filesystem
@udf(returnType=StringType())
def read_config(id):
    # Trying to read file in UDF - will fail
    with open("/dbfs/mnt/config/config.txt", "r") as f:
        config = f.read()
    return config

df8 = df7.withColumn(
    "config_data",
    read_config(col("id"))
)

# Force execution - will cause Python pickling/serialization errors
df8.show()

# Additional operation that will fail
df8.groupBy("code").agg(
    collect_list("complex_result").alias("results")
).show()
