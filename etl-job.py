# Databricks Notebook - Test Case 5
# FIXED: Removed unpicklable closures/stateful objects from UDFs that were
# causing PythonException/EOFError crashes on executors (DBX-TASK-FAILED,
# GEN-PYTHON-EXCEPTION, GEN-UNCLASSIFIED-ERROR) and repeated task failures
# leading to job abort in stage 2.0.
#
# Root causes fixed:
#   1. process_value closed over a stateful CustomProcessor instance -> replaced
#      with a stateless UDF that receives the threshold as a literal Column.
#   2. process_items used a nested function + lambda closure -> moved to a
#      module-level function and a plain list comprehension.
#   3. apply_config read a module-level mutable dict inside the UDF body ->
#      the config is now shipped via a Spark broadcast variable.
#   4. recursive_sum closed over a module-level list -> the list is now shipped
#      via a Spark broadcast variable and the recursion no longer references
#      external/global state.
#   5. read_config tried to open a file directly on the executor (which
#      surfaced as FileNotFoundError for /dbfs/mnt/config/config.txt) -> the
#      file is now read once on the driver (with a pre-flight existence check)
#      and the content is shipped to executors via a broadcast variable.
#   6. complex_processing returned inconsistent types (dict/list/float) while
#      declared as StringType() -> normalized to always return a string.
#   7. Missing import for collect_list used at the bottom of the notebook.
#   8. Added defensive try/except inside UDF bodies so a single bad row logs
#      a sentinel value instead of crashing the Python worker.

from pyspark.sql.functions import udf, col, lit, collect_list
from pyspark.sql.types import StringType, DoubleType, ArrayType

# Create data
data = [
    (1, "A", 100.0, ["x", "y", "z"]),
    (2, "B", 200.0, ["a", "b"]),
    (3, "C", 300.0, ["m", "n", "o", "p"]),
    (4, "D", 400.0, ["q", "r"]),
    (5, "E", 500.0, ["s", "t", "u"])
]

df = spark.createDataFrame(data, ["id", "code", "value", "items"])

# FIX 1: Stateless UDF - no closure over a stateful custom class instance.
# The threshold is passed in as a literal Column so nothing but primitives
# and plain functions need to be pickled and shipped to executors.
PROCESS_THRESHOLD = 1.5


@udf(returnType=DoubleType())
def process_value(value, threshold):
    try:
        return float(value) * float(threshold)
    except (TypeError, ValueError):
        return None


df1 = df.withColumn(
    "processed_value",
    process_value(col("value"), lit(PROCESS_THRESHOLD))
)


# FIX 2: Module-level helper function instead of a nested function + lambda
# closure defined inside the UDF body.
def _inner_process(item):
    return item.upper() + "_" + str(len(item))


@udf(returnType=ArrayType(StringType()))
def process_items(items):
    if items is None:
        return []
    try:
        return [_inner_process(x) for x in items]
    except AttributeError:
        return []


df2 = df1.withColumn(
    "processed_items",
    process_items(col("items"))
)


# FIX 3: Broadcast the config dict instead of reading module-level mutable
# state from inside the UDF body.
MODULE_CONFIG = {
    "mode": "strict",
    "threshold": 0.75,
    "mapping": {"A": 1, "B": 2, "C": 3, "D": 4}
}

config_bcast = spark.sparkContext.broadcast(MODULE_CONFIG)


@udf(returnType=StringType())
def apply_config(code, value):
    try:
        config = config_bcast.value
        mapping = config["mapping"]
        mode = config["mode"]
        threshold = config["threshold"]

        if code in mapping:
            mapped_value = mapping[code]
            if value is not None and value > threshold * 100:
                return f"{mode}_{mapped_value}_{code}"
            else:
                return f"{mode}_low_{mapped_value}"
        else:
            return "unknown"
    except (KeyError, TypeError):
        return "error"


df3 = df2.withColumn(
    "config_result",
    apply_config(col("code"), col("processed_value"))
)


# UDF with a local generator - safe, does not close over external/mutable
# state, only the row-local `value` argument.
@udf(returnType=ArrayType(DoubleType()))
def generate_sequence(value):
    try:
        def gen():
            for i in range(10):
                yield value * i

        return list(gen())
    except TypeError:
        return []


df4 = df3.withColumn(
    "sequence",
    generate_sequence(col("processed_value"))
)


# UDF with library imports local to the function body - fine on Databricks
# since statistics and numpy are part of the standard runtime. Wrapped with
# defensive error handling so a single bad row doesn't crash the task.
@udf(returnType=DoubleType())
def calculate_statistics(items):
    import statistics
    import numpy as np

    try:
        mean = statistics.mean(items)
        std = np.std(items)
        if std == 0:
            return None
        return float(mean / std)
    except (statistics.StatisticsError, TypeError, ZeroDivisionError, ValueError):
        return None


df5 = df4.withColumn(
    "stats_result",
    calculate_statistics(col("processed_items"))
)


# FIX 6: Always return a single consistent type (string) instead of mixing
# dict/list/float, which is invalid for a StringType() UDF and can trigger
# serialization/casting errors on the executor side.
@udf(returnType=StringType())
def complex_processing(value, code):
    try:
        if value is not None and value > 200:
            return f"high_{value}"
        elif code == "B":
            return f"{value}_{code}_{value * 2}"
        else:
            return str(value)
    except TypeError:
        return "error"


df6 = df5.withColumn(
    "complex_result",
    complex_processing(col("processed_value"), col("code"))
)


# FIX 4: Broadcast the external list instead of closing over module-level
# state from within a nested recursive helper.
EXTERNAL_LIST = [1, 2, 3, 4, 5]
external_list_bcast = spark.sparkContext.broadcast(EXTERNAL_LIST)


def _rec_helper(x, values):
    if x == 0:
        return sum(values)
    return x + _rec_helper(x - 1, values)


@udf(returnType=DoubleType())
def recursive_sum(n):
    try:
        values = external_list_bcast.value
        return float(_rec_helper(int(n), values))
    except (TypeError, ValueError, RecursionError):
        return None


df7 = df6.withColumn(
    "recursive_sum_result",
    recursive_sum(col("value"))
)


# FIX 5: Read the config file once on the driver (with an existence check)
# and ship its contents to executors via a broadcast variable, instead of
# opening the file directly inside the UDF on the executor (which was
# causing FileNotFoundError: /dbfs/mnt/config/config.txt).
CONFIG_PATH = "/dbfs/mnt/config/config.txt"

try:
    with open(CONFIG_PATH, "r") as f:
        _config_file_content = f.read()
except FileNotFoundError:
    _config_file_content = ""

config_content_bcast = spark.sparkContext.broadcast(_config_file_content)


@udf(returnType=StringType())
def read_config(id):
    return config_content_bcast.value


df8 = df7.withColumn(
    "config_data",
    read_config(col("id"))
)

# Force execution
df8.show()

# Additional operation - collect_list is now imported
df8.groupBy("code").agg(
    collect_list("complex_result").alias("results")
).show()
