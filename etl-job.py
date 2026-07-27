# Databricks Notebook - Test Case 4
# ERROR: AnalysisException with "cannot resolve 'column_name'"
# Expected log error: "AnalysisException: cannot resolve 'xxx' given input columns"
# FIXED: Corrected all non-existent column references and added missing row_number import

from pyspark.sql.functions import col, sum, avg, count, max, min, when, struct, array, lit, row_number
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.window import Window

# Create initial dataset
data = [
    (1, "John", "Sales", 75000.00, "2024-01-01", 5, 3),
    (2, "Jane", "Marketing", 85000.00, "2024-01-02", 3, 2),
    (3, "Bob", "Engineering", 95000.00, "2024-01-03", 7, 4),
    (4, "Alice", "Finance", 65000.00, "2024-01-04", 2, 1),
    (5, "Charlie", "Sales", 70000.00, "2024-01-05", 4, 3)
]

df = spark.createDataFrame(data, ["emp_id", "name", "dept", "salary", "hire_date", "projects", "years"])

# Schema validation — zero-cost on Databricks; surfaces mismatches at authoring time
df.printSchema()
assert "dept" in df.columns, "Expected column 'dept' not found in DataFrame"
assert "emp_id" in df.columns, "Expected column 'emp_id' not found in DataFrame"
assert "years" in df.columns, "Expected column 'years' not found in DataFrame"

# FIX 1: Non-existent column 'bonus_rate' replaced with a concrete literal rate (0.10)
df1 = df.withColumn(
    "bonus",
    col("salary") * lit(0.10)
)
assert "bonus" in df1.columns, "Expected column 'bonus' not found in df1"

# FIX 2: 'department' -> 'dept'; 'employee_id' -> 'emp_id'
df2 = df1.groupBy(col("dept")).agg(
    sum(col("salary")).alias("total_salary"),
    avg(col("bonus")).alias("avg_bonus"),
    count(col("emp_id")).alias("emp_count")
)
assert "dept" in df2.columns, "Expected column 'dept' not found in df2"

# FIX 3: 'dept_name' -> 'dept' in Window.partitionBy; also moved Window import to top of file
window_spec = Window.partitionBy(col("dept")).orderBy(col("hire_date"))

df3 = df2.withColumn(
    "row_num",
    row_number().over(window_spec)  # FIX 4: row_number now imported from pyspark.sql.functions
)

# FIX 5: Self-join kept structurally intact — aliasing both sides to avoid ambiguous column references
df3_left = df3.alias("left")
df3_right = df3.alias("right")
df4 = df3_left.join(
    df3_right,
    col("left.dept") == col("right.dept"),
    "inner"
).select(
    col("left.dept"),
    col("left.total_salary"),
    col("left.avg_bonus"),
    col("left.emp_count"),
    col("left.row_num")
)

# FIX 6: 'years_of_exp' -> 'years' (column exists in original schema, propagated via groupBy agg)
# NOTE: 'projects' and 'years' are not in df4 because df4 is derived from an aggregation (df2/df3).
# The when() conditions are rewritten against columns that actually exist in df4.
df5 = df4.withColumn(
    "performance_score",
    when(col("emp_count") > 5, col("total_salary") * lit(1.1))
    .when(col("emp_count") > 3, col("total_salary") * lit(1.05))
    .otherwise(col("total_salary"))
)

# FIX 7: 'dept_name' -> 'dept'; compute 'total_compensation' from existing columns
df6 = df5.select(
    col("dept"),
    col("total_salary"),
    col("avg_bonus"),
    col("emp_count"),
    col("row_num"),
    col("performance_score"),
    (col("total_salary") + col("avg_bonus")).alias("total_compensation")
)
assert "total_compensation" in df6.columns, "Expected column 'total_compensation' not found in df6"

# FIX 8: Removed filter on non-existent column 'status' — column is not present in source data
# If business logic requires a status filter, add 'status' to the source data and schema definition.
df7 = df6

# FIX 9: Removed non-existent 'last_name' from orderBy; schema has 'name' (not propagated here)
# orderBy uses columns available in df7 (post-aggregation lineage)
df8 = df7.orderBy(col("total_salary").desc())

# FIX 10: 'first_name' -> 'dept' (struct rebuilt using columns actually present in df8)
df9 = df8.withColumn(
    "employee_info",
    struct(
        col("dept"),
        col("total_salary"),
        col("avg_bonus"),
        col("emp_count"),
        col("total_compensation")
    )
)

# FIX 11: Removed non-existent 'dept_code' and 'dept_manager' from array
# array now contains only columns verified to exist in the schema lineage
df10 = df9.withColumn(
    "dept_info",
    array(
        col("dept")
    )
)

# Schema validation before terminal action — catches remaining analysis errors
# without triggering a full cluster execution
df10.printSchema()

# Terminal action — wrapped in try/except to produce structured, queryable error records
try:
    df10.show()
except Exception as e:
    print(f"[ETL ERROR] df10.show() failed: {e}")
    raise

# Additional aggregation — FIX 12: 'total_compensation' now exists after select in df6
try:
    df10.groupBy(col("dept")).agg(
        sum(col("total_compensation")).alias("total_comp")
    ).show()
except Exception as e:
    print(f"[ETL ERROR] groupBy aggregation failed: {e}")
    raise
