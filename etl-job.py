# Databricks Notebook - Test Case 4 (FIXED)
# Original error: AnalysisException / UNRESOLVED_COLUMN.WITH_SUGGESTION
# Root cause: multiple withColumn/groupBy/select/orderBy/window/join operations
# referenced columns that were never created in the source schema
# (emp_id, name, dept, salary, hire_date, projects, years).
# This version fixes every unresolved column reference and keeps the
# transformation chain internally consistent so each stage only uses
# columns that actually exist in its input DataFrame.

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

# 1. Fixed: 'bonus_rate' did not exist -- use a literal rate instead, and add
#    a 'status' column up front so downstream filters have a real column to use.
df1 = df.withColumn(
    "bonus",
    col("salary") * lit(0.05)
).withColumn(
    "status",
    lit("ACTIVE")
)

# 2. Fixed: 'department' -> 'dept' and 'employee_id' -> 'emp_id' to match the
#    actual schema created above. This produces a department-level summary
#    DataFrame that is independent of the row-level chain below.
df2 = df1.groupBy("dept").agg(
    sum("salary").alias("total_salary"),
    avg("bonus").alias("avg_bonus"),
    count("emp_id").alias("emp_count")
)

# 3. Fixed: 'dept_name' -> 'dept'. Also, the window function must operate on
#    row-level data (df1), not on the aggregated df2, since df2 no longer has
#    per-employee columns such as hire_date.
window_spec = Window.partitionBy("dept").orderBy("hire_date")

df3 = df1.withColumn(
    "row_num",
    row_number().over(window_spec)
)

# 4. Fixed: self-join now uses a subset of columns from the right side with
#    unique aliases (emp_id2/salary2) to avoid ambiguous column references
#    after the join.
df4 = df3.join(
    df3.select(col("emp_id").alias("emp_id2"), col("salary").alias("salary2")),
    df3.emp_id == col("emp_id2"),
    "inner"
).drop("emp_id2")

# 5. Fixed: 'years_of_exp' -> 'years' to match the actual schema column.
df5 = df4.withColumn(
    "performance_score",
    when(col("projects") > 5, col("salary") * 1.1)
    .when(col("years") > 3, col("salary") * 1.05)
    .otherwise(col("salary"))
)

# 6. Fixed: 'dept_name' -> 'dept'; 'total_compensation' is now computed
#    explicitly before being selected. 'status' and 'hire_date' are carried
#    through since they are required by later steps (filter/orderBy).
df6 = df5.withColumn(
    "total_compensation",
    col("salary") + col("bonus")
).select(
    col("emp_id"),
    col("name"),
    col("dept"),
    col("status"),
    col("hire_date"),
    col("total_compensation")
)

# 7. Fixed: 'status' column now exists (added in df1), so this filter is valid.
df7 = df6.filter(col("status") == "ACTIVE")

# 8. Fixed: 'last_name' does not exist -- use 'name' instead.
df8 = df7.orderBy(col("hire_date").desc(), col("name"))

# 9. Fixed: 'first_name' -> 'name' to match the actual schema column.
df9 = df8.withColumn(
    "employee_info",
    struct(
        col("emp_id"),
        col("name"),
        col("dept"),
        col("total_compensation"),
        col("hire_date")
    )
)

# 10. Fixed: 'dept_code' and 'dept_manager' do not exist and were removed --
#     the array now only references the real 'dept' column.
df10 = df9.withColumn(
    "dept_info",
    array(
        col("dept")
    )
)

# Force execution - all column references above now resolve correctly.
df10.show()

# Additional operation - 'total_compensation' now exists in df10, so this
# aggregation resolves correctly.
df10.groupBy("dept").agg(
    sum("total_compensation").alias("total_comp")
).show()
