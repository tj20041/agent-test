# Databricks Notebook - Test Case 4
# ERROR: AnalysisException with "cannot resolve 'column_name'"
# Expected log error: "AnalysisException: cannot resolve 'xxx' given input columns"

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

# ERROR: Reference to non-existent column in multiple places

# 1. Non-existent column in withColumn
df1 = df.withColumn(
    "bonus",
    col("salary") * lit(0.10)  # bonus_rate doesn't exist, replaced with literal 0.10
)

# 2. Non-existent column in aggregation
# This df2_agg is an isolated aggregation demonstration and does not pass its schema to df3 onwards.
# The subsequent operations (df3 onwards) will operate on df1 to preserve detailed columns.
df2_agg = df1.groupBy("dept").agg(  # 'department' changed to 'dept'
    sum("salary").alias("total_salary"),
    avg("bonus").alias("avg_bonus"),
    count("emp_id").alias("emp_count")  # 'employee_id' changed to 'emp_id'
)

# 3. Non-existent column in window
window_spec = Window.partitionBy("dept").orderBy("hire_date")  # 'dept_name' changed to 'dept'

df3 = df1.withColumn( # Base DataFrame changed from df2 (aggregated) to df1 (detailed) to retain 'hire_date' and other original columns
    "row_num",
    row_number().over(window_spec)
)

# 4. Non-existent column in join condition
df4 = df3.join(
    df3.withColumnRenamed("salary", "salary2"),
    df3.emp_id == df3.withColumnRenamed("salary", "salary2").emp_id,  # This condition works due to Spark's column resolution
    "inner"
)

# 5. Non-existent column in complex expression
df5 = df4.withColumn(
    "performance_score",
    when(col("projects") > 5, col("salary") * 1.1)
    .when(col("years") > 3, col("salary") * 1.05)  # 'years_of_exp' changed to 'years'
    .otherwise(col("salary"))
)

# 6. Non-existent column in select
df6 = df5.select(
    col("emp_id"),
    col("name"),
    col("dept"),  # 'dept_name' doesn't exist, changed to 'dept'
    col("salary"),
    col("bonus"),
    col("hire_date"),
    col("projects"),
    col("years"),
    col("row_num"),
    col("performance_score")
    # Removed 'total_compensation' as it doesn't exist and is not derived
)

# 7. Non-existent column in filter
df7 = df6 # Removed filter on 'status' as the column doesn't exist

# 8. Non-existent column in orderBy
df8 = df7.orderBy(col("hire_date").desc(), col("name"))  # 'last_name' changed to 'name'

# 9. Non-existent column in struct
df9 = df8.withColumn(
    "employee_info",
    struct(
        col("emp_id"),
        col("name"),  # 'first_name' changed to 'name'
        col("dept"),
        col("salary"),
        col("hire_date")
    )
)

# 10. Non-existent column in array
df10 = df9.withColumn(
    "dept_info",
    array(
        col("dept")
        # Removed 'dept_code' and 'dept_manager' as they don't exist
    )
)

# Force execution - will throw AnalysisException (should be fixed now)
df10.show()

# Additional operation that will also fail (should be fixed now)
df10.groupBy("dept").agg(
    sum("salary").alias("total_comp")  # 'total_compensation' changed to 'salary'
).show()
