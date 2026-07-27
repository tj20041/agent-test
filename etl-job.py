# Databricks Notebook - Test Case 4
# ERROR: AnalysisException with "cannot resolve 'column_name'"
# Expected log error: "AnalysisException: cannot resolve 'xxx' given input columns"

from pyspark.sql.functions import col, sum, avg, count, max, min, when, struct, array, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

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
    col("salary") * col("bonus_rate")  # bonus_rate doesn't exist
)

# 2. Non-existent column in aggregation
df2 = df1.groupBy("department").agg(  # department doesn't exist (should be dept)
    sum("salary").alias("total_salary"),
    avg("bonus").alias("avg_bonus"),
    count("employee_id").alias("emp_count")  # employee_id doesn't exist (should be emp_id)
)

# 3. Non-existent column in window
from pyspark.sql.window import Window
window_spec = Window.partitionBy("dept_name").orderBy("hire_date")  # dept_name doesn't exist

df3 = df2.withColumn(
    "row_num",
    row_number().over(window_spec)
)

# 4. Non-existent column in join condition
df4 = df3.join(
    df3.withColumnRenamed("salary", "salary2"),
    df3.emp_id == df3.withColumnRenamed("salary", "salary2").emp_id,  # This might work
    "inner"
)

# 5. Non-existent column in complex expression
df5 = df4.withColumn(
    "performance_score",
    when(col("projects") > 5, col("salary") * 1.1)
    .when(col("years_of_exp") > 3, col("salary") * 1.05)  # years_of_exp doesn't exist
    .otherwise(col("salary"))
)

# 6. Non-existent column in select
df6 = df5.select(
    col("emp_id"),
    col("name"),
    col("dept_name"),  # doesn't exist
    col("total_compensation")  # doesn't exist
)

# 7. Non-existent column in filter
df7 = df6.filter(col("status") == "ACTIVE")  # status doesn't exist

# 8. Non-existent column in orderBy
df8 = df7.orderBy(col("hire_date").desc(), col("last_name"))  # last_name doesn't exist

# 9. Non-existent column in struct
df9 = df8.withColumn(
    "employee_info",
    struct(
        col("emp_id"),
        col("first_name"),  # doesn't exist (should be name)
        col("dept"),
        col("salary"),
        col("hire_date")
    )
)

# 10. Non-existent column in array
df10 = df9.withColumn(
    "dept_info",
    array(
        col("dept"),
        col("dept_code"),  # doesn't exist
        col("dept_manager")  # doesn't exist
    )
)

# Force execution - will throw AnalysisException
df10.show()

# Additional operation that will also fail
df10.groupBy("dept").agg(
    sum("total_compensation").alias("total_comp")  # total_compensation doesn't exist
).show()
