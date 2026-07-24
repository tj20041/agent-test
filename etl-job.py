# Databricks Notebook - Test Case 1
# ERROR: NullPointerException in UDF when processing null values
# Expected log error: "java.lang.NullPointerException" or "PythonException: An exception was thrown from the UDF"

from pyspark.sql.functions import col, when, upper, lit, concat, sum
from pyspark.sql.types import StringType, IntegerType, DoubleType
import pandas as pd # pandas is not used for the fix but might be for other parts of the script

# Create data with nulls in critical columns
data = [
    (1, "Product_A", 100.50, "Electronics", "2024-01-01"),
    (2, "Product_B", None, "Clothing", "2024-01-02"),  # Null price
    (3, "Product_C", 250.00, None, "2024-01-03"),      # Null category
    (4, None, 300.00, "Electronics", None),            # Null name and date
    (5, "Product_E", 450.50, "Clothing", "2024-01-05")
]

df = spark.createDataFrame(data, ["id", "product_name", "price", "category", "date"])

# FIX: Refactor calculate_discount to use native Spark SQL functions for null-safe operations
# and improved performance. This replaces the problematic UDF that caused TypeError.
df_with_discount = df.withColumn(
    "discount_amount",
    when(upper(col("category")) == "ELECTRONICS", col("price") * 0.20)
    .when(upper(col("category")) == "CLOTHING", col("price") * 0.15)
    .otherwise(col("price") * 0.10) # Handles other categories and null categories correctly
)

# FIX: Refactor categorize_product to use native Spark SQL functions for null-safe operations
# and improved performance. This replaces the problematic UDF.
df_final = df_with_discount.withColumn(
    "product_category",
    when(col("product_name").isNull(), lit("UNKNOWN_PRODUCT_NAME"))
    .when(col("discount_amount").isNull(), lit("PRICE_UNDETERMINED"))
    .when(col("discount_amount") > 200, concat(lit("PREMIUM_"), upper(col("product_name"))))
    .otherwise(concat(lit("STANDARD_"), upper(col("product_name"))))
)

# Force execution - now should pass without TypeError or PythonException
df_final.show()

# Additional processing that will now work correctly
df_final.groupBy("category").agg(
    sum("discount_amount").alias("total_discount")
).show()
