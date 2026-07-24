# Databricks Notebook - Test Case 1
# ERROR: NullPointerException in UDF when processing null values
# Expected log error: "java.lang.NullPointerException" or "PythonException: An exception was thrown from the UDF"

from pyspark.sql.functions import udf, col, when
from pyspark.sql.types import StringType, IntegerType, DoubleType
import pandas as pd

# Create data with nulls in critical columns
data = [
    (1, "Product_A", 100.50, "Electronics", "2024-01-01"),
    (2, "Product_B", None, "Clothing", "2024-01-02"),  # Null price
    (3, "Product_C", 250.00, None, "2024-01-03"),      # Null category
    (4, None, 300.00, "Electronics", None),            # Null name and date
    (5, "Product_E", 450.50, "Clothing", "2024-01-05")
]

df = spark.createDataFrame(data, ["id", "product_name", "price", "category", "date"])

# ERROR: UDF that doesn't handle nulls
@udf(returnType=DoubleType())
def calculate_discount(price, category):
    # This will throw NullPointerException when price or category is None
    if category.upper() == "ELECTRONICS":
        discount = price * 0.20  # price is None for row 2 - will cause NPE
    elif category.upper() == "CLOTHING":
        discount = price * 0.15
    else:
        discount = price * 0.10
    return discount

# Apply UDF - will fail at row with null values
df_with_discount = df.withColumn(
    "discount_amount",
    calculate_discount(col("price"), col("category"))
)

# Another UDF with similar issue but different null scenario
@udf(returnType=StringType())
def categorize_product(product_name, price):
    # Null check missing for price
    if price > 200 and product_name is not None:
        return "PREMIUM_" + product_name.upper()
    elif price <= 200 and product_name is not None:
        return "STANDARD_" + product_name.upper()
    else:
        # This branch will cause NPE when product_name is None
        return product_name.upper() + "_UNKNOWN"  # product_name.upper() fails if None

df_final = df_with_discount.withColumn(
    "product_category",
    categorize_product(col("product_name"), col("discount_amount"))
)

# Force execution - will throw NullPointerException
df_final.show()

# Additional processing that will fail
df_final.groupBy("category").agg(
    sum("discount_amount").alias("total_discount")
).show()
