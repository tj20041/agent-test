# Databricks Notebook - Test Case 1
# ERROR: NullPointerException in UDF when processing null values
# Expected log error: "java.lang.NullPointerException" or "PythonException: An exception was thrown from the UDF"

from pyspark.sql.functions import udf, col, sum, when
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
    # Handle None values for price to prevent TypeError
    if price is None:
        return 0.0  # Return 0.0 discount if price is null
    
    # If category is None, apply a default discount to the non-null price
    if category is None:
        return price * 0.10
    
    # Proceed with category-specific discounts (category is now guaranteed not None)
    if category.upper() == "ELECTRONICS":
        discount = price * 0.20
    elif category.upper() == "CLOTHING":
        discount = price * 0.15
    else:
        discount = price * 0.10
    return discount

# Apply UDF - will now handle null values gracefully
df_with_discount = df.withColumn(
    "discount_amount",
    calculate_discount(col("price"), col("category"))
)

# Another UDF with similar issue but different null scenario
@udf(returnType=StringType())
def categorize_product(product_name, price):
    # Handle None product_name to prevent AttributeError on .upper()
    if product_name is None:
        return "UNKNOWN_PRODUCT"

    # Ensure price is not None for comparison. Assume 0.0 if None.
    # (price here is discount_amount, which calculate_discount now ensures is not None)
    effective_price = price if price is not None else 0.0

    if effective_price > 200:
        return "PREMIUM_" + product_name.upper()
    else:
        return "STANDARD_" + product_name.upper()

df_final = df_with_discount.withColumn(
    "product_category",
    categorize_product(col("product_name"), col("discount_amount"))
)

# Force execution - will now succeed without NullPointerException
df_final.show()

# Additional processing that will now succeed
df_final.groupBy("category").agg(
    sum("discount_amount").alias("total_discount")
).show()
