# Import the necessary libraries
from pyspark.sql import SparkSession

# 1. Create a SparkSession
spark = SparkSession.builder \
    .appName("HealthRiskPrediction") \
    .getOrCreate()

print("Spark session created successfully!")

# --- Load and Inspect Heart Disease Dataset ---
try:
    heart_df = spark.read.csv('heart.csv', header=True, inferSchema=True)
    print("\n--- Heart Disease Dataset Loaded ---")
    print("Schema:")
    heart_df.printSchema()
    print("Preview:")
    heart_df.show(5)

except Exception as e:
    print(f"\nError loading heart.csv: {e}")
    print("Please make sure 'heart.csv' is in the same folder as this script.")


# --- Load and Inspect Diabetes Dataset ---
try:
    diabetes_df = spark.read.csv('diabetes.csv', header=True, inferSchema=True)
    print("\n--- Diabetes Dataset Loaded ---")
    print("Schema:")
    diabetes_df.printSchema()
    print("Preview:")
    diabetes_df.show(5)

except Exception as e:
    print(f"\nError loading diabetes.csv: {e}")
    print("Please make sure 'diabetes.csv' is in the same folder as this script.")


# Stop the Spark session to release resources
spark.stop()