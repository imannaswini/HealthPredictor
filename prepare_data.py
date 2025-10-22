from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler

# --- Boilerplate Spark Session ---
spark = SparkSession.builder \
    .appName("HealthRiskPrediction-Prepare") \
    .getOrCreate()

#---------------------------------------------------
# Process 1: Prepare Heart Disease Data
#---------------------------------------------------
try:
    heart_df = spark.read.csv('heart.csv', header=True, inferSchema=True)

    # Define feature columns (using a simple numeric subset for now)
    heart_feature_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    
    # Create the binary 'label' column from 'num'
    heart_df = heart_df.withColumn('label', when(col('num') > 0, 1).otherwise(0))
    
    # Assemble the feature vector
    heart_assembler = VectorAssembler(inputCols=heart_feature_cols, outputCol='features')
    final_heart_df = heart_assembler.transform(heart_df).select('features', 'label').dropna()

    # Show the result
    print("--- 1. Prepared Heart Disease Data ---")
    final_heart_df.show(5, truncate=False)

except Exception as e:
    print(f"Could not process heart.csv. Error: {e}")


#---------------------------------------------------
# Process 2: Prepare Diabetes Data
#---------------------------------------------------
try:
    diabetes_df = spark.read.csv('diabetes.csv', header=True, inferSchema=True)
    
    # Define feature columns (all columns except the target 'Outcome')
    diabetes_feature_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    # The 'Outcome' column is already our label. Let's rename it for consistency.
    diabetes_df = diabetes_df.withColumnRenamed('Outcome', 'label')

    # Assemble the feature vector
    diabetes_assembler = VectorAssembler(inputCols=diabetes_feature_cols, outputCol='features')
    final_diabetes_df = diabetes_assembler.transform(diabetes_df).select('features', 'label').dropna()

    # Show the result
    print("\n--- 2. Prepared Diabetes Data ---")
    final_diabetes_df.show(5, truncate=False)

except Exception as e:
    print(f"Could not process diabetes.csv. Error: {e}")


spark.stop()