from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

def train_and_evaluate(df, feature_cols, label_col, model_name, model_save_path):
    """A helper function to train, evaluate, and save a model."""
    
    # Prepare data
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features', handleInvalid="skip")
    final_df = assembler.transform(df).select('features', label_col).withColumnRenamed(label_col, 'label').dropna()

    # Split Data
    train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=2025)

    # Train Model
    lr = LogisticRegression(featuresCol='features', labelCol='label')
    print(f"Training the {model_name} model...")
    lr_model = lr.fit(train_data)
    print("Training complete.")

    # Evaluate
    predictions = lr_model.transform(test_data)
    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())
    print(f"--- {model_name} Model Accuracy: {accuracy:.4f} ---")
    
    # --- NEW: Save the model ---
    # Overwrite if it already exists
    if os.path.exists(model_save_path):
        import shutil
        shutil.rmtree(model_save_path)
    lr_model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("-" * 40)

# --- Main Execution ---
if __name__ == "__main__":
    spark = SparkSession.builder.appName("HealthRiskPrediction-SaveModels").getOrCreate()

    # --- Process & Save Heart Disease Model ---
    try:
        heart_df = spark.read.csv('heart.csv', header=True, inferSchema=True)
        heart_df = heart_df.withColumn('label', when(col('num') > 0, 1).otherwise(0))
        heart_feature_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
        train_and_evaluate(heart_df, heart_feature_cols, 'label', "Heart Disease", "models/heart_model")
    except Exception as e:
        print(f"Error in Heart Disease pipeline: {e}")

    # --- Process & Save Diabetes Model ---
    try:
        diabetes_df = spark.read.csv('diabetes.csv', header=True, inferSchema=True)
        diabetes_feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        train_and_evaluate(diabetes_df, diabetes_feature_cols, 'Outcome', "Diabetes", "models/diabetes_model")
    except Exception as e:
        print(f"Error in Diabetes pipeline: {e}")

    spark.stop()
    