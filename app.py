from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler

# Initialize Flask app
app = Flask(__name__)

# Initialize Spark Session with Direct Configuration
spark = SparkSession.builder \
    .appName("PredictionAPI") \
    .config("spark.driver.extraJavaOptions", "-Dhadoop.home.dir=C:/hadoop") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

# Load Both Models on Startup
try:
    diabetes_model_path = "models/diabetes_model"
    diabetes_model = LogisticRegressionModel.load(diabetes_model_path)
    print("Diabetes model loaded successfully!")
except Exception as e:
    print(f"Error loading diabetes model: {e}")
    diabetes_model = None

try:
    heart_model_path = "models/heart_model"
    heart_model = LogisticRegressionModel.load(heart_model_path)
    print("Heart disease model loaded successfully!")
except Exception as e:
    print(f"Error loading heart disease model: {e}")
    heart_model = None


# Define Feature Columns for Both Models
DIABETES_FEATURE_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
HEART_FEATURE_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']


# API Endpoint for Diabetes Prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model:
        return jsonify({"error": "Diabetes model not loaded"}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # <-- FIX: Convert generator to a list containing a tuple
        input_data = [tuple(data[col] for col in DIABETES_FEATURE_COLS)]
        df = spark.createDataFrame(input_data, DIABETES_FEATURE_COLS)
        
        assembler = VectorAssembler(inputCols=DIABETES_FEATURE_COLS, outputCol='features')
        df_assembled = assembler.transform(df)

        prediction = diabetes_model.transform(df_assembled)
        result = prediction.select("prediction").first()['prediction']

        return jsonify({"prediction": int(result)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# API Endpoint for Heart Disease Prediction
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    if not heart_model:
        return jsonify({"error": "Heart disease model not loaded"}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # <-- FIX: Convert generator to a list containing a tuple
        input_data = [tuple(data[col] for col in HEART_FEATURE_COLS)]
        df = spark.createDataFrame(input_data, HEART_FEATURE_COLS)
        
        assembler = VectorAssembler(inputCols=HEART_FEATURE_COLS, outputCol='features')
        df_assembled = assembler.transform(df)

        prediction = heart_model.transform(df_assembled)
        result = prediction.select("prediction").first()['prediction']
        
        risk = "High Risk" if int(result) == 1 else "Low Risk"
        return jsonify({"prediction": int(result), "risk": risk})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)