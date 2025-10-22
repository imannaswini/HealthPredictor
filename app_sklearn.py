from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# This line enables Cross-Origin Resource Sharing (CORS) for the app
CORS(app)

# Load the trained scikit-learn models
try:
    diabetes_model = joblib.load('models/diabetes_model.joblib')
    heart_model = joblib.load('models/heart_model.joblib')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    diabetes_model = None
    heart_model = None

# API endpoint for Diabetes prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if not diabetes_model:
        return jsonify({"error": "Diabetes model not loaded"}), 500
    
    data = request.get_json()
    # Convert the incoming JSON to a pandas DataFrame
    df = pd.DataFrame([data])
    prediction = diabetes_model.predict(df)
    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

# API endpoint for Heart Disease prediction
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    if not heart_model:
        return jsonify({"error": "Heart disease model not loaded"}), 500
    
    data = request.get_json()
    # Convert the incoming JSON to a pandas DataFrame
    df = pd.DataFrame([data])
    prediction = heart_model.predict(df)
    risk = "High Risk" if int(prediction[0]) == 1 else "Low Risk"
    # Return the prediction and a human-readable risk level as JSON
    return jsonify({'prediction': int(prediction[0]), 'risk': risk})

if __name__ == '__main__':
    # Running the app on port 5001
    app.run(port=5001)