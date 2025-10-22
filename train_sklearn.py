import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# --- Train Diabetes Model ---
try:
    print("Training Diabetes Model...")
    diabetes_df = pd.read_csv('diabetes.csv')
    X_diabetes = diabetes_df.drop('Outcome', axis=1)
    y_diabetes = diabetes_df['Outcome']
    diabetes_model = LogisticRegression(max_iter=1000)
    diabetes_model.fit(X_diabetes, y_diabetes)
    joblib.dump(diabetes_model, 'models/diabetes_model.joblib')
    print("Diabetes model saved successfully.")
except Exception as e:
    print(f"Error training diabetes model: {e}")

print("-" * 30)

# --- Train Heart Disease Model ---
try:
    print("Training Heart Disease Model...")
    heart_df = pd.read_csv('heart.csv')
    
    # --- FIX: Drop rows with missing values ---
    heart_df.dropna(inplace=True)
    
    heart_df['num'] = heart_df['num'].apply(lambda x: 1 if x > 0 else 0)
    heart_feature_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    X_heart = heart_df[heart_feature_cols]
    y_heart = heart_df['num']
    heart_model = LogisticRegression(max_iter=1000)
    heart_model.fit(X_heart, y_heart)
    joblib.dump(heart_model, 'models/heart_model.joblib')
    print("Heart Disease model saved successfully.")
except Exception as e:
    print(f"Error training heart disease model: {e}")