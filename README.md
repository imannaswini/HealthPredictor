

# Health Risk Predictor 

A full-stack web application that uses machine learning to predict the risk of diabetes and heart disease based on user-provided health data.

This project integrates a **Python-Flask backend** (serving `scikit-learn` models) with a **vanilla JavaScript frontend**, demonstrating a complete end-to-end data science application.


## ✨ Features

  * **Dual-Model Prediction:** Predicts risk for two different chronic conditions: Diabetes and Heart Disease.
  * **Dynamic UI:** The user interface dynamically generates the correct input form based on the selected model.
  * **Decoupled Architecture:** A clean, full-stack design with a standalone frontend that communicates with a backend REST API.
  * **Real-Time Predictions:** Users receive instant risk assessments from the trained machine learning models.

-----

## 🛠️ Tech Stack

### Backend

  * **Python 3.13**
  * **Flask**: For the API server.
  * **Flask-CORS**: To handle cross-origin requests from the frontend.
  * **scikit-learn**: For training and using the `LogisticRegression` models.
  * **pandas**: For loading and cleaning the datasets.
  * **joblib**: For saving and loading the trained models.

### Frontend

  * **HTML5**
  * **CSS3** (Embedded in `<style>` tag)
  * **JavaScript (ES6+)**: For dynamic UI logic and using the `fetch` API.

-----

## 📂 Project Structure

```
.
├── models/
│   ├── diabetes_model.joblib
│   └── heart_model.joblib
├── app_sklearn.py        # The Flask API server
├── train_sklearn.py      # Script to train and save the models
├── index.html            # The all-in-one frontend (HTML, CSS, JS)
├── diabetes.csv          # Dataset
├── heart.csv             # Dataset
└── README.md             # You are here
```

-----

## 🏃 How to Run Locally

Follow these steps to set up and run the project on your local machine.

### Prerequisites

  * [Python 3.10+](https://www.python.org/downloads/)
  * [pip](https://pip.pypa.io/en/stable/installation/)

### 1\. Clone the Repository

```bash
git clone https://your-repository-url.git
cd health-risk-predictor
```

### 2\. Set Up the Backend

First, create a virtual environment and install the required Python packages.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install Flask Flask-CORS scikit-learn pandas
```

### 3\. Train the Models

Before running the server, you need to train the models using the provided datasets.

```bash
# This will create the 'models' folder with the .joblib files
python train_sklearn.py
```

### 4\. Run the API Server

Start the Flask server. It will run on `http://127.0.0.1:5001`.

```bash
python app_sklearn.py
```

### 5\. Launch the Frontend

In your file explorer, simply **double-click the `index.html` file** to open it in your default web browser.

You can now select a model, enter the data, and get live predictions from your local server\!

-----

## 📖 API Endpoints

The Flask server provides two API endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict/diabetes` | Accepts JSON data for diabetes features and returns a prediction. |
| `POST` | `/predict/heart` | Accepts JSON data for heart disease features and returns a prediction. |

**Example `POST` Request (Diabetes):**

```bash
curl -X POST http://127.0.0.1:5001/predict/diabetes \
     -H "Content-Type: application/json" \
     -d '{"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31}'
```

**Example Response:**

```json
{
  "prediction": 0
}
```
