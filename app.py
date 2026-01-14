import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and imputation means
model = joblib.load("logistic_regression_model.joblib")
imputation_means = joblib.load("imputation_means.joblib")

# Feature order MUST match training data
feature_columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Columns where 0 should be treated as missing
columns_to_impute = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Diabetes Prediction API is running 🚀",
        "endpoint": "/predict",
        "method": "POST"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Ensure all required features are present
        missing_features = [col for col in feature_columns if col not in json_data]
        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Convert input JSON to DataFrame (single row)
        input_df = pd.DataFrame([json_data], columns=feature_columns)

        # Replace 0 with mean values for selected columns
        for col in columns_to_impute:
            input_df[col] = input_df[col].replace(0, imputation_means.get(col))

        # Fill any remaining NaN values safely
        for col in feature_columns:
            if col in imputation_means:
                input_df[col] = input_df[col].fillna(imputation_means.get(col))

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability_of_diabetes": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Required for Render / Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
