import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from

app = Flask(__name__)

# ---------------- SWAGGER CONFIG (FIXED) ----------------
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Diabetes Prediction API",
        "description": "Predict diabetes using a trained Logistic Regression model",
        "version": "1.0.0"
    }
}

Swagger(app, config=swagger_config, template=swagger_template)
# -------------------------------------------------------

# Load trained model and imputation means
model = joblib.load("logistic_regression_model.joblib")
imputation_means = joblib.load("imputation_means.joblib")

# Feature order (must match training)
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
        "swagger_url": "/swagger"
    })


@app.route("/predict", methods=["POST"])
@swag_from({
    "tags": ["Prediction"],
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "Pregnancies": {"type": "integer", "example": 2},
                    "Glucose": {"type": "number", "example": 120},
                    "BloodPressure": {"type": "number", "example": 70},
                    "SkinThickness": {"type": "number", "example": 20},
                    "Insulin": {"type": "number", "example": 79},
                    "BMI": {"type": "number", "example": 25.4},
                    "DiabetesPedigreeFunction": {"type": "number", "example": 0.35},
                    "Age": {"type": "integer", "example": 31}
                },
                "required": [
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age"
                ]
            }
        }
    ],
    "responses": {
        200: {
            "description": "Prediction result",
            "schema": {
                "type": "object",
                "properties": {
                    "prediction": {"type": "integer"},
                    "probability_of_diabetes": {"type": "number"}
                }
            }
        },
        400: {"description": "Invalid input"},
        500: {"description": "Server error"}
    }
})
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        missing_features = [col for col in feature_columns if col not in data]
        if missing_features:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        input_df = pd.DataFrame([data], columns=feature_columns)

        # Replace zero values with mean
        for col in columns_to_impute:
            input_df[col] = input_df[col].replace(0, imputation_means.get(col))

        # Fill NaNs
        for col in feature_columns:
            if col in imputation_means:
                input_df[col] = input_df[col].fillna(imputation_means.get(col))

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability_of_diabetes": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
