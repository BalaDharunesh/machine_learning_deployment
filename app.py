
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and imputation means
model = joblib.load('logistic_regression_model.joblib')
imputation_means = joblib.load('imputation_means.joblib')

# Define the columns that were used for training the model
# Assuming the order is the same as X_train
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'No data provided in the request'}), 400

        # Convert input data to DataFrame
        # Ensure single record is handled correctly and columns are in order
        input_df = pd.DataFrame([json_data], columns=feature_columns)

        # Apply imputation for 0 values in specified columns
        for col in columns_to_impute:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace(0, imputation_means.get(col))
        
        # Handle any potential remaining NaN values by imputing with the loaded means
        # This ensures robustness if a new NaN appears due to other data issues
        for col in feature_columns:
            if col in imputation_means and input_df[col].isnull().any():
                input_df[col] = input_df[col].fillna(imputation_means.get(col))

        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        return jsonify({
            'prediction': int(prediction[0]),
            'probability_of_diabetes': float(probability[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
