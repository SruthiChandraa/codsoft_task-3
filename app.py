# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('sales_prediction_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the JSON request
    inputs = request.get_json()  # Correctly parse JSON data
    inputs_df = pd.DataFrame([inputs])  # Convert to DataFrame
    
    # Convert numeric inputs if necessary
    for column in inputs_df.columns:
        inputs_df[column] = pd.to_numeric(inputs_df[column], errors='coerce')
    
    # Perform prediction
    prediction = model.predict(inputs_df)
    
    # Return prediction result
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
