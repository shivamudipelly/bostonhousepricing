import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pickle files to the model
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # This should be a dictionary of feature values

    print("Received Data:", data)

    # Extract feature values in the correct order
    input_features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                      'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
    
    # Create NumPy array from values in correct order
    input_array = np.array([data[feature] for feature in input_features]).reshape(1, -1)

    print("Input Array:", input_array)

    # Scale and predict
    new_data = scaler.transform(input_array)
    output = model.predict(new_data)

    print("Prediction:", output[0])
    return jsonify({'prediction': float(output[0])})


if __name__ == "__main__":
    app.run(debug=True)
