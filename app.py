from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('loan_outcome_model.pkl')

# Create a Flask app
app = Flask(__name__) # Changed _name_ to __name__

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input
    data = request.get_json()

    # Convert to DataFrame
    input_data = pd.DataFrame([data])

    # Ensure data matches the model's expected features
    input_data = pd.get_dummies(input_data, drop_first=True).reindex(columns=X_train.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    # Return the result
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__': # Changed _name_ to __name__
    app.run(debug=True)