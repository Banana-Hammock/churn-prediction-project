from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load model, scaler, PCA, and expected feature names
model = pickle.load(open('model_pca.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))  # List of feature column names

@app.route('/')
def home():
    return render_template('Customer Churn Prediction App.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'CreditScore': float(request.form['CreditScore']),
            'Geography': request.form['Geography'],
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Tenure': int(request.form['Tenure']),
            'Balance': float(request.form['Balance']),
            'NumOfProducts': int(request.form['NumOfProducts']),
            'HasCrCard': int(request.form['HasCrCard']),
            'IsActiveMember': int(request.form['IsActiveMember']),
            'EstimatedSalary': float(request.form['EstimatedSalary']),
        }

        # One-hot encode as per training
        geography = data['Geography']
        gender = data['Gender']

        geo_dict = {'France': [1, 0, 0], 'Germany': [0, 1, 0], 'Spain': [0, 0, 1]}
        gender_dict = {'Female': [1, 0], 'Male': [0, 1]}  # Gender_Female, Gender_Male

        encoded_geo = geo_dict.get(geography, [0, 0, 0])
        encoded_gender = gender_dict.get(gender, [0, 0])

        final_input = [
            data['CreditScore']
        ] + encoded_geo + encoded_gender + [
            data['Age'],
            data['Tenure'],
            data['Balance'],
            data['NumOfProducts'],
            data['HasCrCard'],
            data['IsActiveMember'],
            data['EstimatedSalary']
        ]

        # Convert to array
        final_input = np.array(final_input).reshape(1, -1)

        # Scale, then PCA transform
        scaled_input = scaler.transform(final_input)
        pca_input = pca.transform(scaled_input)

        # Predict
        prediction = model.predict(pca_input)[0]
        result = "Churn" if prediction == 1 else "No Churn"

        # Return result via redirect
        return redirect(url_for('home', result=result))

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
