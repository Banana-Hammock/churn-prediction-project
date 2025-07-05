from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['Time_spent_Alone']),
            float(request.form['Social_event_attendance']),
            float(request.form['Going_outside']),
            float(request.form['Friends_circle_size']),
            float(request.form['Post_frequency']),
            1 if request.form['Stage_fear'] == 'Yes' else 0,
            1 if request.form['Drained_after_socializing'] == 'Yes' else 0
        ]
        prediction = model.predict([inputs])[0]
        personality = "Introvert" if prediction == 1 else "Extrovert"
    except Exception as e:
        personality = f"Error: {e}"

    return render_template('form.html', result=personality)

if __name__ == '__main__':
    app.run(debug=True)
