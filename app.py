from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

def clean_input_data(data):
    data['Age'] = float(data['Age']) if data['Age'] else 29.0
    data['Fare'] = float(data['Fare']) if data['Fare'] else 32.0

    data['Sex'] = encoders['Sex'].transform([data['Sex']])[0]
    data['Embarked'] = encoders['Embarked'].transform([data['Embarked']])[0]

    return np.array([[data['Pclass'], data['Sex'], data['Age'], data['SibSp'],
data['Parch'], data['Fare'], data['Embarked']]])

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': request.form['Age'],
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Fare': request.form['Fare'],
            'Embarked': request.form['Embarked']
        }

        cleaned = clean_input_data(input_data)
        prediction = model.predict(cleaned)[0]

        if prediction == 1:
            result = "✅ You have survived the Titanic crisis! 😄"
        else:
            result = "❌ Sorry, you did not survive the disaster. 😢"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        print(f"❌ Error: {e}")
        return render_template('index.html', prediction_text="⚠️ Error processing input. Please check your entries.")

if __name__ == '__main__':
    app.run(debug=True)
