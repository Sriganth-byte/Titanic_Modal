Titanic Survival Prediction Web App
This is a machine learning-powered web application that predicts the likelihood of survival for passengers aboard the Titanic based on various features such as passenger class, age, sex, family size, and port of embarkation.
The model is trained on the Titanic dataset and is deployed using Flask, allowing users to interact with the app through a simple web interface.

Features
Survival Prediction: The app uses a trained Logistic Regression model to predict whether a passenger survived the Titanic disaster or not.
Interactive Web Interface: Users can input passenger details such as class, age, sex, family size, and port of embarkation to receive a prediction.
Machine Learning Model: The app uses a Logistic Regression model trained on the Titanic dataset from Kaggle.
Data Preprocessing: The app handles missing values, encodes categorical variables, and scales data to prepare it for prediction.

Tech Stack
Frontend:
HTML
CSS (Styled with custom styles for a modern and interactive look)

Backend:
Python
Flask (for creating the web server)

Machine Learning:
Python
scikit-learn (for creating the Logistic Regression model)
Pandas (for data manipulation)
Pickle (for model and encoder persistence)

Setup and Installation
To get the project up and running on your local machine, follow these steps:
1. Clone this repository to your local machine:
git clone https://github.com/Sriganth-byte/Titanic_Modal.git
cd Titanic_Modal

2. Install Dependencies
Install the required dependencies by running:
pip install -r requirements.txt

3. Run the App
Start the Flask server by running:
python app.py

Files and Directory Structure
app.py: Flask backend application that runs the web server and handles user input.

model.py: Python script for training the Logistic Regression model.

index.html: HTML template for the front-end form.

style.css: Custom styles for the web application.

TITANIC.jpeg: Background image used on the website.

model.pkl: The trained Logistic Regression model saved using Pickle.

encoders.pkl: Encoders for transforming categorical variables like 'Sex' and 'Embarked'.

titanic.train.csv: Titanic training dataset (for model training).

titanic.test.csv: Titanic test dataset (for model evaluation).

Data Preprocessing and Model Training
The machine learning model used in this project is a Logistic Regression classifier. It was trained on the Titanic dataset, which includes the following steps:

Handling Missing Data:

Missing values in the Age, Fare, and Embarked columns were filled using the median or mode, respectively.

Label Encoding:

The Sex and Embarked columns were encoded using LabelEncoder from scikit-learn to transform categorical data into numerical values.

Model Training:

A Logistic Regression model was trained on the processed data using scikit-learn.

The model was saved as model.pkl for use in the web app.

Usage
Web App
Input:

The user is prompted to provide passenger details: Pclass, Sex, Age, SibSp, Parch, Fare, and Embarked.

Prediction:

Upon clicking "Predict Survival", the app makes a prediction whether the passenger survived (1) or did not survive (0), based on the trained model.

Model Persistence
Pickle:

Both the trained model (model.pkl) and the label encoders (encoders.pkl) are saved using Pickle. This allows the web app to load the model and encoders without retraining each time.

Deployment
This application can be deployed to platforms like Render, Heroku, or Vercel. The following are general steps for deploying the app:

Create an account on the deployment platform (Render, Heroku, etc.).

Push the code to a Git repository (GitHub, GitLab, etc.).

Link the repository to the platform and set up the deployment environment.

Make sure to add any necessary environment variables or configuration for deployment.

After deployment, visit the app's URL to interact with it online.

Contributing
Feel free to fork this repository, submit issues, and create pull requests. If you would like to contribute, ensure that your changes are well-documented and follow the project's coding style.

Steps for Contributing:
Fork the repository.

Create a new branch for your changes.

Commit your changes.

Push to your forked repository.

Create a pull request.

License
This project is open source and available under the MIT License.
