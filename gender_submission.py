# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('gender_submission.csv')

# Preprocessing - For simplicity, assuming no missing values and only binary classification
X = data.drop(columns=['Survived'])  # Assuming 'Survived' is the target variable
y = data['Survived']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Flask application
app = Flask(__name__)

# API route for prediction
@app.route('/predict', methods=['GET'])
def predict():
    # Get input parameters
    # For simplicity, assuming the input parameters are passed as query parameters
    # You may need to adjust this based on your actual input requirements
    passenger_id = int(request.args.get('PassengerId'))
    # Assuming other features are also passed in a similar way
    
    # Make prediction
    prediction = model.predict([[passenger_id]])[0]  # Assuming 'passenger_id' is the feature for prediction
    
    # Prepare API response
    response = {
        'PassengerId': passenger_id,
        'Prediction': prediction
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
