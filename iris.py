import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import joblib

# Data loading and exploration (optional)
try:
    data = pd.read_csv("IRIS.csv")
    print(data.head())  # Uncomment to view the first few rows of the data
except FileNotFoundError:
    print("Error: IRIS.csv file not found. Please ensure the file exists and is in the correct location.")
    exit(1)

# Handle missing values (if any)
data.dropna(inplace=True)

# Encode the categorical variable
le = LabelEncoder()
data["species"] = le.fit_transform(data["species"])

# Split data into features and target variable
X = data.drop("species", axis=1)
y = data["species"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# Function to make predictions
def predict(sepal_length, sepal_width, petal_length, petal_width):
    # Convert input to a NumPy array
    new_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])

    # Encode new data using the same encoder as the training data
    new_data["species"] = le.transform(new_data["species"])
    new_data.drop("species", axis=1, inplace=True)  # Drop the encoded column

    # Make prediction
    prediction = model.predict(new_data)[0]

    # Decode the prediction back to the original species name
    predicted_species = le.inverse_transform([prediction])[0]

    return {"predicted_species": predicted_species}

# Create a Flask app
app = Flask(__name__)

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def make_prediction():
    # Get the request data
    data = request.get_json()

    # Extract flower measurements
    sepal_length = data.get("sepal_length")
    sepal_width = data.get("sepal_width")
    petal_length = data.get("petal_length")
    petal_width = data.get("petal_width")
# Ensure all required parameters are present
    if not all([sepal_length, sepal_width, petal_length, petal_width]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Make prediction and return the response
    try:
        prediction = predict(sepal_length, sepal_width, petal_length, petal_width)
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
