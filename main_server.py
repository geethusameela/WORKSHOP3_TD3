import requests
from flask import Flask, jsonify

app = Flask(__name__)

def get_prediction(url):
    response = requests.get(url + '/predict')
    data = response.json()
    return data['prediction']

@app.route('/aggregate_predict')
def aggregate_predict():
    # Get predictions from Model 1 and Model 2
    prediction1 = get_prediction('http://localhost:5000')
    prediction2 = get_prediction('http://localhost:5001')

    # Aggregate predictions (simple average)
    consensus_prediction = (prediction1 + prediction2) / 2

    return jsonify({'consensus_prediction': consensus_prediction})

if __name__ == '__main__':
    app.run(debug=True)