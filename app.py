from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Cancer Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # convert inputs to float
    features = np.array(data["features"], dtype=float).reshape(1, -1)

    prediction = model.predict(features)

    result = int(prediction[0])

    return jsonify({
        "prediction": result
    })

if __name__ == "__main__":
    app.run(debug=True)