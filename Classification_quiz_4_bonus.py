
# Standard scientific Python imports
# Import datasets, classifiers and performance metrics
from flask import Flask, jsonify, request
import numpy as np

app = Flask(__name__)

@app.route('/svm', methods=['POST'])
def run_app():
    values = request.get_json()
    url = 'http://10.0.2.15:5000/predict'
    response = requests.post(url,json=values)
    return jsonify(response), 201

app.run(host='0.0.0.0', port=5000)
