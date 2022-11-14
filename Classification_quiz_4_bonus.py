from flask import Flask, jsonify, request
import numpy as np
import requests
app = Flask(__name__)

@app.route('/svm', methods=['POST'])
def run_app():
    values = request.get_json()
    print(values)
    url = "http://172.17.0.1:5000/predict"
    response = requests.post(url,json={"img1":values["img1"],"img2":values["img2"]})
    response=response.content
    return response, 201

app.run(host='172.17.0.2', port=5000)

