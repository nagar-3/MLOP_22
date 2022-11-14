from flask import Flask, jsonify, request
import numpy as np
from joblib import load


def predict(data,clf):
    predicted_result=[]
    for i in data:
        predict_test = clf.predict([i])
        predicted_result.append(predict_test)
    return predicted_result

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def run_app():
    best_model = load("svm_Best_Model.joblib")
    values = request.get_json()
    img1 = values['img1']
    img2 = values['img2']
    model_choice=["svm"]
    predicted_result=predict([img1,img2],best_model)
    if predicted_result[0][0]==predicted_result[1][0]:
        response = {"Result ":"Both image are same","Image#1":int(predicted_result[0][0]),"Image#2":int(predicted_result[1][0])}
    else:
        response = {"Result ":"Both Image are different","Image#1":int(predicted_result[0][0]),"Image#2":int(predicted_result[1][0])}
    return jsonify(response), 201
app.run(host='172.17.0.1', port=5000)

