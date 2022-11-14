# Standard scientific Python imports
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utills.utill import mean_std
from utills.model import svm_model,DecisionTree_model,depth
from utills.utill import label_comp
from joblib import dump, load
from flask import Flask, jsonify, request
import numpy as np

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#define 5 different splits of train/test/valid.
train_frac_split=0.8
dev_frac_split=0.1


# Split data into  train, test and dev subsets
def train_dev_test_split(data, label, train_frac, dev_frac):
    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(data, label, test_size=dev_test_frac, shuffle=True)
    x_test, x_dev, y_test, y_dev = train_test_split(x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True)
    return x_train, y_train, x_dev, y_dev, x_test,y_test



print("\n\n Model Comparsion in progress..................\n")

def save_model(clf,best_param_configi,model_path):
    if type(clf) == svm.SVC:
        model_type = "svm"
    best_model_name = model_type + "_Best_Model" + ".joblib"
    model_path = best_model_name
    dump(clf, model_path)
    return model_path


def load_model(actual_model_path):
        best_model = load(actual_model_path)
        predicted = best_model.predict(x_test)


def training(depth,model_choice):
    for i in range(5):
        best_acc=0
        X_train, y_train, X_dev, y_dev, X_test,y_test=train_dev_test_split(data,digits.target,train_frac_split,dev_frac_split)
        for j in model_choice:
            for k in range(len(depth)):
                clf,parms=svm_model(k)
                clf.fit(X_train, y_train)
                predict_dev = clf.predict(X_dev)
                predict_test = clf.predict(X_test)
                acc_dev =round(accuracy_score(y_dev, predict_dev),3)
                acc_test =round(accuracy_score(y_test, predict_test),3)
                #comapring accuracy for each occurance to get the best model.
                if acc_test>best_acc:
                   best_acc=acc_test
                   best_model=clf
                   best_param_config=k
    #save_model(clf,best_param_config,None)
    return best_model

def predict(data,clf):
    predicted_result=[]
    for i in data:
        predict_test = clf.predict([i])
        predicted_result.append(predict_test)
    return predicted_result
    
        
    
app = Flask(__name__)

@app.route('/svm', methods=['POST'])
def run_app():
    values = request.get_json()
    img1 = values['img1']
    img2 = values['img2']
    model_choice=["svm"]
    best_model=training(depth,model_choice)
    predicted_result=predict([img1,img2],best_model)
    if predicted_result[0][0]==predicted_result[1][0]:
        response = {"Result ":"Both image are same","Image#1":int(predicted_result[0][0]),"Image#2":int(predicted_result[1][0])}
    else:
        response = {"Result ":"Both Image are different","Image#1":int(predicted_result[0][0]),"Image#2":int(predicted_result[1][0])}
    return jsonify(response), 201

app.run(host='0.0.0.0', port=5000)

