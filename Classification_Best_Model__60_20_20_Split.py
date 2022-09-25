# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:54:27 2022

@author: mnagar
"""

# Standard scientific Python imports
import matplotlib.pyplot as plt
import statistics

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

gamma = [0.001,0.005,0.001,0.0005,0.0001]
c_value=[0.2,0.3,0.7,5]


print("\n")
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


train_fr=0.6
test_fr=0.1
dev_fr=0.1
dev_test_fr=test_fr+dev_fr

#Define Best accuracy
best_acc=0

# Split data into 80% train and 10% test and 10% val subsets
X_train, X_dev, y_train, y_dev = train_test_split(data, digits.target, test_size=dev_test_fr, shuffle=True)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=dev_fr/dev_test_fr, shuffle=False)
acc_valt,acc_testt,acc_traint=[],[],[]

print("Model Training in progress..................\n")
for i,j in zip(gamma,c_value):
  for j in c_value:
    #Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=i,C=j)
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_dev)
    predicted2 = clf.predict(X_test)
    predicted3 = clf.predict(X_train)
    acc_val=round(accuracy_score(y_dev, predicted),2)
    acc_valt.append(acc_val)
    acc_test=round(accuracy_score(y_test, predicted2),2)
    acc_testt.append(acc_test)
    acc_train=round(accuracy_score(y_train, predicted3),2)
    acc_traint.append(acc_train)
    print("[ Gamma: {}, C: {} ] ===> Train Accuracy : {}  ,Dev Accuracy : {} , Test Accuracy : {} ".format(i,j,acc_train,acc_val,acc_test))

    if acc_test>best_acc:
        best_acc=acc_test
        best_prms={"Gamma":i ,"C":j}
        val_test_acc={"Train_acc":acc_train,"val_acc" :acc_val, "Test_acc":acc_test}
        Report_val=f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test, predicted2)}\n"
        Report_test=f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test, predicted2)}\n"
        Report_train=f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_train, predicted3)}\n"

    #plt.show()
print("\n")
print("------------------Best Results ----------------------------------")
print(val_test_acc)
print("\n----------------------------------------------------------------")
print("Best Parms Gamma: {} , C: {}".format(best_prms["Gamma"],best_prms["C"]))
print("\n------------------------- train Report ---------------------------")
print(Report_train)
print("\n------------------------- dev Report ---------------------------")
print(Report_val)
print("\n------------------------- Test Report --------------------------")
print(Report_test)
print("--------------------------Mean , Mode , Median, {Train data}---------------------------------")
print("Mean {}  , median : {} , Mode {}".format(statistics.mean(acc_traint),statistics.median(acc_traint),statistics.mode(acc_traint)))
print("\n")
print("--------------------------Mean , Mode , Median, {Dev Data} ---------------------------------")
print("Mean {}  , median : {} , Mode {}".format(statistics.mean(acc_valt),statistics.median(acc_valt),statistics.mode(acc_valt)))
print("\n")
print("--------------------------Mean , Mode , Median, {Test Data}---------------------------------")
print("Mean {}  , median : {} , Mode {}".format(statistics.mean(acc_testt),statistics.median(acc_testt),statistics.mode(acc_testt)))
print("\n")
