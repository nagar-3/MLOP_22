# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 09:54:27 2022

@author: mnagar
"""

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np

digits = datasets.load_digits()

gamma = [0.001,0.005,0.001,0.0005,0.0001]
c_value=[0.2,0.3,0.7,1,3,5,7,10]


#check image size.
print("\n Original Image size: {} \n".format(digits.images.shape))

#Setting paramter for image resizing.
new_size=[.5,1.5,2]

for i in new_size:
    #image re-sizing
    image_resized_1 =np.array([resize(k, (int(digits.images[0].shape[0]*i),int(digits.images[0].shape[1]*i)),anti_aliasing=True)  for k in digits.images])
    print("---------------------------------------------------------------------")
    print("New resized image shape ",image_resized_1.shape,"\n")
    # flatten the images
    n_samples = len(image_resized_1)
    data = image_resized_1.reshape((n_samples, -1))

    #data Split parms
    train_fr=0.8
    test_fr=0.1
    dev_fr=0.1
    dev_test_fr=test_fr+dev_fr

    #Define Best accuracy 
    best_acc=0

    # Split data into 80% train and 10% test and 10% val subsets
    X_train, X_dev, y_train, y_dev = train_test_split(data, digits.target, test_size=dev_test_fr, shuffle=True)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, test_size=dev_fr/dev_test_fr, shuffle=False)


    print("Model Training in progress..................\n")
    for i,j in zip(gamma,c_value):
        for j in c_value:
            #Create a classifier: a support vector classifier
            clf = svm.SVC(gamma=i,C=j)
            # Learn the digits on the train subset
            clf.fit(X_train, y_train)
            
            # Predict the value of the digit on the test/dev subset
            predicted = clf.predict(X_dev)
            predicted2 = clf.predict(X_test)
            
            #calcualting accuracy score
            acc_val=round(accuracy_score(y_dev, predicted),2)
            acc_test=round(accuracy_score(y_test, predicted2),2)
            
            print("[ Gamma: {}, C: {} ] ===> Dev Accuracy : {} , Test Accuracy : {} ".format(i,j,acc_val,acc_test))

            #Comapring accuray. 
            if acc_test>best_acc:
                best_acc=acc_test
                best_prms={"Gamma":i ,"C":j}
                val_test_acc={"val_acc":acc_val, "Test_acc":acc_test}
                Report_val=f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test, predicted2)}\n"
                Report_test=f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test, predicted2)}\n"
    #Print best Results
    print("\n")
    print("------------------Best Results ----------------------------------")
    print(val_test_acc)
    print("\n----------------------------------------------------------------")
    print("Best Parms Gamma: {} , C: {}".format(best_prms["Gamma"],best_prms["C"]))
    print("\n------------------------- dev Report ---------------------------")
    print(Report_val)
    print("\n------------------------- Test Report --------------------------")
    print(Report_test)

