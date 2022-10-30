# [Digits classification]

Reff # https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py  
Reff# https://scikit-learn.org/stable/modules/tree.html#classification

## [Model Comparison]
**Accuracy for SVM and Decision Tree Classifier for each Model(each Split case)**

**SVM Vs Decision Tree**
```
Model#1:SPlit case#1: train_frac:dev_frac:(0.8, 0.1)  
Model#2:SPlit case#2: train_frac:dev_frac:(0.7, 0.1)  
Model#3:SPlit case#3: train_frac:dev_frac:(0.6, 0.1)  
Model#4:SPlit case#4: train_frac:dev_frac:(0.4, 0.2)  
Model#5:SPlit case#5: train_frac:dev_frac:(0.5, 0.2)  

--------------------------------------------------------------------------------------------------
 svm_acc_model#1    : [0.972, 0.989, 0.989, 0.989, 0.972, 0.989, 0.989, 0.989, 0.961]
 DTr_acc_model#1    : [0.626, 0.883, 0.86, 0.877, 0.872, 0.877, 0.877, 0.888, 0.877]
 svm_acc_model#2    : [0.975, 0.981, 0.989, 0.992, 0.975, 0.981, 0.989, 0.992, 0.969]
 DTr_acc_model#2    : [0.686, 0.864, 0.864, 0.856, 0.883, 0.878, 0.861, 0.886, 0.878]
 svm_acc_model#3    : [0.97, 0.978, 0.983, 0.985, 0.97, 0.978, 0.983, 0.985, 0.965]
 DTr_acc_model#3    : [0.673, 0.87, 0.881, 0.878, 0.865, 0.872, 0.878, 0.878, 0.861]
 svm_acc_model#4    : [0.974, 0.978, 0.986, 0.989, 0.974, 0.978, 0.986, 0.989, 0.958]
 DTr_acc_model#4    : [0.751, 0.8, 0.801, 0.825, 0.797, 0.803, 0.791, 0.808, 0.811]
 svm_acc_model#5    : [0.961, 0.974, 0.991, 0.993, 0.961, 0.974, 0.991, 0.993, 0.957]
 DTr_acc_model#5    : [0.727, 0.788, 0.768, 0.781, 0.785, 0.777, 0.772, 0.781, 0.783]
 -----------------------------------------------------------------------------------------
```
**Computing the mean and standard deviations of both the classifier's performances.**

```
---------------------------------------------------------------------------
        SVM_Model_mean  SVM_Model_std  DecisionTree_mean  DecisionTree_std
Models
1                0.982          0.011              0.849             0.084
2                0.983          0.008              0.851             0.063
3                0.977          0.007              0.851             0.067
4                0.979          0.010              0.799             0.020
5                0.977          0.015              0.774             0.019
---------------------------------------------------------------------------
```
  
 ***Model Training and Testing results*** 

```
Model#1 Best Results:
Processing step###  1  ##-------------------------------------------------------------------------------------------
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.967  Test Acc : 0.972  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.972  Test Acc : 0.989  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.972  Test Acc : 0.989  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.983  Test Acc : 0.989  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.967  Test Acc : 0.972  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.972  Test Acc : 0.989  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.972  Test Acc : 0.989  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.983  Test Acc : 0.989  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.8, 0.1)  Val Acc : 0.967  Test Acc : 0.961  Gamma :0.0005  C :0.2

------------------Best Results ----------------------------------
Best Parms SVM Gamma: 0.0005  C: 0.2  train_frac:dev_frac:(0.8, 0.1)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=0.3, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        25
           2       1.00      1.00      1.00        18
           3       1.00      1.00      1.00        19
           4       1.00      1.00      1.00        13
           5       1.00      0.94      0.97        17
           6       0.95      1.00      0.97        18
           7       0.94      1.00      0.97        17
           8       1.00      1.00      1.00        17
           9       1.00      0.94      0.97        16

    accuracy                           0.99       179
   macro avg       0.99      0.99      0.99       179
weighted avg       0.99      0.99      0.99       179


DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.696  Test Acc : 0.626  Depth : 5
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.834  Test Acc : 0.883  Depth : 10
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.862  Test Acc : 0.86   Depth : 15
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.851  Test Acc : 0.877  Depth : 20
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.856  Test Acc : 0.872  Depth : 25
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.845  Test Acc : 0.877  Depth : 30
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.851  Test Acc : 0.877  Depth : 35
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.867  Test Acc : 0.888  Depth : 40
DTree # Split_Ratio: (0.8, 0.1)  Val Acc : 0.856  Test Acc : 0.877  Depth : 45

----------------------------------------------------------------
Best DecisionTree Depth: 45     train_frac:dev_frac:(0.8, 0.1)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=0.3, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        25
           2       1.00      1.00      1.00        18
           3       1.00      1.00      1.00        19
           4       1.00      1.00      1.00        13
           5       1.00      0.94      0.97        17
           6       0.95      1.00      0.97        18
           7       0.94      1.00      0.97        17
           8       1.00      1.00      1.00        17
           9       1.00      0.94      0.97        16

    accuracy                           0.99       179
   macro avg       0.99      0.99      0.99       179
weighted avg       0.99      0.99      0.99       179



Processing step###  2  ##-------------------------------------------------------------------------------------------
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.972  Test Acc : 0.975  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.983  Test Acc : 0.981  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.983  Test Acc : 0.989  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.983  Test Acc : 0.992  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.972  Test Acc : 0.975  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.983  Test Acc : 0.981  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.983  Test Acc : 0.989  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.983  Test Acc : 0.992  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.7, 0.1)  Val Acc : 0.972  Test Acc : 0.969  Gamma :0.0005  C :0.2

------------------Best Results ----------------------------------
Best Parms SVM Gamma: 0.0005  C: 0.2  train_frac:dev_frac:(0.7, 0.1)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        41
           1       0.97      1.00      0.99        38
           2       1.00      1.00      1.00        46
           3       1.00      1.00      1.00        24
           4       1.00      1.00      1.00        36
           5       1.00      1.00      1.00        32
           6       0.97      1.00      0.98        28
           7       0.98      1.00      0.99        45
           8       1.00      0.94      0.97        35
           9       1.00      0.97      0.99        35

    accuracy                           0.99       360
   macro avg       0.99      0.99      0.99       360
weighted avg       0.99      0.99      0.99       360


DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.678  Test Acc : 0.686  Depth : 5
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.833  Test Acc : 0.864  Depth : 10
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.844  Test Acc : 0.864  Depth : 15
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.833  Test Acc : 0.856  Depth : 20
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.844  Test Acc : 0.883  Depth : 25
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.839  Test Acc : 0.878  Depth : 30
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.844  Test Acc : 0.861  Depth : 35
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.856  Test Acc : 0.886  Depth : 40
DTree # Split_Ratio: (0.7, 0.1)  Val Acc : 0.867  Test Acc : 0.878  Depth : 45

----------------------------------------------------------------
Best DecisionTree Depth: 45     train_frac:dev_frac:(0.7, 0.1)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        41
           1       0.97      1.00      0.99        38
           2       1.00      1.00      1.00        46
           3       1.00      1.00      1.00        24
           4       1.00      1.00      1.00        36
           5       1.00      1.00      1.00        32
           6       0.97      1.00      0.98        28
           7       0.98      1.00      0.99        45
           8       1.00      0.94      0.97        35
           9       1.00      0.97      0.99        35

    accuracy                           0.99       360
   macro avg       0.99      0.99      0.99       360
weighted avg       0.99      0.99      0.99       360



Processing step###  3  ##-------------------------------------------------------------------------------------------
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.983  Test Acc : 0.97   Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.983  Test Acc : 0.978  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.994  Test Acc : 0.983  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.994  Test Acc : 0.985  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.983  Test Acc : 0.97   Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.983  Test Acc : 0.978  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.994  Test Acc : 0.983  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.994  Test Acc : 0.985  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.6, 0.1)  Val Acc : 0.978  Test Acc : 0.965  Gamma :0.0005  C :0.2

------------------Best Results ----------------------------------
Best Parms SVM Gamma: 0.0005  C: 0.2  train_frac:dev_frac:(0.6, 0.1)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        50
           1       1.00      1.00      1.00        59
           2       1.00      1.00      1.00        63
           3       0.98      0.98      0.98        44
           4       0.98      0.95      0.96        55
           5       0.97      0.99      0.98        67
           6       1.00      1.00      1.00        44
           7       1.00      1.00      1.00        52
           8       1.00      1.00      1.00        47
           9       0.93      0.97      0.95        58

    accuracy                           0.99       539
   macro avg       0.99      0.99      0.99       539
weighted avg       0.99      0.99      0.99       539


DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.661  Test Acc : 0.673  Depth : 5
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.817  Test Acc : 0.87   Depth : 10
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.822  Test Acc : 0.881  Depth : 15
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.817  Test Acc : 0.878  Depth : 20
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.822  Test Acc : 0.865  Depth : 25
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.833  Test Acc : 0.872  Depth : 30
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.794  Test Acc : 0.878  Depth : 35
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.828  Test Acc : 0.878  Depth : 40
DTree # Split_Ratio: (0.6, 0.1)  Val Acc : 0.806  Test Acc : 0.861  Depth : 45

----------------------------------------------------------------
Best DecisionTree Depth: 45     train_frac:dev_frac:(0.6, 0.1)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      0.98      0.99        50
           1       1.00      1.00      1.00        59
           2       1.00      1.00      1.00        63
           3       0.98      0.98      0.98        44
           4       0.98      0.95      0.96        55
           5       0.97      0.99      0.98        67
           6       1.00      1.00      1.00        44
           7       1.00      1.00      1.00        52
           8       1.00      1.00      1.00        47
           9       0.93      0.97      0.95        58

    accuracy                           0.99       539
   macro avg       0.99      0.99      0.99       539
weighted avg       0.99      0.99      0.99       539



Processing step###  4  ##-------------------------------------------------------------------------------------------
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.969  Test Acc : 0.974  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.972  Test Acc : 0.978  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.986  Test Acc : 0.986  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.986  Test Acc : 0.989  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.969  Test Acc : 0.974  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.972  Test Acc : 0.978  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.986  Test Acc : 0.986  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.986  Test Acc : 0.989  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.4, 0.2)  Val Acc : 0.972  Test Acc : 0.958  Gamma :0.0005  C :0.2

------------------Best Results ----------------------------------
Best Parms SVM Gamma: 0.0005  C: 0.2  train_frac:dev_frac:(0.4, 0.2)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        73
           1       0.97      1.00      0.99        74
           2       1.00      1.00      1.00        65
           3       0.99      0.98      0.98        81
           4       1.00      1.00      1.00        66
           5       0.97      0.99      0.98        76
           6       1.00      1.00      1.00        72
           7       0.99      1.00      0.99        71
           8       0.99      0.97      0.98        69
           9       0.99      0.96      0.97        72

    accuracy                           0.99       719
   macro avg       0.99      0.99      0.99       719
weighted avg       0.99      0.99      0.99       719


DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.731  Test Acc : 0.751  Depth : 5
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.833  Test Acc : 0.8    Depth : 10
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.825  Test Acc : 0.801  Depth : 15
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.856  Test Acc : 0.825  Depth : 20
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.814  Test Acc : 0.797  Depth : 25
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.831  Test Acc : 0.803  Depth : 30
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.828  Test Acc : 0.791  Depth : 35
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.833  Test Acc : 0.808  Depth : 40
DTree # Split_Ratio: (0.4, 0.2)  Val Acc : 0.825  Test Acc : 0.811  Depth : 45

----------------------------------------------------------------
Best DecisionTree Depth: 45     train_frac:dev_frac:(0.4, 0.2)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        73
           1       0.97      1.00      0.99        74
           2       1.00      1.00      1.00        65
           3       0.99      0.98      0.98        81
           4       1.00      1.00      1.00        66
           5       0.97      0.99      0.98        76
           6       1.00      1.00      1.00        72
           7       0.99      1.00      0.99        71
           8       0.99      0.97      0.98        69
           9       0.99      0.96      0.97        72

    accuracy                           0.99       719
   macro avg       0.99      0.99      0.99       719
weighted avg       0.99      0.99      0.99       719



Processing step###  5  ##-------------------------------------------------------------------------------------------
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.958  Test Acc : 0.961  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.969  Test Acc : 0.974  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.978  Test Acc : 0.991  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.986  Test Acc : 0.993  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.958  Test Acc : 0.961  Gamma :0.001   C :0.2
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.969  Test Acc : 0.974  Gamma :0.001   C :0.3
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.978  Test Acc : 0.991  Gamma :0.001   C :0.7
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.986  Test Acc : 0.993  Gamma :0.001   C :5
SVM   # Split_Ratio: (0.5, 0.2)  Val Acc : 0.953  Test Acc : 0.957  Gamma :0.0005  C :0.2

------------------Best Results ----------------------------------
Best Parms SVM Gamma: 0.0005  C: 0.2  train_frac:dev_frac:(0.5, 0.2)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        52
           1       0.98      1.00      0.99        52
           2       1.00      1.00      1.00        53
           3       0.98      1.00      0.99        60
           4       1.00      1.00      1.00        61
           5       0.97      1.00      0.98        58
           6       1.00      1.00      1.00        50
           7       1.00      1.00      1.00        56
           8       1.00      0.98      0.99        54
           9       1.00      0.93      0.96        43

    accuracy                           0.99       539
   macro avg       0.99      0.99      0.99       539
weighted avg       0.99      0.99      0.99       539


DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.75   Test Acc : 0.727  Depth : 5
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.8    Test Acc : 0.788  Depth : 10
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.806  Test Acc : 0.768  Depth : 15
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.806  Test Acc : 0.781  Depth : 20
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.811  Test Acc : 0.785  Depth : 25
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.822  Test Acc : 0.777  Depth : 30
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.797  Test Acc : 0.772  Depth : 35
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.789  Test Acc : 0.781  Depth : 40
DTree # Split_Ratio: (0.5, 0.2)  Val Acc : 0.814  Test Acc : 0.783  Depth : 45

----------------------------------------------------------------
Best DecisionTree Depth: 45     train_frac:dev_frac:(0.5, 0.2)
------------------------- Test Report --------------------------

Classification report for classifier SVC(C=5, gamma=0.001):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        52
           1       0.98      1.00      0.99        52
           2       1.00      1.00      1.00        53
           3       0.98      1.00      0.99        60
           4       1.00      1.00      1.00        61
           5       0.97      1.00      0.98        58
           6       1.00      1.00      1.00        50
           7       1.00      1.00      1.00        56
           8       1.00      0.98      0.99        54
           9       1.00      0.93      0.96        43

    accuracy                           0.99       539
   macro avg       0.99      0.99      0.99       539
weighted avg       0.99      0.99      0.99       539



--------------------------------------------------------------------------------------
```
