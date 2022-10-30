# [Digits classification]

Reff # https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py  
Reff# https://scikit-learn.org/stable/modules/tree.html#classification

## [Model Comparison]
### Accuracy for SVM and Decision Tree Classifier for each Model(each Split case)

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
![image](https://user-images.githubusercontent.com/89742374/198873873-eefa8490-7d23-4752-a824-44c93a296ab5.png)

### Computing the mean and standard deviations of both the classifier's performances.

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

![image](https://user-images.githubusercontent.com/89742374/198873883-97e261e6-972b-47e7-84f3-39f5d9cef40c.png)


***Best Model : SVM***  
