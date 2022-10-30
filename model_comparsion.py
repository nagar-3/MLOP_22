# Standard scientific Python imports
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utills.utill import mean_std
from utills.model import svm_model,DecisionTree_model,depth
from utills.utill import label_comp
import pandas as pd

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#define 5 different splits of train/test/valid.
train_frac_split=[0.8,0.7,0.6,0.4,0.5]
dev_frac_split=[0.1,0.1,0.1,0.2,0.2]

#Available models
model_choice=["svm_model","DecisionTree_model"]
best_DecisionTree_parm,best_svm_parm=0,dict()


# Split data into  train, test and dev subsets
def train_dev_test_split(data, label, train_frac, dev_frac):
    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(data, label, test_size=dev_test_frac, shuffle=True)
    x_test, x_dev, y_test, y_dev = train_test_split(x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True)
    return x_train, y_train, x_dev, y_dev, x_test,y_test


def print_result(model,best_split):
    """This function is created to print all best results from each iteration with respect to each split case."""
    if ("svm_model"== model) or ("svm_dtr"==model):
        print("\n------------------Best Results ----------------------------------")
        print("Best Parms SVM Gamma: {0}  C: {1}  train_frac:dev_frac:{2}".format(best_svm_parm["gamma"],best_svm_parm["C"],best_split))
        print("------------------------- Test Report --------------------------\n")
        print(Report_test)
    if ("DecisionTree_model"== model) or ("svm_dtr"==model):
        print("\n----------------------------------------------------------------")
        print("Best DecisionTree Depth: {0:<4}   train_frac:dev_frac:{1}".format(best_DecisionTree_parm,best_split))
        print("------------------------- Test Report --------------------------\n")
        print(Report_test)

print("\n\n Model Comparsion in progress..................\n")


result,lbl=dict(),dict()
for i in range(5):
  best_acc=0
  print("\nProcessing step### ",i+1," ##-------------------------------------------------------------------------------------------")
  result["svm_acc_model#"+str(i+1)],result["DTr_acc_model#"+str(i+1)]=[],[]
  lbl["label_S_"+str(i+1)],lbl["label_R_"+str(i+1)]=[],[]
  spli_ratio=(train_frac_split[i],dev_frac_split[i])
  X_train, y_train, X_dev, y_dev, X_test,y_test=train_dev_test_split(data,digits.target,train_frac_split[i],dev_frac_split[i])
  for j in model_choice:
    for k in range(len(depth)):
      clf,parms=svm_model(k) if j=="svm_model" else DecisionTree_model(k)
      clf.fit(X_train, y_train)
      predict_dev = clf.predict(X_dev)
      predict_test = clf.predict(X_test)
      acc_dev =round(accuracy_score(y_dev, predict_dev),3)
      acc_test =round(accuracy_score(y_test, predict_test),3)
      ground_truth=pd.value_counts(y_test).sort_index()
      pre_count=pd.value_counts(predict_test).sort_index()
      diff=abs(ground_truth-pre_count)
      if j=="svm_model":
        print("SVM   # Split_Ratio: {0}  Val Acc : {3: <5}  Test Acc : {4:<5}  Gamma :{1: <6}  C :{2}  ".format(spli_ratio,parms["gamma"],parms["C"],acc_dev,acc_test))
        best_svm_parm=parms
        result["svm_acc_model#"+str(i+1)].append(acc_test)
        lbl["label_S_"+str(i+1)].append(diff)
      else:
        print("DTree # Split_Ratio: {0}  Val Acc : {2: <5}  Test Acc : {3:<5}  Depth : {1: <6}   ".format(spli_ratio,parms,acc_dev,acc_test))
        best_DecisionTree_parm=parms
        result["DTr_acc_model#"+str(i+1)].append(acc_test)
        lbl["label_R_"+str(i+1)].append(diff)
      
      #comapring accuracy for each occurance to get the best model. 
      if acc_test>best_acc:
          best_acc=acc_test
          best_split=spli_ratio
          Report_test=f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test, predict_test)}\n"
    print_result(j,best_split)



if __name__ == "__main__":
    #initializing mean_std label_comp print_result functions.
    mean_std(result)
    label_comp(lbl,y_test)
    print_result('svm_dtr',best_split)
    print("--------------------------------------- END --------------------------------\n\n")

