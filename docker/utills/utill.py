import statistics
import pandas as pd

def mean_std(result):
    '''This function is created to to calculate Mean and std for each Model for test accuracy.'''
    print("\n--------------------------------------------------------------------------------------------------")
    result_mean_std={"svm_mean":[],"svm_std":[],"DecisionTree_mean":[],"DecisionTree_std":[]}
    for i,j in result.items():
        print(" {0:<17}  : {1}".format(i,j))
        if "svm" in i:
            result_mean_std["svm_mean"].append(round(statistics.mean(j),3))
            result_mean_std["svm_std"].append(round(statistics.stdev(j),3))
        else:
            result_mean_std["DecisionTree_mean"].append(round(statistics.mean(j),3))
            result_mean_std["DecisionTree_std"].append(round(statistics.stdev(j),3))
    print("\n")
    df = pd.DataFrame({"SVM_Model_mean":result_mean_std["svm_mean"],"SVM_Model_std":result_mean_std["svm_std"],"DecisionTree_mean":result_mean_std["DecisionTree_mean"],"DecisionTree_std":result_mean_std["DecisionTree_std"]})
    df.index.names=["Models"]
    df.index += 1
    print(df)
    print("--------------------------------------------------------------------------------------------------------------------------------------------\n")

def label_comp(lbl,y_test):
    "This function is created to compare test data ground truth values with the predicted labels from SVM and DecisionTree classifier  model"
    var3_ground_truth=pd.value_counts(y_test).sort_index()
    df=pd.DataFrame()
    print("------------------Comparing total wrong prediction classwise for each Model(split) case-------------------------------------------------------")
    for i in range(1,6):
        #print(sum(bl["label_S_"+str(i)]))
        var1_svm=pd.DataFrame(lbl["label_S_"+str(i)]).sum(axis = 0)
        var2_rand=pd.DataFrame(lbl["label_R_"+str(i)]).sum(axis = 0)
        df["WPre_SVM_M"+str(i)]=var1_svm.astype('Int64')
        df["WPre_DTr_M"+str(i)]=var2_rand.astype('Int64')
    df.index.names=["Class_label"]
    print("\n--------------------------------------------------------------------------------------------------------------------------------------------")
    print(df)
    print("\nTotal count of all wrong prediction with respect to each Models(each Split case.)--------")
    print(df.sum(axis = 0))
    print("--------------------------------------------------------------------------------------------------------------------------------------------\n")

