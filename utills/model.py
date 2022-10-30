
#importing python modules
from sklearn import datasets, svm,metrics
from sklearn.tree import DecisionTreeClassifier

# define hyper-parameter combinataion for turning.
params = {"gamma":[0.001,0.001,0.0005],"c_value":[0.2,0.3,0.7,5]}
depth=[5,10,15,20,25,30,35,40,45]


def svm_model(index):
    '''This function is created to perform SVM clasification task with diff diff hyper-parameters'''
    h_param_comb = [{"gamma": g, "C": c} for g in params['gamma'] for c in params['c_value']]
    clf=svm.SVC(gamma=h_param_comb[index]["gamma"],C=h_param_comb[index]["C"])
    return clf,h_param_comb[index]

def DecisionTree_model(index):
    '''This function is created to perform DecisionTree clasification task with diff depth hyper-parameters'''
    decision_tree = DecisionTreeClassifier(max_depth=depth[index])
    return decision_tree,depth[index]

