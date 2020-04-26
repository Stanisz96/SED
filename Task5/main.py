from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np


iris = load_iris(return_X_y=True)
features_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
class_names = ['setosa', 'versicolor', 'virginica']
irisData = pd.DataFrame({'Class': iris[1], features_names[0]: iris[0][:,0], features_names[1]: iris[0][:,1],
                         features_names[2]: iris[0][:,2], features_names[3]: iris[0][:,3]})

# PU_idx = np.random.rand(len(irisData)) < 0.8
# irisData_train = irisData[PU_idx]
# irisData_test = irisData[~PU_idx]

def get_clf_predict(estimator, n_estimators):
    clf = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators, max_samples=0.8)
    clf = clf.fit(irisData.loc[:,irisData.columns != 'Class'].values, irisData['Class'].values)
    predicted_Data = clf.predict(irisData.loc[:,irisData.columns != 'Class'].values)
    temp = {'Class': irisData.Class.values, 'Predicted': predicted_Data}
    clfData = pd.DataFrame(data=temp)
    clfData = clfData.set_index([pd.Index(irisData.index.values)])
    return clfData


err_LDA_table = pd.DataFrame()
err_Tree_table = pd.DataFrame()

for i in range(1,51):
    tempData_LDA = get_clf_predict(LDA(), i)
    tempData_Tree = get_clf_predict(None, i)
    errLDA = sum(tempData_LDA.Class.values != tempData_LDA.Predicted.values) / len(tempData_LDA.Class.values) * 100
    errTree = sum(tempData_Tree.Class.values != tempData_Tree.Predicted.values)/len(tempData_Tree.Class.values) * 100
    err_LDA_table = err_LDA_table.append({'Number of estimators': i, 'Error [%]': round(errLDA, 2)},ignore_index=True)
    err_Tree_table = err_Tree_table.append({'Number of estimators': i, 'Error [%]': round(errTree, 2)},ignore_index=True)

print(err_LDA_table)
print(err_Tree_table)

# fig, ax = plt.subplots()
# fig.set_figheight(6)
# fig.set_figwidth(8)
# ax.plot(err_LDA_table['Number of estimators'].values, err_LDA_table['Error [%]'].values, marker='o', drawstyle="steps-post")
# ax.set_xlabel("Number of estimators")
# ax.set_ylabel("Error of wrong predicted [%]")
# ax.set_title("Dependence of error on number of estimators for LDA")
# plt.savefig("LDA.png",dpi=150)
# plt.close()

# fig, ax = plt.subplots()
# fig.set_figheight(6)
# fig.set_figwidth(8)
# ax.plot(err_Tree_table['Number of estimators'].values, err_Tree_table['Error [%]'].values, marker='o', drawstyle="steps-post")
# ax.set_xlabel("Number of estimators")
# ax.set_ylabel("Error of wrong predicted [%]")
# ax.set_title("Dependence of error on number of estimators for Decision Tree")
# plt.savefig("Tree.png",dpi=150)
# plt.close()