from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np


iris = load_iris(return_X_y=True)
features_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
class_names = ['setosa', 'versicolor', 'virginica']
irisData = pd.DataFrame({'Class': iris[1], features_names[0]: iris[0][:,0], features_names[1]: iris[0][:,1],
                         features_names[2]: iris[0][:,2], features_names[3]: iris[0][:,3]})

PU_idx = np.random.rand(len(irisData)) < 0.8
irisData_train = irisData[PU_idx]
irisData_test = irisData[~PU_idx]

clf = BaggingClassifier(n_estimators=50)
clf = clf.fit(irisData_train.loc[:,irisData_train.columns != 'Class'].values, irisData_train['Class'].values)
predicted_Data = clf.predict(irisData_test.loc[:,irisData_test.columns != 'Class'].values)
print(predicted_Data)
print(irisData_test['Class'].values)

