from sklearn import tree
import pandas as pd
# from sklearn.metrics import confusion_matrix
import graphviz
import numpy as np
import os
from sklearn.model_selection import KFold

os.environ["PATH"] += os.pathsep + 'D:/ProgramFiles/Graphviz/release/bin'

file = open("wine.data", "r")
wineText = file.readlines()
wineData = pd.DataFrame(
    {"Class": [], "Alcohol": [], "Malic acid": [], "Ash": [], "Alcalinity of ash": [], "Magnesium": [],
     "Total phenols": [], "Flavanoids": [], "Nonflavanoid phenols": [], "Proanthocyanins": [],
     "Color intensity": [], "Hue": [], "OD280/OD315 of diluted wines": [], "Proline": []})
for wineTextLine in wineText:
    wVL = wineTextLine.split(",")  # array of values in wine text line [wine Values Line]
    wVL[-1] = wVL[-1].replace('\n', '')
    tempData = pd.Series(
        {"Class": wVL[0], "Alcohol": wVL[1], "Malic acid": wVL[2], "Ash": wVL[3], "Alcalinity of ash": wVL[4],
         "Magnesium": wVL[5],
         "Total phenols": wVL[6], "Flavanoids": wVL[7], "Nonflavanoid phenols": wVL[8], "Proanthocyanins": wVL[9],
         "Color intensity": wVL[10], "Hue": wVL[11], "OD280/OD315 of diluted wines": wVL[12], "Proline": wVL[13]})
    wineData = wineData.append(tempData, ignore_index=True)
wineData = wineData.astype('float')
wineData = wineData.astype({"Class": "int64", "Proline": "int64"})

### Create full decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(wineData.loc[:,wineData.columns != 'Class'].values, wineData['Class'].values)

# ### Draw full decision tree
# column_names = wineData.loc[:,wineData.columns != 'Class'].columns.values
# class_names = ["1","2","3"]
# graph_data = tree.export_graphviz(clf, out_file=None, feature_names=column_names, filled=True,
#                                   rounded=True, special_characters=True, class_names=class_names)
# graph = graphviz.Source(graph_data, format='png')
# graph.render("FullTree")


### Check accuracy of the full tree
cls_predict = pd.DataFrame()
cls_predict["Class"] = wineData.Class.values
cls_predict["Predicted"] = clf.predict(wineData.loc[:,wineData.columns != 'Class'].values)


def CM(data):
    cls_n = data['Class'].max()
    confusion_matrix = np.zeros([cls_n, cls_n])

    for i in range(cls_n):
        for j in range(cls_n):
            confusion_matrix[i, j] = sum(data.Predicted[data.Class == i + 1] == j + 1)

    return confusion_matrix


def derivationsCM(CM):
    tp = [0 for x in range(np.size(CM, 0))]
    fp = [0 for x in range(np.size(CM, 0))]
    fn = [0 for x in range(np.size(CM, 0))]
    tn = [0 for x in range(np.size(CM, 0))]
    acc = [0 for x in range(np.size(CM, 0))]
    tpr = [0 for x in range(np.size(CM, 0))]
    fpr = [0 for x in range(np.size(CM, 0))]

    for i in range(np.size(CM, 0)):
        for j in range(np.size(CM, 0)):
            if i == j:
                tp[i] = CM[i, j]
            fp[i] += CM[j,i]
            fn[i] += CM[i,j]
        fp[i] -= tp[i]
        fn[i] -= tp[i]
        tn[i] = np.sum(CM) - tp[i] - fp[i] - fn[i]
        acc[i] = (tp[i]+tn[i])/(tp[i]+tn[i]+fp[i]+fn[i])
        tpr[i] = tp[i]/(tp[i]+fn[i])
        fpr[i] = fp[i]/(fp[i]+tn[i])

    derivations = pd.DataFrame({"Class": [],"ACC": [],"TP": [],"TN": [],"TPR": [],"FPR": []})
    for i in range(np.size(CM, 0)):
        temp = pd.Series({"Class": i+1,"ACC": acc[i],"TP": tp[i],"TN": tn[i],"TPR": tpr[i],"FPR": fpr[i]})
        derivations = derivations.append(temp, ignore_index=True)

    derivations = derivations.astype({"Class": 'int64',"TP": 'int64',"TN": 'int64'})
    return derivations


confusion_matrix = CM(cls_predict)
derivations_CM = derivationsCM(confusion_matrix)


### Full decision tree with cross-validation
kf = KFold(n_splits=10,shuffle=True)
k_errors = [0,0,0,0,0,0,0,0,0,0]
d_CM = [0,0,0,0,0,0,0,0,0,0]
n = 0

for train_idx, test_idx in kf.split(wineData):
    wineData_Train = wineData.loc[train_idx, :]
    wineData_Test = wineData.loc[test_idx, :]
    # train data
    TRAIN = tree.DecisionTreeClassifier()
    TRAIN = TRAIN.fit(wineData_Train.loc[:,wineData_Train.columns != 'Class'].values, wineData_Train['Class'].values)
    # predict class
    wineDataPredict = pd.DataFrame()
    wineDataPredict["Class"] = wineData.Class.values
    wineDataPredict["Predicted"] = TRAIN.predict(wineData.loc[:, wineData.columns != 'Class'].values)
    k_errors[n] = sum(wineDataPredict.Class.values != wineDataPredict.Predicted.values)
    confusion_matrix = CM(wineDataPredict)
    d_CM[n] = derivationsCM(confusion_matrix)
    n += 1

print(sum(d_CM)/10)


# ### Minimal Cost-Complexity Pruning
# clf_min = tree.DecisionTreeClassifier()
# # clf_min = clf_min.decision_path(wineData.loc[:,wineData.columns != 'Class'].values)
# # clf_min = clf_min.cost_complexity_pruning_path(wineData.loc[:,wineData.columns != 'Class'].values, wineData['Class'].values)
# print(clf_min.fit(wineData.loc[:,wineData.columns != 'Class'].values, wineData['Class'].values).decision_path(wineData.loc[:,wineData.columns != 'Class'].values))
# # clf_min = clf_min.fit(wineData.loc[:,wineData.columns != 'Class'].values, wineData['Class'].values)



# ### Draw full decision tree
# column_names = wineData.loc[:,wineData.columns != 'Class'].columns.values
# class_names = ["1","2","3"]
# graph_data = tree.export_graphviz(clf_min, out_file=None, feature_names=column_names, filled=True,
#                                   rounded=True, special_characters=True, class_names=class_names)
# graph = graphviz.Source(graph_data, format='png')
# graph.render("MinTree")