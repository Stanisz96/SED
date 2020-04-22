from sklearn import tree
import pandas as pd
# from sklearn.metrics import confusion_matrix
import graphviz
import numpy as np
import os
import matplotlib.pyplot as plt
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
kf = KFold(n_splits=5,shuffle=True)
train_derivations_CM = [0,0,0,0,0]
test_derivations_CM = [0,0,0,0,0]
n = 0

for train_idx, test_idx in kf.split(wineData):
    wineData_Train = wineData.loc[train_idx, :]
    wineData_Test = wineData.loc[test_idx, :]
    # train data
    TRAIN = tree.DecisionTreeClassifier()
    TRAIN = TRAIN.fit(wineData_Train.loc[:,wineData_Train.columns != 'Class'].values, wineData_Train['Class'].values)
    # predict class
    wineDataTrainPredict = pd.DataFrame()
    wineDataTestPredict = pd.DataFrame()
    wineDataTrainPredict["Class"] = wineData_Train.Class.values
    wineDataTestPredict["Class"] = wineData_Test.Class.values
    wineDataTrainPredict["Predicted"] = TRAIN.predict(wineData_Train.loc[:, wineData_Train.columns != 'Class'].values)
    wineDataTestPredict["Predicted"] = TRAIN.predict(wineData_Test.loc[:, wineData_Test.columns != 'Class'].values)
    train_confusion_matrix = CM(wineDataTrainPredict)
    test_confusion_matrix = CM(wineDataTestPredict)
    train_derivations_CM[n] = derivationsCM(train_confusion_matrix)
    test_derivations_CM[n] = derivationsCM(test_confusion_matrix)
    n += 1

train_mean_results = sum(train_derivations_CM)/5
test_mean_results = sum(test_derivations_CM)/5

print(train_mean_results)
print(test_mean_results)


### Minimal Cost-Complexity Pruning using k-fold cross-validation
train_scores = []
test_scores = []
ccp_alphas = []
impurities = []

for train_idx, test_idx in kf.split(wineData):
    wineData_Train = wineData.loc[train_idx, :]
    wineData_Test = wineData.loc[test_idx, :]
    clf_min = tree.DecisionTreeClassifier()
    path = clf_min.cost_complexity_pruning_path(wineData_Train.loc[:,wineData_Train.columns != 'Class'].values, wineData_Train['Class'].values)
    ccp_alphas.append(path.ccp_alphas)
    impurities.append(path.impurities)

    clfs_min = []
    for ccp_alpha in ccp_alphas[-1]:
        clf_min = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf_min.fit(wineData_Train.loc[:,wineData_Train.columns != 'Class'].values, wineData_Train['Class'].values)
        clfs_min.append(clf_min)

    # fig, ax = plt.subplots()
    # ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    # ax.set_xlabel("effective alpha")
    # ax.set_ylabel("total impurity of leaves")
    # ax.set_title("Total Impurity vs effective alpha for training set")
    # plt.show()

    train_scores.append([clf_min.score(wineData_Train.loc[:,wineData_Train.columns != 'Class'].values, wineData_Train['Class'].values) for clf_min in clfs_min])
    test_scores.append([clf_min.score(wineData_Test.loc[:,wineData_Test.columns != 'Class'].values, wineData_Test['Class'].values) for clf_min in clfs_min])

# MEANS
max_len = max([len(i) for i in ccp_alphas])
max_train_score_idxs = [np.where(i == np.amax(i)) for i in train_scores]
max_test_score_idxs = [np.where(i == np.amax(i)) for i in test_scores]
# print(max_test_score_idxs[0])
mean_ccp_alphas_train = [sum(ccp_alphas[i][max_train_score_idxs[i][0]])/len(max_train_score_idxs[i][0]) for i in range(5)]
mean_ccp_alphas_test = [sum(ccp_alphas[i][max_test_score_idxs[i][0]])/len(max_test_score_idxs[i][0]) for i in range(5)]
# print(sum(ccp_alphas[0][max_test_score_idxs[0][0]])/len(max_test_score_idxs[0][0]))
# print(max_len_ccp)
# print(mean_cpp_alphas)

k = [1,2,3,4,5]
fig, ax = plt.subplots()
ax.set_xlabel("K")
ax.set_ylabel("ccp_alpha")
ax.set_title("Mean ccp_alpha for k's test set with maximum accuracy")
ax.plot(k, mean_ccp_alphas_test, marker='o', label="test",
        drawstyle="steps-post")
# ax.legend()
plt.savefig("5fold_ccp_alphas.png",dpi=150)
plt.show()
mean_total_ccp_alpha = sum(mean_ccp_alphas_test)/len(mean_ccp_alphas_test)
print("Mean ccp_alpha: ",mean_total_ccp_alpha) # Mean ccp_alpha:  0.01071973699112266

# ### Draw  decision tree for ccp_alpha
# clf_ccp_alpha = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=mean_total_ccp_alpha)
# clf_ccp_alpha.fit(wineData.loc[:,wineData.columns != 'Class'].values, wineData['Class'].values)
# column_names = wineData.loc[:,wineData.columns != 'Class'].columns.values
# class_names = ["1","2","3"]
# graph_data = tree.export_graphviz(clf_ccp_alpha, out_file=None, feature_names=column_names, filled=True,
#                                   rounded=True, special_characters=True, class_names=class_names)
# graph = graphviz.Source(graph_data, format='png')
# graph.render("CppAlphaTree")