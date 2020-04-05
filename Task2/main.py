import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

file = open("wine.data", "r")
wineText = file.readlines()
wineData = pd.DataFrame(
    {"Class": [], "Alcohol": [], "Malic acid": [], "Ash": [], "Alcalinity of ash": [], "Magnesium": [],
     "Total phenols": [], "Flavanoids": [], "Nonflavanoid phenols": [], "Proanthocyanins": [],
     "Color intensity": [], "Hue": [], "OD280/OD315 of diluted wines": [], "Proline": []})
for wineTextLine in wineText:
    wVL = wineTextLine.split(",")  # array of values in wine text line
    wVL[-1] = wVL[-1].replace('\n', '')
    tempData = pd.Series(
        {"Class": wVL[0], "Alcohol": wVL[1], "Malic acid": wVL[2], "Ash": wVL[3], "Alcalinity of ash": wVL[4],
         "Magnesium": wVL[5],
         "Total phenols": wVL[6], "Flavanoids": wVL[7], "Nonflavanoid phenols": wVL[8], "Proanthocyanins": wVL[9],
         "Color intensity": wVL[10], "Hue": wVL[11], "OD280/OD315 of diluted wines": wVL[12], "Proline": wVL[13]})
    wineData = wineData.append(tempData, ignore_index=True)
wineData = wineData.astype('float')
wineData = wineData.astype({"Class": "int64", "Proline": "int64"})

##### Determination of confusion matrix
# Train classificators on Training Dataset (TD)
def train(data):
    qda = QDA()
    lda = LDA()
    gnb = GaussianNB()

    lda_fit = lda.fit(data.loc[:,data.columns != 'Class'].values, data['Class'].values)
    qda_fit = qda.fit(data.loc[:,data.columns != 'Class'].values, data['Class'].values)
    gnb_fit = gnb.partial_fit(data.loc[:,data.columns != 'Class'].values, data['Class'].values,
                              np.unique(data['Class'].values)) 

    ObjectData = [lda_fit, qda_fit, gnb_fit]

    return ObjectData


## Use methods lda,qda, nb on dataset
TD = train(wineData)
# print(TD)


# ## Add predicted classes to data
wineDataPredict = pd.DataFrame()
wineDataPredict["Class"] = wineData.Class.values
wineDataPredict["lda_cls"] = TD[0].predict(wineData.loc[:,wineData.columns != 'Class'].values)
wineDataPredict["qda_cls"] = TD[1].predict(wineData.loc[:,wineData.columns != 'Class'].values)
wineDataPredict["gnb_cls"] = TD[2].predict(wineData.loc[:,wineData.columns != 'Class'].values)
# print(wineDataPredict)


# Confusion matrix
def CM(data):
    cls_n = data['Class'].max()
    CM_lda = np.zeros([cls_n, cls_n])
    CM_qda = np.zeros([cls_n, cls_n])
    CM_gnb = np.zeros([cls_n, cls_n])

    for i in range(cls_n):
        for j in range(cls_n):
            CM_lda[i, j] = sum(data.lda_cls[data.Class == i + 1] == j + 1)
            CM_qda[i, j] = sum(data.qda_cls[data.Class == i + 1] == j + 1)
            CM_gnb[i, j] = sum(data.gnb_cls[data.Class == i + 1] == j + 1)

    return CM_lda, CM_qda, CM_gnb

wineDataCM = CM(wineDataPredict)


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

# for i in range(np.size(wineDataCM,0)):
#     print(derivationsCM(wineDataCM[i]))


def getCMResultsForCls(data_CM):
    result_CM = [pd.DataFrame({"Class": [],"ACC": [],"TP": [],"TN": [],"TPR": [],"FPR": []}) for x in range(np.size(data_CM,1))]

    for i in range(np.size(data_CM,1)):
        for x in range(np.size(data_CM,0)):
            temp = derivationsCM(data_CM[x])
            temp = temp.loc[temp.Class == i+1]
            result_CM[i] = result_CM[i].append(temp, ignore_index=True)
        result_CM[i] = result_CM[i].rename(index={0:'QDA',1:'LDA',2:'GNB'})
        result_CM[i] = result_CM[i].astype({"TP": 'int64',"TN": 'int64'})
        result_CM[i] = result_CM[i].drop(['Class'],axis=1)
    return result_CM

# print(getCMResultsForCls(wineDataCM))

### Limited to the 2, 5 and 10 first components and check accuracy of classifiers
# Create DataFrame with 2, 5, 10 first attributes
wineData_N = [wineData.loc[:,wineData.columns[:3]],wineData.loc[:,wineData.columns[:6]],wineData.loc[:,wineData.columns[:11]]]


# Use methods lda,qda, nb on dataset
TD_N = [train(wineData_N[0]),train(wineData_N[1]),train(wineData_N[2])] # 2,5,10


# print(TD_N[1][1])

# ## Add predicted classes to data
wineDataPredict_N = [pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
for i in range(3):
    wineDataPredict_N[i]["Class"] = wineData.Class.values
    wineDataPredict_N[i]["lda_cls"] = TD_N[0][i].predict(wineData_N[0].loc[:, wineData_N[0].columns != 'Class'].values)
    wineDataPredict_N[i]["qda_cls"] = TD_N[1][i].predict(wineData_N[1].loc[:, wineData_N[1].columns != 'Class'].values)
    wineDataPredict_N[i]["gnb_cls"] = TD_N[2][i].predict(wineData_N[2].loc[:, wineData_N[2].columns != 'Class'].values)

# Confusion matrices
wineDataCM_N = [CM(wineDataPredict_N[0]),CM(wineDataPredict_N[1]),CM(wineDataPredict_N[2])]

# print("Limited to the 2, 5 and 10 first components and check accuracy of classifiers")
# print(getCMResultsForCls(wineDataCM_N[0]))
# print(getCMResultsForCls(wineDataCM_N[1]))
# print(getCMResultsForCls(wineDataCM_N[2]))



### Limited to the 2 first variables, divide data set to (TRD, VD, TD) 50/25/25 and in this way choose from  _LDA, QDA_ and _NB_
wineData_TRD = pd.DataFrame()
wineData_VD = pd.DataFrame()
wineData_TD = pd.DataFrame()

# print(wineData.loc[wineData.Class == 1, wineData.columns[:3]].index)
indexCls = np.array([wineData.loc[wineData.Class == 1, wineData.columns[:3]].index.values, wineData.loc[wineData.Class == 2, wineData.columns[:3]].index.values, wineData.loc[wineData.Class == 3, wineData.columns[:3]].index.values])
indexClsRandom = [0,0,0]

for i in range(3):
    indexClsRandom[i] = np.random.permutation(indexCls[i])
    wineData_TRD = wineData_TRD.append(wineData.loc[indexClsRandom[i], wineData.columns[:3]][:int(len(wineData.loc[wineData.Class == i+1]) * 0.5)], ignore_index=True)
    wineData_VD = wineData_VD.append(wineData.loc[indexClsRandom[i], wineData.columns[:3]][int(len(wineData.loc[wineData.Class == i+1]) * 0.5):int(len(wineData.loc[wineData.Class == i+1]) * (0.75))], ignore_index=True)
    wineData_TD = wineData_TD.append(wineData.loc[indexClsRandom[i], wineData.columns[:3]][int(len(wineData.loc[wineData.Class == i+1]) * 0.75):], ignore_index=True)


# train TRD
TRD = train(wineData_TRD)

#predict class for VD
wineDataPredict_VD = pd.DataFrame()
wineDataPredict_VD["Class"] = wineData_VD.Class.values
wineDataPredict_VD["lda_cls"] = TRD[0].predict(wineData_VD.loc[:, wineData_VD.columns != 'Class'].values)
wineDataPredict_VD["qda_cls"] = TRD[1].predict(wineData_VD.loc[:, wineData_VD.columns != 'Class'].values)
wineDataPredict_VD["gnb_cls"] = TRD[2].predict(wineData_VD.loc[:, wineData_VD.columns != 'Class'].values)

# # print(wineDataPredict_VD)
# print(len(wineData_VD.Class))
# print("LDA: ",round(1-sum(wineData_VD.Class.values == wineDataPredict_VD.lda_cls.values)/len(wineData_VD.Class),2))
# print("QDA: ",round(1-sum(wineData_VD.Class.values == wineDataPredict_VD.qda_cls.values)/len(wineData_VD.Class),2))
# print("NB: ",round(1-sum(wineData_VD.Class.values == wineDataPredict_VD.gnb_cls.values)/len(wineData_VD.Class),2))

wineData_VD_CM = CM(wineDataPredict_VD)
# print(getCMResultsForCls(wineData_VD_CM))



# cross validation k=5
kf = KFold(n_splits=5,shuffle=True)

error_prob = np.array([0,0,0])

for train_idx, test_idx in kf.split(wineData_N[0]):
    wineData_Train = wineData.loc[train_idx, wineData.columns[:3]]
    wineData_Test = wineData.loc[test_idx, wineData.columns[:3]]
    # train data
    TRAIN = train(wineData_Train)
    # predict class
    wineDataPredict = pd.DataFrame()
    wineDataPredict["Class"] = wineData_Test.Class.values
    wineDataPredict["lda_cls"] = TRAIN[0].predict(wineData_Test.loc[:, wineData_Test.columns != 'Class'].values)
    wineDataPredict["qda_cls"] = TRAIN[1].predict(wineData_Test.loc[:, wineData_Test.columns != 'Class'].values)
    wineDataPredict["gnb_cls"] = TRAIN[2].predict(wineData_Test.loc[:, wineData_Test.columns != 'Class'].values)
    k_errors = [sum(wineData_Test.Class.values != wineDataPredict.lda_cls.values),sum(wineData_Test.Class.values != wineDataPredict.qda_cls.values),sum(wineData_Test.Class.values != wineDataPredict.gnb_cls.values)]
    error_prob = np.sum([error_prob,k_errors], axis=0)

error_prob = error_prob/len(wineData.Class)

print("LDA: ",round(error_prob[0],2)*100,"%")
print("QDA: ",round(error_prob[1],2)*100,"%")
print("NB: ",round(error_prob[2],2)*100,"%")