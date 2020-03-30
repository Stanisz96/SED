import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB


#### Exercise 1: "Confusion matrix" ####
def generateGaussData(S, m, n, cls_n):
    # S -> array of matrix covariances for n classes
    # m -> array of matrix means for n classes
    # n -> array of number of points for every class
    # cls_n -> number of classes

    if cls_n == 1:
        if (np.size(S, 0) != np.size(m, 0)) | (np.size([n], 0) != cls_n):
            return print("Wrong size of data1!")
        S = [S]
        m = [m]
        n = [n]
    elif np.size(S, 0) != cls_n | np.size(m, 0) != cls_n | (np.size(S, 1) != np.size(m, 1)) | (
            np.size(S, 2) != np.size(m, 1)):
        return print("Wrong size of data2!")

    S = np.array(S)
    m = np.array(m)
    n = np.array(n)
    X = [0 for x in range(cls_n)]
    dataFrame = pd.DataFrame()
    for i in range(cls_n):
        X[i] = np.random.multivariate_normal(m[i], S[i], n[i])
        x, y = zip(*X[i])
        temp = pd.DataFrame({"Class": [i + 1 for x in range(np.size(X[i], 0))], "x": x, "y": y})
        dataFrame = dataFrame.append(temp, ignore_index=True)

    return dataFrame


# Set properties for normal distribution points
S1 = np.array([[4, 2], [2, 4]])
S2 = np.array([[4, 2], [2, 2]])
S3 = np.array([[2, 1], [1, 2]])
m1 = np.array([-1, -1])
m2 = np.array([2, 2])
m3 = np.array([4, -1])
m = [m1, m2]
S = [S1, S2]

# Generate data structure with classes
data = generateGaussData(S, m, [80, 60], 2)
## 3 classes
data3 = generateGaussData([S1,S2,S3],[m1,m2,m3],[80,60,70],3)

# Train classificators on Training Dataset (TD)
def train(data):
    qda = QDA()
    lda = LDA()
    gnb = GaussianNB()

    cls_pred_qda = qda.fit(data[['x', 'y']].values, data['Class'].values)  # .predict(data[['x', 'y']].values)
    cls_pred_lda = lda.fit(data[['x', 'y']].values, data['Class'].values)  # .predict(data[['x', 'y']].values)
    cls_pred_gnb = gnb.partial_fit(data[['x', 'y']].values, data['Class'].values,
                                   np.unique(data['Class'].values))  # .predict(data[['x', 'y']].values)

    TD = [cls_pred_qda, cls_pred_lda, cls_pred_gnb]

    return TD


## Use methods qda,lda, nb on dataset
TD = train(data)
## 3 classes
TD3 = train(data3)

## Add predicted classes to data
data["qda_cls"] = TD[0].predict(data[['x', 'y']].values)
data["lda_cls"] = TD[1].predict(data[['x', 'y']].values)
data["gnb_cls"] = TD[2].predict(data[['x', 'y']].values)
## 3 classes
data3["qda_cls"] = TD3[0].predict(data3[['x', 'y']].values)
data3["lda_cls"] = TD3[1].predict(data3[['x', 'y']].values)
data3["gnb_cls"] = TD3[2].predict(data3[['x', 'y']].values)

# Function to create confusion matrix

def CM(data):
    cls_n = data['Class'].max()
    CM_qda = np.zeros([cls_n, cls_n])
    CM_lda = np.zeros([cls_n, cls_n])
    CM_gnb = np.zeros([cls_n, cls_n])

    for i in range(cls_n):
        for j in range(cls_n):
            CM_qda[i, j] = sum(data.qda_cls[data.Class == i + 1] == j + 1)
            CM_lda[i, j] = sum(data.lda_cls[data.Class == i + 1] == j + 1)
            CM_gnb[i, j] = sum(data.gnb_cls[data.Class == i + 1] == j + 1)

    return CM_qda, CM_lda, CM_gnb


data_CM = CM(data)
## 3 classes
data3_CM = CM(data3)

# Derivations CM (ACC, TP, TN, TPR, FPR)

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


# Create new data testing set (TS)

dataTS = generateGaussData(S, m, [30, 30], 2)
## 3 classes
data3TS = generateGaussData([S1,S2,S3],[m1,m2,m3],[30,30,30],3)

## Add predicted classes to data
dataTS["qda_cls"] = TD[0].predict(dataTS[['x', 'y']].values)
dataTS["lda_cls"] = TD[1].predict(dataTS[['x', 'y']].values)
dataTS["gnb_cls"] = TD[2].predict(dataTS[['x', 'y']].values)
## 3 classes
data3TS["qda_cls"] = TD3[0].predict(data3TS[['x', 'y']].values)
data3TS["lda_cls"] = TD3[1].predict(data3TS[['x', 'y']].values)
data3TS["gnb_cls"] = TD3[2].predict(data3TS[['x', 'y']].values)

# CM
dataTS_CM = CM(dataTS)
## 3 classes
data3TS_CM = CM(data3TS)

print(getCMResultsForCls(data3_CM))
print("-------------------------------------------")
print(getCMResultsForCls(data3TS_CM))