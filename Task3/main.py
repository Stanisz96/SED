import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



## Solution
### Generate Data

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
m1 = np.array([-1, -1])
m2 = np.array([2, 2])
n1 = 30
n2 = 20
n = n1 + n2
m = [m1, m2]
S = [S1, S2]

# Generate data structure with classes
data = generateGaussData(S, m, [n1, n2], 2)

# KNN & CM
cls_predict = [0 for x in range(21)]
CM = [0 for x in range(21)]

for i in range(21):
    kNN = KNeighborsClassifier(n_neighbors=i+1)
    cls_predict[i] = kNN.fit(data[["x","y"]].values,data["Class"].values).predict(data[["x","y"]].values)
    CM[i] = confusion_matrix(data["Class"].values,cls_predict[i]).ravel() # [tn, fp, fn, tp]

print(CM[0][0])
idx_kNN = np.linspace(1,21,21,dtype='int64')
acc_kNN = [0 for x in range(21)]
tp_kNN = [0 for x in range(21)]
tn_kNN = [0 for x in range(21)]
for i in range(21):
    acc_kNN[i] = (CM[i][3]+CM[i][0])/(CM[i][0]+CM[i][1]+CM[i][2]+CM[i][3])
    tp_kNN[i] = CM[i][3]
    tn_kNN[i] = CM[i][0]

# Plot for acc
plt.figure(figsize=(8, 6))
plt.xticks(idx_kNN,idx_kNN)
plt.plot(idx_kNN,acc_kNN)
plt.ylabel("ACC")
plt.title("Accuracy depend on $N[k_{NN}]$")
plt.xlabel(r'$N[k_{NN}]$')
plt.grid(True)
plt.savefig("acc.png",dpi=150)
plt.close()
# Plot for TP and TN
plt.figure(figsize=(8, 6))
plt.xticks(idx_kNN,idx_kNN)
plt.plot(idx_kNN,tp_kNN, label="TP")
plt.plot(idx_kNN,tn_kNN, label="TN")
plt.title("TP and TN depend on $N[k_{NN}]$")
plt.ylabel("count")
plt.xlabel(r'$N[k_{NN}]$')
plt.legend(loc="center right")
plt.grid(True)
plt.savefig("tp_tn.png",dpi=150)
plt.close()

# TEst data
data_test = generateGaussData(S, m, [10, 5], 2)
data = data.append(data_test,ignore_index=True)
# KNN & CM
cls_predict_test = [0 for x in range(21)]
CM_test = [0 for x in range(21)]

for i in range(21):
    kNN = KNeighborsClassifier(n_neighbors=i+1)
    cls_predict_test[i] = kNN.fit(data[["x","y"]].values,data["Class"].values).predict(data[["x","y"]].values)
    CM_test[i] = confusion_matrix(data["Class"].values,cls_predict_test[i]).ravel() # [tn, fp, fn, tp]


idx_kNN = np.linspace(1,21,21,dtype='int64')
acc_kNN = [0 for x in range(21)]
tp_kNN = [0 for x in range(21)]
tn_kNN = [0 for x in range(21)]
for i in range(21):
    acc_kNN[i] = (CM_test[i][3]+CM_test[i][0])/(CM_test[i][0]+CM_test[i][1]+CM_test[i][2]+CM_test[i][3])
    tp_kNN[i] = CM_test[i][3]
    tn_kNN[i] = CM_test[i][0]

# Plot for acc test
plt.figure(figsize=(8, 6))
plt.xticks(idx_kNN,idx_kNN)
plt.plot(idx_kNN,acc_kNN)
plt.ylabel("ACC")
plt.title("Accuracy for extended data depend on $N[k_{NN}]$")
plt.xlabel(r'$N[k_{NN}]$')
plt.grid(True)
plt.savefig("ext_acc.png",dpi=150)
plt.close()
# Plot for TP and TN test
plt.figure(figsize=(8, 6))
plt.xticks(idx_kNN,idx_kNN)
plt.plot(idx_kNN,tp_kNN, label="TP")
plt.plot(idx_kNN,tn_kNN, label="TN")
plt.title("TP and TN for extended data depend on $N[k_{NN}]$")
plt.ylabel("count")
plt.xlabel(r'$N[k_{NN}]$')
plt.legend(loc="center right")
plt.grid(True)
plt.savefig("ext_tp_tn.png",dpi=150)
plt.close()