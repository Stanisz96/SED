import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.svm import SVC

# Properties for generated data
m1, S1, n1 = np.array([-1, 1]), np.array([[1, 0], [0, 1]]), 30
m2, S2, n2 = np.array([2, 4]), np.array([[1, 0], [0, 1]]), 30
m3, S3, n3 = np.array([-2, 2]), np.array([[1, 0], [0, 1]]), 30


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


gaussData = generateGaussData([S1,S2,S3], [m1,m2,m3],[n1,n2,n3],3)
print(gaussData)
# Divide data to train and test
# PU_idx = np.random.rand(len(gaussData)) < 0.8
# gaussData_train = gaussData[PU_idx]
# gaussData_test = gaussData[~PU_idx]


# LDA
clf_LDA = LDA()
clf_LDA = clf_LDA.fit(gaussData.loc[:,gaussData.columns != 'Class'].values, gaussData['Class'].values)
testData_LDA = pd.DataFrame()
testData_LDA["Class"] = gaussData.Class.values
testData_LDA["lda_cls"] = clf_LDA.predict(gaussData.loc[:,gaussData.columns != 'Class'].values)

# SVC
clf_SVC = SVC(C=1.0, kernel="linear")
clf_SVC = clf_SVC.fit(gaussData.loc[:,gaussData.columns != 'Class'].values, gaussData['Class'].values)
testData_SVC = pd.DataFrame()
testData_SVC["Class"] = gaussData.Class.values
testData_SVC["svc_cls"] = clf_SVC.predict(gaussData.loc[:,gaussData.columns != 'Class'].values)


def plot_function_N(da, X, cls, cls_pred, cls_n):
    cls_col = ["#EC7063","#A569BD","#5DADE2","#27AE60","#F1C40F","#E67E22"]
    cls_col_dark = ["#922B21","#76448A","#2471A3","#1E8449","#B7950B","#AF601A"]
    cls_col_map = ["#f4aca4","#d5b8e0","#a8d3f0","#acecc7","#f9e79f","f2bc8c"]
    cmap = colors.ListedColormap([cls_col_map[x] for x in range(cls_n)])
    plt.figure(figsize=(8, 6))
    tp = (cls == cls_pred)
    tpC = [0 for x in range(cls_n)]
    XC = [0 for x in range(cls_n)]
    XC_tp = [0 for x in range(cls_n)]
    XC_fp = [0 for x in range(cls_n)]
    for i in range(cls_n):
        tpC[i] = tp[cls == i+1]
        XC[i] = X[cls == i+1]
        XC_tp[i] = XC[i][tpC[i]]
        XC_fp[i] = XC[i][~tpC[i]]

        # class N: points
        plt.scatter(XC_tp[i][:, 0], XC_tp[i][:, 1], marker='.',s=30, color=cls_col[i])
        plt.scatter(XC_fp[i][:, 0], XC_fp[i][:, 1], marker='x',s=20, color=cls_col_dark[i])

    # class 1 & 2: areas
    nx, ny = 150, 150
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z_cls = da.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_cls = Z_cls.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z_cls, cmap=cmap, zorder=0)
    plt.contour(xx, yy, Z_cls, linewidths=1., colors='white', zorder=0)


plot_function_N(clf_LDA, gaussData[['x','y']].values,gaussData["Class"].values,testData_LDA["lda_cls"].values,3)
# plt.legend(["True positive for class 1","False negative for class 1","True positive for class 2","False negative for class 2"],loc="lower right")
plt.title("LDA")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("img/lda.png",dpi=150)
plt.show()

plot_function_N(clf_SVC, gaussData[['x','y']].values,gaussData["Class"].values,testData_SVC["svc_cls"].values,3)
# plt.legend(["True positive for class 1","False negative for class 1","True positive for class 2","False negative for class 2"],loc="lower right")
plt.title("SVC with linear kernel")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("img/svc.png",dpi=150)
plt.show()