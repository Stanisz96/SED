import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib import colors


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
S1 = np.array([[4, 0], [0, 4]])
S2 = np.array([[4, 0], [0, 4]])
m1 = np.array([-3, -1])
m2 = np.array([2, 2])
n1 = 40
n2 = 30
n = n1 + n2
m = [m1, m2]
S = [S1, S2]

# Generate data structure with classes
data = generateGaussData(S, m, [n1, n2], 2)
print("1. :)")
print(data)


def naiveBayes(trainingSet,testSet):

    predictedClass = pd.DataFrame({"Class": [], "x": [], "y": []})
    pi = [trainingSet.Class.loc[trainingSet.Class == 1].count() / trainingSet.Class.count(),
          trainingSet.Class.loc[trainingSet.Class == 2].count() / trainingSet.Class.count()]
    varX = [trainingSet.loc[trainingSet.Class == 1, 'x'].var(), trainingSet.loc[trainingSet.Class == 2, 'x'].var()]
    varY =[trainingSet.loc[trainingSet.Class == 1, 'y'].var(), trainingSet.loc[trainingSet.Class == 2, 'y'].var()]
    mX = [trainingSet.loc[trainingSet.Class == 1, 'x'].mean(),trainingSet.loc[trainingSet.Class == 2, 'x'].mean()]
    mY = [trainingSet.loc[trainingSet.Class == 1, 'y'].mean(),trainingSet.loc[trainingSet.Class == 2, 'y'].mean()]
    px, py = [0,0], [0,0]
    cls = np.zeros((2,testSet.x.count()))
    x,y = testSet.x.values, testSet.y.values
    for k in range(trainingSet.Class.max(axis=0)):
        px[k] = (1 / np.sqrt(2 * np.pi * varX[k])) * np.exp(-(x-mX[k])**2/(2*varX[k]))
        py[k] = (1 / np.sqrt(2 * np.pi * varY[k])) * np.exp(-(y-mY[k])**2/(2*varY[k]))
        cls[k] = pi[k] * px[k] * py[k]

    temp_cls = [np.argmax([cls[0,n],cls[1,n]])+1 for n in range(np.size(cls,1))]
    temp = pd.DataFrame({"Class": temp_cls, "x": x, "y": y})
    predictedClass = predictedClass.append(temp, ignore_index=True)
    predictedClass = predictedClass.astype({"Class": 'int64'})

    return predictedClass


print("2. ;)")
naiveBayesData = naiveBayes(data,data)
print(naiveBayesData)

gnb = GaussianNB()
cls_pred = gnb.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)
print("3. ;)")
print(cls_pred)


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
    if type(da).__name__ == 'DataFrame':
        temp = pd.DataFrame({"x": xx.ravel(), "y":  yy.ravel()})
        Z_cls = np.array(naiveBayes(da,temp).Class.values)
    else:
        Z_cls = da.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_cls = Z_cls.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z_cls, cmap=cmap, zorder=0)
    # plt.contour(xx, yy, Z_cls, [x+1 for x in range(cls_n)], linewidths=1., colors='white', zorder=0)


plot_function_N(data, data[['x', 'y']].values, data['Class'].values, naiveBayesData.Class,2)
plt.legend(["True positive for class 1","False negative for class 1","True positive for class 2","False negative for class 2"],loc="lower right")
plt.title("Predicted class using created function for Naive Bayes")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("task1_0.png",dpi=150)
plt.close()

plot_function_N(gnb, data[['x', 'y']].values, data['Class'].values, cls_pred,2)
plt.legend(["True positive for class 1","False negative for class 1","True positive for class 2","False negative for class 2"],loc="lower right")
plt.title("Predicted class using function GaussianNB()")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("task1_1.png",dpi=150)
plt.close()