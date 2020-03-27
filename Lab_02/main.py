import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.stats as st
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB

#### EXERCISE 1: "2D DISTRIBUTION DENSITY" ####

# Matrices COV for classes
S1 = np.array([[4, 0], [0, 4]])
S2 = np.array([[2, 0], [0, 2]])
S3 = np.array([[2, 1], [1, 2]])
S4 = np.array([[1, 0], [0, 1]])

# Mean
m1 = np.array([-3, -1])
m2 = np.array([2, 2])
m3 = np.array([-2, 3])
m4 = np.array([1, -2])

# Number of points in classes
n1 = 200
n2 = 150
n3 = 150
n4 = 100
n = n1 + n2 + n3 + n4

# Generate normal distribution
X1 = np.random.multivariate_normal(m1, S1, n1)
X2 = np.random.multivariate_normal(m2, S2, n2)
X3 = np.random.multivariate_normal(m3, S3, n3)
X4 = np.random.multivariate_normal(m4, S4, n4)

# Classes components
x1, y1 = zip(*X1)
x2, y2 = zip(*X2)
x3, y3 = zip(*X3)
x4, y4 = zip(*X4)


pdf_X1 = st.multivariate_normal.pdf(X1, m1, S1)
pdf_X2 = st.multivariate_normal.pdf(X2, m2, S2)

plt.figure(figsize=(8, 6))
CS1 = plt.tricontour(x1, y1, pdf_X1,
                     colors=["#ffe6e6", "#ffb3b3", "#ff8080", "#ff4d4d", "#ff0000", "#cc0000", "#990000", "#660000"])
CS2 = plt.tricontour(x2, y2, pdf_X2,
                     colors=["#e6ffe6", "#b3ffb3", "#80ff80", "#33ff33", "#00e600", "#00b300", "#006600", "#001a00"])
plt.clabel(CS1, fontsize=11)
plt.clabel(CS2, fontsize=11)
plt.plot(x1, y1, ".", color="red", markersize=8)
plt.plot(x2, y2, ".", color="green", markersize=8)
plt.xlabel(r'$X$', size="x-large")
plt.ylabel(r'$Y$', size="x-large")
plt.title("Plot for two classes", size="xx-large")
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.grid(True)
plt.savefig("Exercise01.png", dpi=150)
plt.close()
# plt.show()


#### EXERCISE 2: "LDA & QDA METHODS" ####

# Create data structure with classes

data = pd.DataFrame({"Class": [1 for x in range(np.size(X1, 0))], "x": x1, "y": y1})
temp = pd.DataFrame({"Class": [2 for x in range(np.size(X2, 0))], "x": x2, "y": y2})
data = data.append(temp, ignore_index=True)

# Discriminant Analysis

## plot function
def plot_function(da, X, cls, cls_pred):
    plt.figure(figsize=(8, 6))

    tp = (cls == cls_pred)
    tp1, tp2 = tp[cls == 1], tp[cls == 2]
    X1, X2 = X[cls == 1], X[cls == 2]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    X2_tp, X2_fp = X2[tp2], X2[~tp2]

    # class 1: points
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='#ff4d4d')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x', s=20, color='#e60000')
    # class 2: points
    plt.scatter(X2_tp[:, 0], X2_tp[:, 1], marker='.', color='#00ace6')
    plt.scatter(X2_fp[:, 0], X2_fp[:, 1], marker='x', s=20, color='#007399')

    # class 1 & 2: areas
    nx, ny = 200, 200
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = da.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='coolwarm_r',
                   norm=colors.Normalize(vmax=1, vmin=0), alpha=0.1, zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white', zorder=0)

    # class 1 & 2: means
    if "GaussianNB" in str(da):
        plt.plot(da.theta_[0][0], da.theta_[0][1],
                 '*', color='yellow', markersize=12, markeredgecolor='grey')
        plt.plot(da.theta_[1][0], da.theta_[1][1],
                 '*', color='yellow', markersize=12, markeredgecolor='grey')
    else:
        plt.plot(da.means_[0][0], da.means_[0][1],
                 '*', color='yellow', markersize=12, markeredgecolor='grey')
        plt.plot(da.means_[1][0], da.means_[1][1],
                 '*', color='yellow', markersize=12, markeredgecolor='grey')

    plt.grid(False)
    plt.xlabel(r'$X$', size="x-large")
    plt.ylabel(r'$Y$', size="x-large")
    # plt.legend(["Class 1","Class 1 predicted for class 2","Class 2","Class 2 predicted for class 1"])


## Linear Discriminant Analysis
lda = LDA(solver="svd", store_covariance=True)
cls_pred = lda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)

plt.figure(figsize=(8, 6))
plot_function(lda, data[['x', 'y']].values, data['Class'].values, cls_pred)
plt.savefig("Exercise02_lda.png", dpi=150)
plt.close()
# plt.show()


## Quadratic Discriminant Analysis
qda = QDA(store_covariance=True)
cls_pred = qda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)

plt.figure(figsize=(8, 6))
plot_function(qda, data[['x', 'y']].values, data['Class'].values, cls_pred)
plt.savefig("Exercise02_qda.png", dpi=150)
plt.close()
# plt.show()


#### EXERCISE 3: "NAIVE BAYES" ####
gnb = GaussianNB()
cls_pred = gnb.partial_fit(data[['x', 'y']].values, data['Class'].values,np.unique(data['Class'].values)).predict(data[['x', 'y']].values)

plt.figure(figsize=(8, 6))
plt.title("Bayes")
plot_function(gnb, data[['x', 'y']].values, data['Class'].values, cls_pred)
plt.savefig("Exercise03.png", dpi=150)
plt.close()

plt.close()


#### EXERCISE 4: "MANY CLASSES" ####
## Set data
data = pd.DataFrame({"Class": [1 for x in range(np.size(X1, 0))], "x": x1, "y": y1})
temp = pd.DataFrame({"Class": [2 for x in range(np.size(X2, 0))], "x": x2, "y": y2})
temp2 = pd.DataFrame({"Class": [3 for x in range(np.size(X3, 0))], "x": x3, "y": y3})
temp3 = pd.DataFrame({"Class": [4 for x in range(np.size(X4, 0))], "x": x4, "y": y4})
data = data.append(temp, ignore_index=True)
data = data.append(temp2, ignore_index=True)

## Create plot for n classes (max 6cls)
def plot_function(da, X, cls, cls_pred, cls_n):
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
    nx, ny = 300, 300
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z_cls = da.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_cls = Z_cls.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z_cls, cmap=cmap, zorder=0)
    plt.contour(xx, yy, Z_cls, [x+1 for x in range(cls_n)], linewidths=1., colors='white', zorder=0)


## For 3 classes
lda = LDA(solver="svd", store_covariance=True)
cls_pred = lda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)

plt.figure(figsize=(8, 6))
plot_function(lda, data[['x', 'y']].values, data['Class'].values, cls_pred,3)
plt.savefig("Exercise04_lda_3.png")
plt.close()

qda = QDA(store_covariance=True)
cls_pred = qda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)

plt.figure(figsize=(8, 6))
plot_function(qda, data[['x', 'y']].values, data['Class'].values, cls_pred,3)
plt.savefig("Exercise04_qda_3.png")
plt.close()


## For 4 classes
data = data.append(temp3, ignore_index=True)


lda = LDA(solver="svd", store_covariance=True)
cls_pred = lda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)

plt.figure(figsize=(8, 6))
plot_function(lda, data[['x', 'y']].values, data['Class'].values, cls_pred,4)
plt.savefig("Exercise04_lda_4.png")
plt.close()

qda = QDA(store_covariance=True)
cls_pred = qda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)

plt.figure(figsize=(8, 6))
plot_function(qda, data[['x', 'y']].values, data['Class'].values, cls_pred,4)
plt.savefig("Exercise04_qda_4.png")
plt.close()


