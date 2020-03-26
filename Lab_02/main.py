import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.stats as st
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#### EXERCISE 1: "GĘSTOŚĆ ROZKŁADÓW 2D" ####

# Matrices COV for classes
S1 = np.array([[4, 0], [0, 4]])
S2 = np.array([[2, 0], [0, 2]])

# Mean
m1 = np.array([-3, -1])
m2 = np.array([2, 2])

# Number of points in classes
n1 = 200
n2 = 150
n = n1 + n2

# Generate normal distribution
X1 = np.random.multivariate_normal(m1, S1, n1)
X2 = np.random.multivariate_normal(m2, S2, n2)

# Classes components
x1, y1 = zip(*X1)
x2, y2 = zip(*X2)

# Create plot
# plt.plot(x1,y1,".",color="red", markersize=12)
# plt.plot(x2,y2,".",color="green", markersize=12)
# plt.xlabel(r'$X$',size="x-large")
# plt.ylabel(r'$Y$',size="x-large")
# plt.title("Plot for two classes",size="xx-large")
# plt.grid(True)

# Add 2D PDF to plot
# ## Define borders
# x1min = min(x1) - (max(x1)-min(x1))/10
# x1max = max(x1) + (max(x1)-min(x1))/10
# y1min = min(y1) - (max(y1)-min(y1))/10
# y1max = max(y1) + (max(y1)-min(y1))/10
# ## Create meshgrid
# borderX1, borderY1 = np.mgrid[x1min:x1max:100j, y1min:y1max:100j]
# # TEST
# positions = np.vstack([borderX1.ravel(), borderY1.ravel()])
# values = np.vstack([x1, y1])
# kernel = st.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, borderX1.shape)
#
# ## Define borders
# x2min = min(x2) - (max(x2)-min(x2))/10
# x2max = max(x2) + (max(x2)-min(x2))/10
# y2min = min(y2) - (max(y2)-min(y2))/10
# y2max = max(y2) + (max(y2)-min(y2))/10
# ## Create meshgrid
# borderX2, borderY2 = np.mgrid[x2min:x2max:100j, y2min:y2max:100j]
# # TEST
# positions2 = np.vstack([borderX2.ravel(), borderY2.ravel()])
# values2 = np.vstack([x2, y2])
# kernel2 = st.gaussian_kde(values2)
# f2 = np.reshape(kernel2(positions2).T, borderX2.shape)
#
# plt.contourf(borderX1,borderY1,f)
# plt.contourf(borderX2,borderY2,f2)
# # plt.imshow(np.rot90(f) cmap='coolwarm', extent=[x1min, x1max, y1min, y1max])

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
plt.close()
# plt.savefig("Exercise01.png", dpi=150)
# plt.show()


#### EXERCISE 2: "METODA LDA" ####

# Create data structure with classes

data = pd.DataFrame({"Class": [1 for x in range(np.size(X1, 0))], "x": x1, "y": y1})
temp = pd.DataFrame({"Class": [2 for x in range(np.size(X2, 0))], "x": x2, "y": y2})
data = data.append(temp, ignore_index=True)


# print(data.loc[data['Class']==1,['x','y']])

# Discriminant Analysis

## plot function
def plot_function(da, X, cls, cls_pred):
    plt.figure(figsize=(8, 6))

    tp = (cls == cls_pred)  # True Positive
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

plot_function(lda, data[['x', 'y']].values, data['Class'].values, cls_pred)
plt.savefig("Exercise02_lda.png", dpi=150)
plt.show()

## Quadratic Discriminant Analysis
qda = QDA(store_covariance=True)
cls_pred = qda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)

plot_function(qda, data[['x', 'y']].values, data['Class'].values, cls_pred)
plt.savefig("Exercise02_qda.png", dpi=150)
plt.show()
