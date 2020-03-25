import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

#### EXERCISE 1: "GĘSTOŚĆ ROZKŁADÓW 2D" ####

# Matrices COV for classes
S1 = np.array([[4,0],[0,4]])
S2 = np.array([[2,0],[0,2]])

# Mean
m1 = np.array([-3,-1])
m2 = np.array([2,2])

# Number of points in classes
n1 = 200
n2 = 150
n = n1 + n2

# Generate normal distribution
X1 = np.random.multivariate_normal(m1,S1,n1)
X2 = np.random.multivariate_normal(m2,S2,n2)

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

pdf_X1 =  st.multivariate_normal.pdf(X1,m1,S1)
pdf_X2 =  st.multivariate_normal.pdf(X2,m2,S2)


plt.figure(figsize=(8,6))
CS1 = plt.tricontour(x1,y1,pdf_X1,colors=["#ffe6e6","#ffb3b3","#ff8080","#ff4d4d","#ff0000","#cc0000","#990000","#660000"])
CS2 = plt.tricontour(x2,y2,pdf_X2,colors=["#e6ffe6","#b3ffb3","#80ff80","#33ff33","#00e600","#00b300","#006600","#001a00"])
plt.clabel(CS1, fontsize=11)
plt.clabel(CS2, fontsize=11)
plt.plot(x1,y1,".",color="red", markersize=8)
plt.plot(x2,y2,".",color="green", markersize=8)
plt.xlabel(r'$X$',size="x-large")
plt.ylabel(r'$Y$',size="x-large")
plt.title("Plot for two classes",size="xx-large")
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.grid(True)
# plt.savefig("Exercise01.png", dpi=150)


plt.show()