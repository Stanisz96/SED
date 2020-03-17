import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#### EXERCISE 1: "DWIE KLASY" ####
# Coordintes of observations belonging to individual classes
X1 = np.array([[2, 2], [2, 1], [2, 3], [1, 2], [3, 2]])
X2 = np.array([[6, 0], [6, 1], [6, -1], [5, 0], [7, 0]])

# Mean
m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)

# Covariance matrix
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)

# Determination of vector a
## Within-group variation matrix
W = ((np.size(X1, 0) - 1) * S1 + (np.size(X2, 0) - 1) * S2) / (np.size(X1, 0) + np.size(X2, 0) - 2)

# Vector a
a = np.dot(np.linalg.inv(W), (m2 - m1))

# Plot for X1 & X2
## Vector components
x1, y1 = zip(*X1)
x2, y2 = zip(*X2)

## Classes
fig1 = plt.figure(1)
plt.plot(x1, y1, 'o', markersize=12, color="#B2DDF7")
plt.plot(x2, y2, 'o', markersize=12, color="#BCFFDB")
# Plot for vector a
coef = np.polyfit([0, -a[0]], [0, a[1]], 1)
x_a = np.linspace(-1, 15, 16)
y_a = coef[0] * x_a + coef[1]
plt.plot(x_a, y_a, ":r")
## Function dividing Classes
x_a = np.linspace(0, 6, 20)
b = -1 / 2 * (np.dot(a, (m1 + m2)))
y_a = (-a[0] * x_a - b) / a[1]
plt.plot(x_a, y_a, '-r')

## Settings
plt.legend(("Class 1", "Class 2", "Function for vector a", "Function dividing Classes"), loc="lower left")
plt.xlim(-2, 8)
plt.ylim(-4, 4)
plt.xlabel("X")
plt.ylabel("Y")
for point in range(np.size(X1, 0)):
    plt.text(x1[point] - 0.1, y1[point] - 0.1, "1")
    plt.text(x2[point] - 0.05, y2[point] - 0.1, "2")
plt.grid(True)
# fig1.savefig('Exercise01.png')
plt.close()

#### EXERCISE 2: "DWIE KLASY, ROZKLAD GAUSSA" ####

# Set props for normal distribution
mu1, S1, n1 = np.array([2, 2]), np.array([[1, 0], [0, 1]]), 60
mu2, S2, n2 = np.array([6, 0]), np.array([[1, 0], [0, 1]]), 60

# Set observations for classes
X1 = np.random.multivariate_normal(mu1, S1, n1)
X2 = np.random.multivariate_normal(mu2, S2, n2)

# Mean
m1 = np.mean(X1, axis=0)
# m1_test = X1_test_view.mean(axis=0)
m2 = np.mean(X2, axis=0)
# Covariance matrix
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)

# Determination of vector a
## Within-group variation matrix
W = ((np.size(X1, 0) - 1) * S1 + (np.size(X2, 0) - 1) * S2) / (np.size(X1, 0) + np.size(X2, 0) - 2)

# Vector a
a = np.dot(np.linalg.inv(W), (m2 - m1))

# Plot for X1 & X2
## Vector components
x1, y1 = zip(*X1)
x2, y2 = zip(*X2)

## Classes
fig2 = plt.figure(2)
plt.plot(x1, y1, 'o', markersize=8, color="#B2DDF7")
plt.plot(x2, y2, 'o', markersize=8, color="#BCFFDB")
## Function for vector a
ax = [-a[0], 0, a[0], 2 * a[0]]
ay = [-a[1], 0, a[1], 2 * a[1]]
plt.plot(ax, ay, ":r")
## Function dividing Classes
x_a = np.linspace(0, 6, 20)
b = -1 / 2 * (np.dot(a, (m1 + m2)))
y_a = (-a[0] * x_a - b) / a[1]
plt.plot(x_a, y_a, '-r')

## Settings
plt.legend(("Class 1", "Class 2", "Function for vector a", "Function dividing Classes"), loc="lower left")
plt.xlim(-2, 8)
plt.ylim(-4, 4)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
# fig2.savefig('Exercise02.png')
plt.close()

#### EXERCISE 3: "DWIE KLASY, 3D" ####

# Set observations for classes
X1 = np.array([[2, 2, 2], [2, 1, 2], [2, 3, 2], [1, 2, 2], [3, 2, 2], [2, 2, 1], [2, 2, 3]])
X2 = np.array([[4, 4, 4], [4, 3, 4], [4, 5, 4], [3, 4, 4], [5, 4, 4], [4, 4, 3], [4, 4, 5]])

# Mean
m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)

# Covariance matrix
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)

# Determination of vector a
## Within-group variation matrix
W = ((np.size(X1, 0) - 1) * S1 + (np.size(X2, 0) - 1) * S2) / (np.size(X1, 0) + np.size(X2, 0) - 2)

# Vector a
a = np.dot(np.linalg.inv(W), (m2 - m1))

# Plot for X1 & X2
## Vector components
x1, y1, z1 = zip(*X1)
x2, y2, z2 = zip(*X2)

fig3 = plt.figure(3)
PlotFig = fig3.add_subplot(111, projection='3d')
PlotFig.plot(x1, y1, z1, 'o', markersize=12, color="#F3A712", label="Class 1")
PlotFig.plot(x2, y2, z2, 'o', markersize=12, color="#86CB92", label="Class 2")
for point in range(np.size(X1, 0)):
    PlotFig.text(x1[point] - 0.05, y1[point] - 0.05, z1[point] - 0.1, "1")
    PlotFig.text(x2[point] - 0.05, y2[point] - 0.05, z2[point] - 0.1, "2")

## Function for vector a
ax = [-a[0], 0, a[0], 2 * a[0]]
ay = [-a[1], 0, a[1], 2 * a[1]]
az = [-a[2], 0, a[2], 2 * a[2]]

PlotFig.plot(ax, ay, az, ":r")

## Function dividing Classes
x_a = np.linspace(-1, 5, 20)
b = -1 / 2 * (np.dot(a, (m1 + m2)))
y_a = (-a[0] * x_a - b) / a[1]
x_a, y_a = np.meshgrid(range(6), range(6))
z_a = (-a[0] * x_a - a[1] * y_a - b) / a[2]
# PlotFig.plot_surface(x_a, y_a, z_a ,alpha=0.8,label="Plane dividing Classes")
surf = PlotFig.plot_surface(x_a, y_a, z_a, alpha=0.8, label="Plane dividing Classes")
surf._facecolors2d = surf._facecolors3d
surf._edgecolors2d = surf._edgecolors3d

# Settings
PlotFig.set_xlim(0, 5)
PlotFig.set_ylim(0, 5)
PlotFig.set_zlim(0, 5)
PlotFig.set_xlabel("X")
PlotFig.set_ylabel("Y")
PlotFig.set_zlabel("Z")
PlotFig.legend()

fig3.savefig('Exercise03.png')
plt.close()

#### EXERCISE 4: "WIELE KLAS" ####

# Coordinates of observations belonging to individual classes
X1 = np.array([[1, 2], [2, 3], [2, 2], [2, 1], [3, 2]])
X2 = np.array([[3, 4], [4, 5], [4, 4], [4, 3], [5, 4]])
X3 = np.array([[4, 6], [5, 7], [5, 6], [5, 5], [6, 6]])
X4 = np.array([[8, 8], [9, 9], [9, 8], [9, 7], [10, 8]])
X5 = np.array([[9, 10], [10, 11], [10, 10], [10, 9], [11, 10]])

# Mean
m1 = np.mean(X1, axis=0)[np.newaxis]
m2 = np.mean(X2, axis=0)[np.newaxis]
m3 = np.mean(X3, axis=0)[np.newaxis]
m4 = np.mean(X4, axis=0)[np.newaxis]
m5 = np.mean(X5, axis=0)[np.newaxis]
## Mean for all points
m = np.mean([m1, m2, m3, m4, m5], axis=0)

# Covariance matrix
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)
S3 = np.cov(X3.T)
S4 = np.cov(X4.T)
S5 = np.cov(X5.T)

# Number of observations in Classes
n1 = np.size(X1, 0)
n2 = np.size(X2, 0)
n3 = np.size(X3, 0)
n4 = np.size(X4, 0)
n5 = np.size(X5, 0)
## All observations and number of Classes
n = sum([n1, n2, n3, n4, n5])
g = 5

# Determination of vector a

## Inter-group variation matrix
B = (n1 * np.dot((m1 - m).T, (m1 - m)) + n2 * np.dot((m2 - m).T, (m2 - m)) + n3 * np.dot((m3 - m).T,
                                                                                         (m3 - m)) + n4 * np.dot(
    (m4 - m).T, (m4 - m)) + n5 * np.dot((m5 - m).T, (m5 - m))) / (g - 1)

## Within-group variation matrix
W = ((n1 - 1) * S1 + (n2 - 1) * S2 + (n3 - 1) * S3 + (n4 - 1) * S4 + (n5 - 1) * S5) / (n - g)

## Auxiliary matrix
U = np.dot(np.linalg.inv(W), B)

## Eigenvalues and eigenvectors
lambda_w, lambda_v = np.linalg.eig(U)
## eigenvector with max eigenvalue
a = lambda_v[np.argmax(lambda_w)]

# Plot for Classes
## Vector components
(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5) = zip(*X1), zip(*X2), zip(*X3), zip(*X4), zip(*X5)
ClassesVec = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]])

## Classes
fig4 = plt.figure(4)
for nrClass in range(len(ClassesVec)):
    plt.plot(ClassesVec[nrClass][0], ClassesVec[nrClass][1], 'o', markersize=12, label='Class %d' % (nrClass + 1))
    for point in range(np.size(ClassesVec[nrClass][0], 0)):
        plt.text(ClassesVec[nrClass][0][point] - 0.1, ClassesVec[nrClass][1][point] - 0.1, '%d' % (nrClass + 1))

# Plot for vector a
coef = np.polyfit([0, -a[0]], [0, a[1]], 1)
x_a = np.linspace(-5, 15, 100)
y_a = coef[0] * x_a + coef[1]
plt.plot(x_a, y_a, ':r')

# Projecting points in Classes

def ProjectPoints(_X,_a):
    A = -(_a[1] / _a[0])
    newXx = (_X[1] * A +_X[0])/ (A**2+1)
    newXy = A*newXx
    # newXx = (_X[1]+_X[0]-_a[0]*_a[1])/(1+_a[0]**2)
    # newXy = -newXx/_a[0]+_X[1]+_X[0]/_a[0]
    # return np.array([newXx,newXy])
    return np.array([newXx,newXy])
colors = ['blue','orange','green','red','purple']
for nrClass in range(len(ClassesVec)):
    ProjectClass = ProjectPoints(ClassesVec[nrClass],a)
    plt.plot(ProjectClass[0], ProjectClass[1], 'o', markersize=7,color=colors[nrClass])


## Settings
plt.legend(loc="upper left")
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)

plt.show()
# fig4.savefig('Exercise04.png')
# plt.close()
