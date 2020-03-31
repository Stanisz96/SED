import numpy as np
import matplotlib.pyplot as plt

mu1, S1, n1 = np.array([-1, 1]), np.array([[1, 0], [0, 1]]), 30
mu2, S2, n2 = np.array([2, 4]), np.array([[1, 0], [0, 1]]), 30
mu3, S3, n3 = np.array([-2, 2]), np.array([[1, 0], [0, 1]]), 30

# Set observations for classes
X1 = np.random.multivariate_normal(mu1, S1, n1)
X2 = np.random.multivariate_normal(mu2, S2, n2)
X3 = np.random.multivariate_normal(mu3, S3, n3)

# Mean
m1 = np.mean(X1, axis=0)[np.newaxis]
m2 = np.mean(X2, axis=0)[np.newaxis]
m3 = np.mean(X3, axis=0)[np.newaxis]
m = np.mean([m1, m2, m3], axis=0)
# Covariance matrix
S1 = np.cov(X1.T)
S2 = np.cov(X2.T)
S3 = np.cov(X3.T)

## All observations and number of Classes
n = sum([n1, n2, n3])
g = 3

# Determination of vector a

## Inter-group variation matrix
B = (n1 * np.dot((m1 - m).T, (m1 - m)) + n2 * np.dot((m2 - m).T, (m2 - m)) + n3 * np.dot((m3 - m).T, (m3 - m))) / (
            g - 1)

# Determination of vector a
## Within-group variation matrix
W = ((n1 - 1) * S1 + (n2 - 1) * S2 + (n3 - 1) * S3) / (n - g)

## Auxiliary matrix
U = np.dot(np.linalg.inv(W), B)

## Eigenvalues and eigenvectors
lambda_w, lambda_v = np.linalg.eig(U)
## eigenvector with max eigenvalue
a = lambda_v[np.argmax(lambda_w)]

# Plot for Classes
## Vector components
(x1, y1), (x2, y2), (x3, y3) = zip(*X1), zip(*X2), zip(*X3)
ClassesVec = np.array([[x1, y1], [x2, y2], [x3, y3]])

## Classes
plt.figure(figsize=(8, 6))
for nrClass in range(len(ClassesVec)):
    plt.plot(ClassesVec[nrClass][0], ClassesVec[nrClass][1], 'o', markersize=4, label='Class %d' % (nrClass + 1))
    # for point in range(np.size(ClassesVec[nrClass][0], 0)):
    #     plt.text(ClassesVec[nrClass][0][point] - 0.1, ClassesVec[nrClass][1][point] - 0.1, '%d' % (nrClass + 1))

# Plot for vector a
coef = np.polyfit([0, -a[0]], [0, a[1]], 1)
x_a = np.linspace(-5, 15, 100)
y_a = coef[0] * x_a + coef[1]
plt.plot(x_a, y_a, ':r')


# Projecting points in Classes

def ProjectPoints(_X, _a):
    A = -(_a[1] / _a[0])
    newXx = (_X[1] * A + _X[0]) / (A ** 2 + 1)
    newXy = A * newXx
    return np.array([newXx, newXy])


colors = ['blue', 'orange', 'green', 'red', 'purple']
for nrClass in range(len(ClassesVec)):
    ProjectClass = ProjectPoints(ClassesVec[nrClass], a)
    plt.plot(ProjectClass[0], ProjectClass[1], 'o', markersize=4, color=colors[nrClass])

## Settings
plt.legend(loc="upper left")
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)
plt.savefig("task0.png", dpi=150)
plt.show()
