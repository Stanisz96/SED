import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    elif np.size(S, 0) != cls_n | np.size(m, 0) != cls_n | (np.size(S, 1) != np.size(m, 1)) | (np.size(S, 2) != np.size(m, 1)):
        return print("Wrong size of data2!")

    S = np.array(S)
    m = np.array(m)
    n = np.array(n)
    X = [0 for x in range(cls_n)]
    for i in range(cls_n):
        X[i] = np.random.multivariate_normal(m[i], S[i], n[i])

    return X

S1 = np.array([[4, 0], [0, 4]])
S2 = np.array([[2, 0], [0, 2]])
m1 = np.array([-3, -1])
m2 = np.array([2, 2])
m = [m1, m2]
S = [S1, S2]
X = generateGaussData(S, m, [4,8],2)

print(X)