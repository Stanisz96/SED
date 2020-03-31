import pandas as pd
import numpy as np
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.naive_bayes import GaussianNB
# import matplotlib.pyplot as plt






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
print(data)

# Naive Bayes
pi_1 = n1/n
pi_2 = n2/n

# print(1/np.sqrt(2*np.pi*np.var(
print(data.loc[data.Class == 1, ['x','y']].var())
# )))

def naiveBayes(trainingSet,testSet):
    # probabilityData = pd.DataFrame({"x": [],"p(x|1)": [],"p(x|2)": [],"|": [],"y": [],"p(y|1)": [],"p(y|2)": []})
    predictedClass = pd.DataFrame({"Class": [], "x": [], "y": []})
    pi = [trainingSet.Class.loc[trainingSet.Class == 1].count() / trainingSet.Class.count(),
          trainingSet.Class.loc[trainingSet.Class == 2].count() / trainingSet.Class.count()]
    varX = [trainingSet.loc[trainingSet.Class == 1, 'x'].var(), trainingSet.loc[trainingSet.Class == 2, 'x'].var()]
    varY =[trainingSet.loc[trainingSet.Class == 1, 'y'].var(), trainingSet.loc[trainingSet.Class == 2, 'y'].var()]
    mX = [trainingSet.loc[trainingSet.Class == 1, 'x'].mean(),trainingSet.loc[trainingSet.Class == 2, 'x'].mean()]
    mY = [trainingSet.loc[trainingSet.Class == 1, 'y'].mean(),trainingSet.loc[trainingSet.Class == 2, 'y'].mean()]
    px, py = [0,0], [0,0]
    cls = [0,0]
    for x,y in zip(testSet.x, testSet.y):
        for k in range(trainingSet.Class.max(axis=0)):
            px[k] = 1 / np.sqrt(2 * np.pi * varX[k]) * np.exp(-(x-mX[k])**2/(2*varX[k]))
            py[k] = 1 / np.sqrt(2 * np.pi * varY[k]) * np.exp(-(x-mY[k])**2/(2*varY[k]))
            cls[k] = pi[k] * px[k] * py[k]
        temp = pd.Series({"Class": np.argmax(cls,axis=0)+1, "x": x, "y": y})
        predictedClass = predictedClass.append(temp, ignore_index=True)
        # temp = pd.Series({"x": x,"p(x|1)": px[0],"p(x|2)": px[1],"|": "|","y": y,"p(y|1)": py[0],"p(y|2)": py[1]})
        # probabilityData = probabilityData.append(temp, ignore_index=True)
    predictedClass = predictedClass.astype({"Class": 'int64'})
    # dataframe for predicted classes

    # predictedData = pd.DataFrame({"Class": [],"x": [], "y": []})

    # for k in range(trainingSet.Class.max(axis=0)):
    #     cls = pi[k]*


    # p = np.zeros((2,2))     # x|1 y|1
    # # p matrix elements ->  # x|2 y|2
    #
    # p[0,1] = data.x.loc[data.x == ].count()

    return predictedClass


print(naiveBayes(data,data))


# print(data.x.loc[data.x>3].count())