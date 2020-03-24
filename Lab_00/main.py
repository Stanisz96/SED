import numpy as np
import pandas as pd

# 2. Data structure
## Vector
a = np.array([1,2,3,4])
b = np.array([[1,2],[3,4],[5,6]])

a = np.linspace(1,5,5)
b = np.zeros(5)

a = np.array([(1,4),(2,5),(3,6)], dtype=[('x','int16'),('y','f4')])
del a,b
## Categorical Data
a = pd.Series(["a", "b", "b", "a", "e", "b"], dtype="category")
# print(a.value_counts())
# print(a.value_counts().b)

bf = pd.DataFrame({"A": ["a", "a", "b", "c"], "B": ["d", "d", "e", "e"]})
# print(bf.A.value_counts())
# print(bf.A.value_counts().b)
# print(bf.B.value_counts().e)

# 3. Operations on matrices
## Add
A = np.array([[1,1],[1,1]])
# print(A)
B = np.array([[2,2],[2,2]])
# print(B)
C = np.add(A,B)
# print(C)

## Subtract
C = np.subtract(A,B)
# print(C)

## Divide
C = np.divide(A,B)
# print(C)

## Multiply
C = np.multiply(A,B)
# print(C)

## Multiplication
C = np.dot(A,B)
# print(C)

## Sqrt
C = np.array([[4,4],[4,4]])
# print(np.sqrt(C))

## Sum
C = np.array([[1,2],[3,4]])
# print(np.sum(C))

## Transpose
C = np.array([[1,4],[2,5],[3,6]])
print(C)
print(C.T)
# print(C)