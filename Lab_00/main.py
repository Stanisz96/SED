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

a = pd.Series(["a", "b", "b", "a", "e", "b"], dtype="category")
# print(a.value_counts())
# print(a.value_counts().b)

bf = pd.DataFrame({"A": ["a", "a", "b", "c"], "B": ["d", "d", "e", "e"]})
print(bf.A.value_counts())
print(bf.A.value_counts().b)
print(bf.B.value_counts().e)