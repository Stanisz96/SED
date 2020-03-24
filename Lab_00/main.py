import numpy as np
import pandas as pd
import math
# from hello import say
import matplotlib.pyplot as plt

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
# print(C)
# print(C.T)

# 4. Control flow
## For Statement
a = ['jeden, ','dwa, ','trzy']
str = ''
for element in a:
    str += element
# print(str)
a = [[1,4],[2,5],[3,6]]
# for x, y in a:
#     print("x: {}, y: {}".format(x,y))

# for x in range(0,10,3):
#     print(x)

## While Statement
x = 1
# while x<5:
#     print("x: {}".format(x))
#     x += 1


##if, else
# x = 1
# if(x == 2):
#     print(x)
# else:
#     print("X is not equal 2")

# 5. Functions
def function(arg):
    print("Function was called")
    return math.sqrt(arg)
# x = function(4)
# print(x)
def fun(*args):
    str = 0
    for arg in args:
        print("Added: {}".format(arg))
        str += arg
    return str
# print("Sum: ",fun(1,2,3,4))

# say()

# 6. Visualization for python
## Plots

# x = np.arange(1,11)
# plt.plot(x,x**2,'.',color="#2eb82e", markersize=15)
# plt.xlabel(r'$x$',size="x-large")
# plt.ylabel(r'$f(x)=x^2$',size="x-large")
# plt.title("Function f(x)",size="xx-large")
# plt.grid(True)
# # plt.savefig("plot02.png")
# plt.show()

# hist

# x = np.array([1,1,1,2,2,4,10])
# plt.hist(x,bins=5)
# plt.ylabel("Frequency")
# plt.xlabel("x")
# plt.title("Histogram of x")
# plt.show()

#histogram with normal distribution

# x = np.random.normal(0,0.1,1000)
# plt.hist(x,bins=20,color="#2eb82e",edgecolor="black")
# plt.ylabel("Frequency")
# plt.xlabel("x")
# plt.title("Histogram of x")
# plt.savefig("hist02.png")
# plt.show()

# permutation
# x = np.arange(9)
# print(x)
# y = np.random.permutation(x)
# print(y)

# distribution
laplace = np.random.laplace(4,1,4000)
normal = np.random.normal(4,0.2,4000)
poisson = np.random.poisson(4,4000)
gamma = np.random.gamma(2,2,4000)

figure, axes = plt.subplots(2,2)
figure.tight_layout(pad=3.0)

axes[0,0].hist(laplace,bins=15,color="#2eb82e",edgecolor="black")
axes[0,0].set_title("Laplace distribution",size="large")

axes[0,1].hist(normal,bins=15,color="#00a3cc",edgecolor="black")
axes[0,1].set_title("Normal distribution",size="large")

axes[1,0].hist(poisson,bins=15,color="#e65c00",edgecolor="black")
axes[1,0].set_title("Poisson distribution",size="large")

axes[1,1].hist(gamma,bins=15,color="#86b300",edgecolor="black")
axes[1,1].set_title("Gamma distribution",size="large")

for x in axes:
    for y in x:
        y.set_xlabel(r'$x$', size="large")
        y.set_ylabel("Frequency", size="large")

plt.show()
# plt.savefig("fig01.png")
