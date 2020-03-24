# Laboratory 0
## 1. Data types <sup> _[[1]]_</sup>
### Number
* _Integer_ - zero, positive and negative whole numbers without fractional part and having unlimited precision,
* _Float_ - positive and negative real numbers with a fractional part denoted by the decimal symbol or the scientific,
* _Complex_ - a complex number is a number with real and imaginary components.
### String
A string object is one of the sequence data types in Python. It is an immutable sequence of Unicode characters. Strings are objects of Python's built-in class 'str'. A sequence is defined as an ordered collection of items. Hence, a string is an ordered collection of characters. The sequence uses an index (starting with zero) to fetch a certain item (a character in case of a string) from it.
### List
In Python, the list is a collection of items of different data types. It is an ordered sequence of items. A list object contains one or more items, not necessarily of the same type, which are separated by comma and enclosed in square brackets [].
### Tuple
Tuple is a collection of items of any Python data type, same as the list type. Unlike the list, tuple is immutable.
### Set
A set is a collection of data types in Python, same as the list and tuple. However, it is not an ordered collection of objects. The set is a Python implementation of the set in Mathematics. A set object has suitable methods to perform mathematical set operations like union, intersection, difference, etc.

## 2. Data structure
### Vector
Vector is an array with object of the same type. Good way of creating vector is to use NumPy library.

_Example 2.1. Create vectors with `array()`, using specific data for one and two dimension array_ 
```
a = np.array([1,2,3,4])
print(a)
> [1 2 3 4]
b = np.array([[1,2],[3,4],[5,6]])
print(b)
> [[1 2]
   [3 4]
   [5 6]]
```
_Example 2.2. Create vector with functions `linspace()` and `zeros()`_
```
a = np.linspace(1,5,5)
print(a)
> [1. 2. 3. 4. 5.]
b = np.zeros(5)
print(b)
> [0. 0. 0. 0. 0.]
```
_Example 2.3. Add data type to array_
```
a = np.array([(1,4),(2,5),(3,6)], dtype=[('x','int16'),('y','f4')])
print(a)
> [(1, 4.) (2, 5.) (3, 6.)]
print(a['x'])
> [1 2 3]
print(a['y'])
> [4. 5. 6.]
```
### Categorical Data
If we want to create categorical data to this in R, we can use Pandas library. 

We can create `Series()`<sup> _[[2]]_</sup>&nbsp; object - which is one-dimensional ndarray with axis labels - and add data type.

_Example 2.4. Create categorical data with `Series()` and get numbers of vector levels_
```
a = pd.Series(["a", "b", "b", "a", "e", "b"], dtype="category")
print(a.value_counts())
> b    3
  a    2
  e    1
  dtype: int64
print(a.value_counts().b)
> 3
```


_Example 2.5. Create categorical data with `DataFrame()`<sup> _[[3]]_</sup>&nbsp; - two-dimensional, size-mutable tabular data._
```
bf = pd.DataFrame({"A": ["a", "a", "b", "c"], "B": ["d", "d", "e", "e"]})
print(bf.A.value_counts())
> a    2
  c    1
  b    1
print(bf.A.value_counts().b)
> 1
print(bf.B.value_counts().e)
> 2
```
## 3. Operations on Matrix
Matrix can be created using function `matrix()` in  NumPy, but is recommended to use regular arrays for matrices<sup> _[[4]]_</sup>. We can use NumPy functions for matrix manipulations<sup> _[[5]]_</sup>&nbsp;:
* `add()` − add elements of two matrices.

_Example 3.1. Add two matrices_
```
A = np.array([[1,1],[1,1]])
print(A)
> [[1 1]
   [1 1]]
B = np.array([[2,2],[2,2]])
print(B)
> [[2 2]
   [2 2]]
C = np.add(A,B)
print(C)
> [[3 3]
   [3 3]]
```
* `subtract()` − subtract elements of two matrices.

_Example 3.2. Subtract two matrices_
```
C = np.subtract(A,B)
print(C)
> [[-1 -1]
   [-1 -1]]
```
* `divide()` − divide elements of two matrices.

_Example 3.3. Divide elements of two matrices_
```
C = np.divide(A,B)
print(C)
> [[0.5 0.5]
   [0.5 0.5]]
```
* `multiply()` − multiply elements of two matrices.

_Example 3.4. Multiply elements of two matrices_
```
C = np.multiply(A,B)
print(C)
> [[2 2]
   [2 2]]
```
* `dot()` − It performs matrix multiplication, does not element wise multiplication.

_Example 3.5. Multiplication two matrices_
```
C = np.dot(A,B)
print(C)
> [[4 4]
   [4 4]]
```
* `sqrt()` − square root of each element of matrix.
_Example 3.6. square root of each element of matrix_
```
C = np.array([[4,4],[4,4]])
print(np.sqrt(C))
> [[2. 2.]
   [2. 2.]]
```
* `sum(x,axis)` − add to all the elements in matrix. Second argument is optional, it is used when we want to compute the column sum if axis is 0 and row sum if axis is 1.

_Example 3.7. Add all elements of matrix_
```
C = np.array([[1,2],[3,4]])
print(np.sum(C))
> 10
```
* `“.T”` − It performs transpose of the specified matrix.

_Example 3.8. Transpose matrix_
```
C = np.array([[1,4],[2,5],[3,6]])
print(C)
> [[1 4]
   [2 5]
   [3 6]]
print(C.T)
> [[1 2 3]
   [4 5 6]]
```

## 4. Control Flow tools in Python
### _for_ Statement
The for statement in python can be done in various way. Below, You can see some of its examples.

_Example 4.1. for Statement using elements of array_
```
a = ['jeden, ','dwa, ','trzy']
str = ''
for element in a:
    str += element
print(str)
> jeden, dwa, trzy
```
_Example 4.2. for Statement using elements of two-dimensional array_
```
a = [[1,4],[2,5],[3,6]]
for x, y in a:
    print("x: {}, y: {}".format(x,y))
> x: 1, y: 4
  x: 2, y: 5
  x: 3, y: 6
```
_Example 4.3. for Statement using `range()`_
```
for x in range(0,10,3):
    print(x)
> 0
  3
  6
  9
```

### _while_ Statement
`While` can be used when we want to loop what is in _while_ statement - depend of variable boolean value or condition.

_Example 4.4. while Statement_
```
x = 1
while x<4:
    print("x: {}".format(x))
    x += 1
> x: 1
  x: 2
  x: 3
```
### _if, else_ Statement
`if` and `else` statement is used to check whether a given condition is met or not.
_Example 4.5. if,else Statement_
```
x = 1
if(x == 2):
    print(X is equal 2)
else:
    print("X is not equal 2")
> X is not equal to 2
```
## 5. Functions
A function is a block of code which only runs when it is called. You can pass data, known as parameters, into a function. A function can return data as a result<sup> _[[6]]_</sup>.

_Example 5.1. Create function with parameter and call it_
```
def function(arg):
    print("Function was called")
    return math.sqrt(arg)
x = function(4)
print('Sqrt(x) = {}'.format(x))
> Function was called
  Sqrt(x) = 2.0
```
_Example 5.2. Function with undefined arguments_
```
def fun(*args):
    str = 0
    for arg in args:
        print("Added: {}".format(arg))
        str += arg
    return str
print("Sum: ",fun(1,2,3,4))
> Added: 1
  Added: 2
  Added: 3
  Added: 4
  Sum:  10
```
_Example 5.3. Import function from another file_
> hello.py
```
def say():
    print("Hello :)")
```
> main.py
```
from hello import say
say()
> Hello :)
```

## 6. Creating data visualisation
For creating plots, graphs, charts etc. - the best option is to use library `matplotlib`.
### Plots
For 2 dimensional plots You can import : `matplotlib.pyplot`

_Example 6.1. Create simple plot_
```
x = np.arange(1,11)
plt.plot(x,x**2)
plt.show()
```
<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_00/plot01.png" height="50%" width="50%">

[1]: https://www.tutorialsteacher.com/python/python-data-types
[2]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html?highlight=series#pandas.Series
[3]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame
[4]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.matrix.html
[5]: https://www.tutorialspoint.com/matrix-manipulation-in-python
[6]: https://www.w3schools.com/python/python_functions.asp