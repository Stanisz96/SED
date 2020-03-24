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

[1]: https://www.tutorialsteacher.com/python/python-data-types
[2]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html?highlight=series#pandas.Series
[3]: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame