# Task 1
## The content of the task
Using the probability formula for the naive Bayes classifier:
><img src="http://latex.codecogs.com/gif.latex?%5Cboldsymbol%7B%5Cmathbf%7Bp%7D%7D%281%7C%5Cboldsymbol%7B%5Cmathbf%7Bx%7D%7D%29%20%5Cpropto%20%5Cpi_1%5Cboldsymbol%7B%5Cmathbf%7Bp%7D%7D%28%5Cboldsymbol%7B%5Cmathbf%7Bx%7D%7D%7C1%29%20%3D%5Cpi_1%5Cboldsymbol%7B%5Cmathbf%7Bp%7D%7D%28x%7C1%29%5Cboldsymbol%7B%5Cmathbf%7Bp%7D%7D%28y%7C1%29" />
and assuming that the probability density in individual classes is described by
the Gaussian distribution, demonstrate equivalence between this approach and the
results obtained using function `GaussianNB()` in the case of observations meeting
the following characteristics:
* 2 classes,
* In _class 1_: 40 observations, in _class 2_: - 30 observations
* Observations from normal distribution with covariance matrix:

    ><img src="http://latex.codecogs.com/gif.latex?S%3D%5Cbegin%7Bpmatrix%7D%204%20%260%20%5C%5C%200%20%26%204%20%5Cend%7Bpmatrix%7D" />

* Means for normal distribution:

    ><img src="http://latex.codecogs.com/gif.latex?m_1%3D%5Cbegin%7Bpmatrix%7D%20-3%20%5C%5C%20-1%20%5Cend%7Bpmatrix%7D%2C%5C%2C%20%5C%3B%20m_2%3D%5Cbegin%7Bpmatrix%7D%202%20%5C%5C%202%20%5Cend%7Bpmatrix%7D" />
    
## Solution
### Generate data
To generate observations for classes from normal distribution is used created function
`generateGaussData()`. This function return _DataFrame_ object with normal distribution
observations - depending on parameters:
* `S` - n-element array of matrix covariances for n classes
* `m` - n-element array of matrix means for n classes
* `n` - n-element array of number of points for every class
* `cls_n` - number of classes

Based on the task conditions, was created data:
```
data = generateGaussData(S, m, [n1, n2], 2)
print(data)
>    Class         x         y
 0       1  0.028639 -4.174586
 1       1 -4.279737 -2.158976
 2       1 -3.713896 -3.505592
 3       1 -0.165549  0.760564
 4       1 -5.649381  0.897266
 ..    ...       ...       ...
 69      2  3.653877  4.529459
```

### Naive Bayes
To predict class using naive Bayes method, it is needed to calculate:
* <img src="http://latex.codecogs.com/gif.latex?%5Cpi_i" /> - probability of occurrence of the i-th class estimating as number of i-th class observations divided by number of all observations
* <img src="http://latex.codecogs.com/gif.latex?%5Cmathbf%7Bp%7D%28%5Cmathbf%7Bx^i%7D%7Ck%29" />  - probability of occurrence of i-th component of x observation, provided that it belongs to class k.

The second value is unknown, but based on assumption about training set - probability can be estimated using probability density for normal distribution:

><img src="http://latex.codecogs.com/gif.latex?p%28x_i%3Dl%7Ck%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%5Csigma%5E2_k%7D%7De%5E%7B-%5Cfrac%7B%28v-%5Cmu_k%29%5E2%7D%7B2%5Csigma%5E2_k%7D%7D" />
 where: 
 * <img src="http://latex.codecogs.com/gif.latex?l" /> is value of i-th component of <img src="http://latex.codecogs.com/gif.latex?x" /> , 
 * <img src="http://latex.codecogs.com/gif.latex?%5Csigma%5E2_k" /> is variance for k class, based on training set
 * <img src="http://latex.codecogs.com/gif.latex?%5Cmu_k" /> is mean for k class, based on training set

To construct a classifier from the probability, is used created function `naiveBayes()`. Parameters for this functions are: _training set_ data and _test set_ data. It calculate probabilities based on above definitions and predict class:

> <img src="http://latex.codecogs.com/gif.latex?%5Chat%7Bc%7D%3Dargmax%7B%28%5Cpi_kp%28x%7Ck%29p%28y%7Ck%29%29%7D" />

Using function `naiveBayes()` it return:
```
print(naiveBayes(data,data))
>     Class         x         y
  0       1 -2.885888 -2.060685
  1       1 -1.005605 -0.434520
  2       1 -4.395286  1.145566
  ..    ...       ...       ...
  69      2  2.098806  1.931322
```
