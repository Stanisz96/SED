# Task 3
## The content of the task
Plot the effectiveness of the _knn_ classifier as a function of the number of closest neighbors
from _knn = 1_ to _knn = 21_. Do the same for TP and TN. Then draw an additional 10 points from
class 1 and 5 points from class 2, treat as a test set and repeat the plots. If possible,
average over 10 draws.

## Solution
### Generate Data
Data are generated from 2D normal distribution for two classes with properties:

><img src="http://latex.codecogs.com/gif.latex?S_1%3D%5Cbegin%7Bpmatrix%7D%204%20%262%20%5C%5C%202%20%26%204%20%5Cend%7Bpmatrix%7D%2C%5C%3B%20S_2%3D%5Cbegin%7Bpmatrix%7D%204%20%262%20%5C%5C%202%20%26%202%20%5Cend%7Bpmatrix%7D" />

><img src="http://latex.codecogs.com/gif.latex?m_1%3D%5Cbegin%7Bpmatrix%7D%20-1%5C%5C%20-1%20%5Cend%7Bpmatrix%7D%2C%5C%3B%20m_2%3D%5Cbegin%7Bpmatrix%7D%202%5C%5C%202%20%5Cend%7Bpmatrix%7D" />

><img src="http://latex.codecogs.com/gif.latex?n_1%3D30%2C%5C%3B%20n_2%3D20" />

To generate observations is used function `generateGaussData()` <sup> _[[1]]_</sup>&nbsp;.

### K-Nearest Neighbors Algorithm _(KNN)_
In k-NN classification, the output is a class membership. An object is classified by a plurality
vote of its neighbors, with the object being assigned to the class most common among its k nearest
neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned
to the class of that single nearest neighbor<sup> _[[2]]_</sup>&nbsp;.

To create object with k-NN classifier is used class from _Sklearn_ - `KNeighborsClassifier()`.


### Plot
Using classificator from _knn = 1_ to _knn = 21_ is created plot for accuracy:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task3/acc.png" height="100%" width="100%">


Based on confusion matrix plot for _true positive_ and _true negative_:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task3/tp_tn.png" height="100%" width="100%">

### Create test set
For created classificator is generated data with 10 observations from class 1 and 5 observation for class 2.
Using test data are created plots like in point above:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task3/ext_acc.png" height="100%" width="100%">
<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task3/ext_tp_tn.png" height="100%" width="100%">

## Conclusion


[1]: https://github.com/Stanisz96/SED/tree/master/Task1
[2]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm