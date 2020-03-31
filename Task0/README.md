# Task 0
## The content of the task
Check if, in the special case of Fisher's discrimination, it is possible to split classes projection into
the optimal direction **a**, if the observations meet the following features:
* 3 classes
* every class has 30 points from 2D normal distribution
* covariance for every class is equal to:
    ><img src="http://latex.codecogs.com/gif.latex?S%3D%5Cbegin%7Bpmatrix%7D%201%20%260%20%5C%5C%200%261%20%5Cend%7Bpmatrix%7D" />
* means in classes are:
    ><img src="http://latex.codecogs.com/gif.latex?m_1%3D%5Cbegin%7Bpmatrix%7D%20-1%5C%5C%201%20%5Cend%7Bpmatrix%7D%2C%5C%3B%20m_2%3D%5Cbegin%7Bpmatrix%7D%202%5C%5C%204%20%5Cend%7Bpmatrix%7D%2C%5C%3B%20m_3%3D%5Cbegin%7Bpmatrix%7D%20-2%5C%5C%202%20%5Cend%7Bpmatrix%7D%2C" />
    
## Solution
To perform Fisher's discriminant on two classes, was calculated:
  * calculate mean for all classes, using generated points
  * covariance matrix,
  * calculate inter-group variation matrix (**B**),
  * within-group variation matrix (**W**),
  * calculate vector **a**, which was eigenvector for maximum eigenvalue of matrix <img src="http://latex.codecogs.com/gif.latex?W%5E-%5E1B">
  * show projecting points of classes, on function of vector **a**
Next, was created plot with classes points, function for **a** and perpendicular function to **a** (which is between mean points of classes, on function for **a**)
Results of this exercise are on image below:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task0/task0.png" height="100%" width="100%">

## Conclusion
Based on the plot obtained, it can be sad that it is not possible to split classes projection
on the optimal direction of vector **a**, to optimally separate classes.
