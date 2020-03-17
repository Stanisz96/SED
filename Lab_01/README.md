# Laboratory 01 - Fisher's Discriminant
The numpy library was used for mathematical calculations on the matrices. Whereas for graph visualization - matplotlib.
## EXERCISE 1: "DWIE KLASY"
To perform Fisher's discriminant on two classes, was calculated:
  * mean for coordinates of observations for every class,
  * covariance matrix,
  * within-group variation matrix (W),
  * determinate a vector **a** - which gave information about the most optimal linear function for projecting points in classes

Next, was created plot with classes points, function for **a** and perpendicular function to **a** (which is between mean points of classes, on function for **a**)
Results of this exercise are on image below:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_01/Exercise01.png" height="60%" width="60%">

## EXERCISE 2: "DWIE KLASY, ROZKLAD GAUSSA"
This task was similar, but for created points of classes - was used normal distribution.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_01/Exercise02.png" height="60%" width="60%">

## EXERCISE 3: "DWIE KLASY, 3D"
This exercise was also similar - only had to add additional dimension to classes.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_01/Exercise03.png" height="60%" width="60%">

## EXERCISE 4: "WIELE KLAS"
The last exercise was to apply Fisher's discriminant to more then two classes. The difference was:
  * calculate mean for all classes
  * calculate inter-group variation matrix (B),
  * calculate vector **a**, which was eigenvector for maximum eigenvalue of matrix <img src="http://latex.codecogs.com/gif.latex?W%5E-%5E1B">
  * show projecting points of classes, on function of vector **a**
  
  <img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_01/Exercise04.png" height="90%" width="90%">
