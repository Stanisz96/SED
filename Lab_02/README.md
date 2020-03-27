# Laboratory 2 - LDA, QDA and naive Bayes
## Exercise 1: "2D DISTRIBUTION DENSITY"
### Data generation
For generating points of every classes, is used function `multivariate_normal()`
from _NumPy_ library. Means, covariances and numbers of points for every class
in _Laboratory 2_ are shown below:

```
# Matrices COV for classes
S1 = np.array([[4, 0], [0, 4]])
S2 = np.array([[2, 0], [0, 2]])
S3 = np.array([[2, 1], [1, 2]])
S4 = np.array([[1, 0], [0, 1]])

# Mean
m1 = np.array([-3, -1])
m2 = np.array([2, 2])
m3 = np.array([-2, 3])
m4 = np.array([1, -2])

# Number of points in classes
n1 = 200
n2 = 150
n3 = 150
n4 = 100
n = n1 + n2 + n3 + n4
```
<br></br>
With this constants, can be used function for generating 2-dimension points
from normal distribution.
```
# Generate normal distribution
X1 = np.random.multivariate_normal(m1, S1, n1)
X2 = np.random.multivariate_normal(m2, S2, n2)
X3 = np.random.multivariate_normal(m3, S3, n3)
X4 = np.random.multivariate_normal(m4, S4, n4)
```
Where _X1,X2,X3,X4_ are a set of points generated for individual classes.

### Probability density function _(PDF)_
Next step is to calculate _PDF_. Function from library _SciPy_ `multivariate_normal.pdf()` 
return array of density for every point in data set, depending on given mean and covariance.
```
# Get normal distribution density
pdf_X1 = st.multivariate_normal.pdf(X1, m1, S1)
pdf_X2 = st.multivariate_normal.pdf(X2, m2, S2)
```
<br></br>
In this example plot is created for two classes.
```
plt.plot(x1, y1, ".", color="red", markersize=8)
plt.plot(x2, y2, ".", color="green", markersize=8)
```
<br></br>
Then adding contours of density to plots:
```
CS1 = plt.tricontour(x1, y1, pdf_X1, colors=["#ffe6e6", ... , "#660000"])
CS2 = plt.tricontour(x2, y2, pdf_X2, colors=["#e6ffe6", ... ,  "#001a00"])
```
Where colors are single contours for set up density levels.
<br></br>

Adding to this plot settings for visualization, result is:
<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise01.png" height="80%" width="80%" />


