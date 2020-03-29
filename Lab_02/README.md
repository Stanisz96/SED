# Laboratory 2 - LDA, QDA and naive Bayes
## Exercise 1: "2D Distribution density"
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


## Exercise 2: "LDA & QDA methods"
### Data structure
In this and next exercises - created data are reworked as `DataFrame` object in _Pandas_.
It is easier to get from one table, array of points for a specific class, or do some other manipulations with data.
Created `DataFrame` structure:
```
     Class         x         y
0        1 -3.368001  2.854359
1        1 -7.772308 -5.948453
2        1 -3.669230 -0.132535
3        1 -5.180073 -2.211185
4        1 -0.845756 -2.800099
..     ...       ...       ...
348      2  2.899116  2.975279
349      2  1.630539  2.497368
```

### Linear Discriminant Analysis
For _LDA_ is used function from _Sklearn_ `LinearDiscriminantAnalysis()`.
This function create object on which can be performed many useful methods and have attributes<sup> _[[1]]_</sup>&nbsp;.
In this case is used methods:
* `fit()` - fit _LDA_ model according to given data,
* `predict()` - return array of predicted class for points in given data

```
lda = LDA()
cls_pred = lda.fit(data[['x', 'y']].values, data['Class'].values)
              .predict(data[['x', 'y']].values)
```

### Quadratic Discriminant Analysis
For _QDA_ is used function from _Sklearn_ `QuadraticDiscriminantAnalysis()`.
This function create object on which can be performed many useful methods and have attributes<sup> _[[2]]_</sup>&nbsp;.
In this case is used methods:
* `fit()` - fit _QDA_ model according to given data,
* `predict()` - return array of predicted class for points in given data

```
qda = QDA()
cls_pred = qda.fit(data[['x', 'y']].values, data['Class'].values)
              .predict(data[['x', 'y']].values)
```

### Plot
To created plot for results of _LDA_ and _QDA_ is used writen function `plot_function()`<sup> _[[3]]_</sup>&nbsp;.
First step for this function is to create arrays with information about correct predict class for given data.
``` 
tp = (cls == cls_pred)
tp1, tp2 = tp[cls == 1], tp[cls == 2]
X1, X2 = X[cls == 1], X[cls == 2]
X1_tp, X1_fp = X1[tp1], X1[~tp1]
X2_tp, X2_fp = X2[tp2], X2[~tp2]
```
Where `tp` is _boolean_ array result for equating class values for
given points with the predicted class values for these points. `tp1` and `tp2`
 are _boolean_ arrays for specific class. `X2_tp, X2_fp` and `X1_tp, X1_fp` are
 accordingly data arrays for points good and wrong predicted.

Another stage for this function is to add points for classes:
```
# class 1: points
plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='#ff4d4d')
plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x', s=20, color='#e60000')
# class 2: points
plt.scatter(X2_tp[:, 0], X2_tp[:, 1], marker='.', color='#00ace6')
plt.scatter(X2_fp[:, 0], X2_fp[:, 1], marker='x', s=20, color='#007399')
```

To create areas for classes, function create `x,y` points and use them to create coordinate matrices.
For this is used function from _NumPy_ `meshgrid()`.
```
nx, ny = 200, 200
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
```
Next step is to get third dimension coordinate matrices with class predict probability.
```
Z = da.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
```
With this, can be filled plot with colors. Also, because this is function for two classes,
can be added contour to plot - based on probability of chosen class equal to 0.5.
```
plt.pcolormesh(xx, yy, Z, cmap='coolwarm_r',
               norm=colors.Normalize(vmax=1, vmin=0), alpha=0.1, zorder=0)
plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white', zorder=0)
```

The last step for function `plot_function()` is to add on plot means
for classes, using atrribute `means_`. 

Called function  return plot for _LDA_:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise02_lda.png" height="80%" width="80%" />

<br></br>

And for _QDA_:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise02_qda.png" height="80%" width="80%" />



## Exercise 3: "Naive Bayes"

To perform classifier _Naive Bayes_ is used function from _Sklearn_ `GaussianNB()`.
On created object with this function is used method `partial_fit()` which return incremental fit on a batch of samples<sup> _[[4]]_</sup>&nbsp;.
```
gnb = GaussianNB()
cls_pred = gnb.partial_fit(data[['x', 'y']].values, data['Class'].values,np.unique(data['Class'].values))
              .predict(data[['x', 'y']].values)
```
In next step is used function `plot_function()` and it gives:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise03.png" height="80%" width="80%" />

## Exercise 4: "Many classes"
In this exercise function `plot_function()` is updated to `plot_function_N()` and can create 
plots for up to 6 classes. Difference in this functions are:
* add `cls_col, cls_col_dark, cls_col_map` - to generate colors for every class
* is used function `ListedColormap()` to generate ColorMap, depend of classes number
* creating variables are depend on `cls_n` (classes number)
* fill plot with colors depend now on predicted classes, not on probability of classes
* is removed means for classes

Results of this function are:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise04_lda_3.png" height="80%" width="80%" />

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise04_qda_3.png" height="80%" width="80%" />

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise04_lda_4.png" height="80%" width="80%" />

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Lab_02/Exercise04_qda_4.png" height="80%" width="80%" />




[1]: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
[3]: https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#sphx-glr-auto-examples-classification-plot-lda-qda-py
[4]: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html


