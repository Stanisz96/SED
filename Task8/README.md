# Task 8 :violin:

## The content of the task
List to do:
* [x] load wine data,
* [x] perform _PCA_ analysis for wine collection,
* [x] plot the cumulative standard deviation from number of principal components.
* [x] plots for 1 and 2 as well as 2 and 3 principal components.

### Load wine data
Based on the description in file _wine.names_ was created object _DataFrame_ with columns for
classes and attributes:
```
print(wineData)
>     Class Alcohol Malic acid  ...   Hue Proline
  0       1   14.23       1.71  ...  1.04    1065
  1       1    13.2       1.78  ...  1.05    1050
  2       1   13.16       2.36  ...  1.03    1185
  ..    ...     ...        ...  ...   ...     ...
  176     3   13.17       2.59  ...   0.6     840
  177     3   14.13        4.1  ...  0.61     560
```

### Perform _PCA_ analysis for wine collection
Wine data has 13 attributes, therefore it can't be projected on plot. To enable data visualization, is needed to reduce
the variables to 2 or 3. For this is used _Principal component analysis (PCA)_.
First create object `PCA()` from _sklearn.decomposition_. Then use method `fit_transform()`. It will return data with new variables.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task8/img/pca_2.png" height="75%" width="75%">

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task8/img/pca_3.png" height="75%" width="75%">

<br></br>

_PCA_ can be also performed using kernels. For this method is used class _KernelPCA_. 
Using 2 principal components other quite good option then _linear_ can be polynomial kernel.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task8/img/pca_test_poly.png" height="75%" width="75%">

From this plot it can be said that class 1 has more different values of attributes then class 2 and 3. It is easier to classify data to class 1 then to class 2 or 3.
<br></br>

Using 3 principal component with cosine kernel - it is much clearer to distinguish 3 classes:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task8/img/pca_3_cosine.png" height="75%" width="75%">


### Cumulative standard deviation from number of principal components
Using attribute _explained_variance__ from `PCA()` object can be drawn plot for cumulative standard deviation depending on number of principal components.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task8/img/pca_sum_sd.png" height="75%" width="75%">

Results shown on the plot suggest that for wine dataset can be use 2 first principal components. 



### Plots for 1 and 2 as well as 2 and 3 principal components

Plot for 1 and 2 principal components:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task8/img/pca_com_1_2.png" height="75%" width="75%">

Plot for 2 and 3 principal components:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task8/img/pca_com_2_3.png" height="75%" width="75%">

It is clear that principal components that explain the most standard deviation can present better shown dataset then for others principal components.