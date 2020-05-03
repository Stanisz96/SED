# Task 6 :guitar:
## The content of the task
List to do:
* [x] generate data from Gaussian Distribution - with parameters as in _Task 0_ <sup> _[[1]]_</sup>&nbsp;,
* [ ] Check SVC with linear kernel for this data,
* [ ] compare SVC with LDA.


## Generate data
To generate observations for classes from normal distribution is used created function
`generateGaussData()`. This function return _DataFrame_ object with normal distribution
observations - depending on parameters:
* `S` - n-element array of matrix covariances for n classes
* `m` - n-element array of matrix means for n classes
* `n` - n-element array of number of points for every class
* `cls_n` - number of classes

Based on the task conditions, was created data:
```
gaussData = generateGaussData([S1,S2,S3], [m1,m2,m3],[n1,n2,n3],3)
print(gaussData)
>    Class         x         y
 0       1 -1.155415  0.842522
 1       1 -0.499944  1.161548
 2       1 -0.292196  0.876615
 ..    ...       ...       ...
 87      3 -0.831283  2.648467
 88      3 -1.616829  3.035725
 89      3 -1.232168  3.934738
```
## SVC with linear kernel
For _Support Vector Classification_ is used object `SVC()`. To create classifier is used method `fit()` using all data as training dataset.
After this is used `predict()` method to predict classes for given data. Next, using function from _Task 1_<sup> _[[2]]_</sup>&nbsp; is created plot.
Used parameters for this _SVC_ classifier are: 
* regularization parameter: C = 1
* kernel: linear

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task6/img/svc.png" height="75%" width="75%">

<br></br>
With the same parameters but different generated data can be shown plot - using created function `plot_SVC()` - with margins:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task6/img/svc_cls.png" height="75%" width="75%">


### SVC accuracy depend on regularization parameter
To get plot for SVC accuracy depend on regularization parameter - is repeated operation as in point above (for different C).

<div style="display: inline;">
<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task6/img/acc_c.png" height="48%" width="48%"> <img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task6/img/acc_c1.png" height="48%" width="48%">
<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task6/img/acc_c2.png" height="48%" width="48%"> <img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task6/img/acc_c3.png" height="48%" width="48%">
</div>

Looking at plots with different generated Gaussian distribution data but with the same parameters - can be said that optimal value for parameter _C_ is about: _0.25-0.3_.




[1]: https://github.com/Stanisz96/SED/tree/master/Task0
[2]: https://github.com/Stanisz96/SED/tree/master/Task1