# Task 6 :guitar:
## The content of the task
List to do:
* [x] generate data from Gaussian Distribution - with parameters as in _Task 0_ <sup> _[[1]]_</sup>&nbsp;,
* [ ] Check SVC with linear kernel for this data,
* [ ] compare SVC with LDA.


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
### SVC with linear kernel





[1]: https://github.com/Stanisz96/SED/tree/master/Task0