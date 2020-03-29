# Laboratory 3 - Rating of classifiers
## Exercise 1: "Confusion matrix"
### Description
Confusion matrix contains _boolean_ values, which is used to evaluate accuracy
 of classifiers.

><img src="http://latex.codecogs.com/gif.latex?CM%3D%5Cbegin%7Bpmatrix%7D%20TN%26FP%20%5C%5C%20FN%26TP%20%5Cend%7Bpmatrix%7D" />

Where individual values means:
* `TN` - _true negative_ (data: negative, prediction: negative)
* `FP` - _false positive_ (data: negative, prediction: positive)
* `FN` - _false negative_ (data: positive, prediction: negative)
* `TP` - _true positive_ (data: negative, prediction: negative)

In measurement of set, accuracy refers to closeness of the measurements to a specific value<sup> _[[1]]_</sup>&nbsp;. 
Accuracy for classifiers is defined:
><img src="http://latex.codecogs.com/gif.latex?ACC%3D%5Cfrac%7BTP&plus;TN%7D%7BTP&plus;TN&plus;FP&plus;FN%7D" />
In this exercise, will also be used sensitivity which measure the proportion of actual
positives that are correctly identified as such <sup> _[[2]]_</sup>&nbsp;:

><img src="http://latex.codecogs.com/gif.latex?TPR%3D%5Cfrac%7BTP%7D%7BTP&plus;FN%7D" />

Information retrieval is the activity of obtaining information system resources that are relevant to an information
 need from a collection of those resources<sup> _[[3]]_</sup>&nbsp;:
><img src="http://latex.codecogs.com/gif.latex?FPR%3D%5Cfrac%7BFP%7D%7BFP&plus;TN%7D%3D1-TPR" />

### Data
For this exercise data, is used normal distribution 2d. Created function `generateGaussData()`
is for generate 2D data from normal distribution and return as _DataFrame_ object:
```
here function
```


[1]: https://en.wikipedia.org/wiki/Accuracy_and_precision
[2]: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
[3]: https://en.wikipedia.org/wiki/Information_retrieval#Fall-out