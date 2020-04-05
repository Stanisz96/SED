# Task 2
## The content of the task
Data for the task are in the file _wine.data_. The description is in file _wine.names_.
Things to do:
* load data,
* name the columns, using the description,
* check confusion matrix for _LDA, QDA_ and _NB_ on the full data set,
* limited to the 2, 5 and 10 first components and check accuracy of classifiers.
* limited to the 2 first variables, divide data set to (TD, TS, FSD) 50/25/25 and in this way choose from  _LDA, QDA_ and _NB_
* limited to the 2 first variables, perform cross-validation on the case _LDA_, compare with the previous point and compare with data re-substitution

## Solution
### Load data and name the columns
Based on the description in file _wine.names_ was created object _DataFrame_ with columns for classes and attributes:
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

### Confusion matrix for _LDA, QDA_ and _NB_ on the full data set
##### Description
Confusion matrix contains sums of common values depends on matrix columns (defined class value)
 and rows (predicted class values) for , which is used to evaluate accuracy of classifiers.
In case of 2 classes, confusion matrix can be written as:

><img src="http://latex.codecogs.com/gif.latex?CM%3D%5Cbegin%7Bpmatrix%7D%20TP%20%26FN%20%5C%5C%20FP%26%20TN%20%5Cend%7Bpmatrix%7D" />

Where individual values means:
* `TN` - _true negative_ (data: negative, prediction: negative)
* `FP` - _false positive_ (data: negative, prediction: positive)
* `FN` - _false negative_ (data: positive, prediction: negative)
* `TP` - _true positive_ (data: negative, prediction: negative)

However if number of classes is more then 2, confusion matrix is _N x N_ dimensional:
><img src="http://latex.codecogs.com/gif.latex?CM%3D%5Cbegin%7Bpmatrix%7D%20c_1_1%20%26%5Ccdots%20%26c_1_n%20%5C%5C%20%5Cvdots%20%26%20%5Cddots%20%26%5Cvdots%20%5C%5C%20c_n_1%26%20%5Ccdots%20%26c_n_n%20%5Cend%7Bpmatrix%7D" />

Also values that are defined above, are equal:

><img src="http://latex.codecogs.com/gif.latex?TP_i%20%3D%20c_i_i" />
><br />
><img src="http://latex.codecogs.com/gif.latex?FP_i%20%3D%20%5Csum_%7Bl%3D1%7D%5E%7Bn%7Dc_l_i%20-%20TP_i" />
><br />
><img src="http://latex.codecogs.com/gif.latex?FN_i%20%3D%20%5Csum_%7Bl%3D1%7D%5E%7Bn%7Dc_i_l%20-%20TP_i" />
><br />
><img src="http://latex.codecogs.com/gif.latex?TN_i%20%3D%20%5Csum_%7Bl%3D1%7D%5E%7Bn%7D%5Csum_%7Bk%3D1%7D%5E%7Bn%7Dc_l_k%20-%20TP_i-%20FP_i-%20FN_i" />


In measurement of set, accuracy refers to closeness of the measurements to a specific value<sup> _[[1]]_</sup>&nbsp;. 
Accuracy for classifiers is defined:
><img src="http://latex.codecogs.com/gif.latex?ACC%3D%5Cfrac%7BTP&plus;TN%7D%7BTP&plus;TN&plus;FP&plus;FN%7D" />
In this exercise, will also be used sensitivity which measure the proportion of actual
positives that are correctly identified as such <sup> _[[2]]_</sup>&nbsp;:

><img src="http://latex.codecogs.com/gif.latex?TPR%3D%5Cfrac%7BTP%7D%7BTP&plus;FN%7D" />

Information retrieval is the activity of obtaining information system resources that are relevant to an information
 need from a collection of those resources<sup> _[[3]]_</sup>&nbsp;:
><img src="http://latex.codecogs.com/gif.latex?FPR%3D%5Cfrac%7BFP%7D%7BFP&plus;TN%7D%3D1-TPR" />

##### Determination of confusion matrix
First, on data must be performed  _LDA, QDA_ and _NB_. For this operation is used function `train()`,
which create objects for discriminants analysis and perform `fit()` method. Function `train()`
return array of this objects. Then call method `predict()` - to get predicted classes:
```
TD = train(wineData)
wineDataPredict = pd.DataFrame()
wineDataPredict["Class"] = wineData.Class.values
wineDataPredict["lda_cls"] = TD[0].predict(wineData.loc[:,wineData.columns != 'Class'].values)
wineDataPredict["qda_cls"] = TD[1].predict(wineData.loc[:,wineData.columns != 'Class'].values)
wineDataPredict["gnb_cls"] = TD[2].predict(wineData.loc[:,wineData.columns != 'Class'].values)
print(wineDataPredict)
>    Class  lda_cls  qda_cls  gnb_cls
0        1        1        1        1
1        1        1        1        1
2        1        1        1        1
..     ...      ...      ...      ...
176      3        3        3        3
177      3        3        3        3
```
To create confusion matrix is used function `CM()`. In this function - columns are predicted classes and rows are defined classes.
```
wineDataCM = CM(wineDataPredict)
print(wineDataCM)
> (array([[59.,  0.,  0.],
          [ 0., 71.,  0.],
          [ 0.,  0., 48.]]),
   array([[59.,  0.,  0.],
          [ 1., 70.,  0.],
          [ 0.,  0., 48.]]),
   array([[58.,  1.,  0.],
          [ 0., 70.,  1.],
          [ 0.,  0., 48.]]))
```
When `CM` matrices are ready, it is time to calculate _Accuracy (ACC),
Sensitivity (TPR)_ and _Information retrieval (FPR)_. For this operation
will be used function `derivationsCM()`:
```
for i in range(np.size(wineDataCM,0)):
    print(derivationsCM(wineDataCM[i]))
>  Class       ACC  TP   TN       TPR       FPR
0      1       1.0  59  119       1.0       0.0
1      2       1.0  71  107       1.0       0.0
2      3       1.0  48  130       1.0       0.0
   Class       ACC  TP   TN       TPR       FPR
0      1  0.994382  59  118  1.000000  0.008403
1      2  0.994382  70  107  0.985915  0.000000
2      3  1.000000  48  130  1.000000  0.000000
   Class       ACC  TP   TN       TPR       FPR
0      1  0.994382  58  119  0.983051  0.000000
1      2  0.988764  70  106  0.985915  0.009346
2      3  0.994382  48  129  1.000000  0.007692
```

The last step to get derivations of confuse matrix is to create _DataFrame_ object with all needed properties.
For this is created function `getCMResultsForCls()`. This function do operations of data and call `derivationsCM()` function:
```
print(getCMResultsForCls(wineDataCM))
> [          ACC  TP   TN       TPR       FPR
   QDA  1.000000  59  119  1.000000  0.000000
   LDA  0.994382  59  118  1.000000  0.008403   <- Class 1
   GNB  0.994382  58  119  0.983051  0.000000,
             ACC  TP   TN       TPR       FPR
   QDA  1.000000  71  107  1.000000  0.000000
   LDA  0.994382  70  107  0.985915  0.000000   <- Class 2
   GNB  0.988764  70  106  0.985915  0.009346,
             ACC  TP   TN       TPR       FPR
   QDA  1.000000  48  130  1.000000  0.000000
   LDA  1.000000  48  130  1.000000  0.000000   <- Class 3
   GNB  0.994382  48  129  1.000000  0.007692]
```

### Limited to the 2, 5 and 10 first components and check accuracy of classifiers
Performing similarly operations to get confusion matrix as in the previous point, only with a limitation 
to 2, 5 and 10 first attributes:


* 2 Attributes
```
print(getCMResultsForCls(wineDataCM_N[0]))
> 
[          ACC  TP   TN       TPR       FPR
 QDA  0.887640  51  107  0.864407  0.100840
 LDA  0.949438  54  115  0.915254  0.033613
 GNB  0.994382  59  118  1.000000  0.008403,
           ACC  TP   TN       TPR       FPR
 QDA  0.887640  61   97  0.859155  0.093458
 LDA  0.915730  65   98  0.915493  0.084112
 GNB  0.988764  70  106  0.985915  0.009346,
           ACC  TP   TN       TPR       FPR
 QDA  0.842697  32  118  0.666667  0.092308
 LDA  0.887640  37  121  0.770833  0.069231
 GNB  0.994382  47  130  0.979167  0.000000]
```

* 5 Attributes
```
print(getCMResultsForCls(wineDataCM_N[1]))
> [          ACC  TP   TN       TPR       FPR
 QDA  0.898876  52  108  0.881356  0.092437
 LDA  0.949438  53  116  0.898305  0.025210
 GNB  0.994382  59  118  1.000000  0.008403,
           ACC  TP   TN       TPR       FPR
 QDA  0.887640  62   96  0.873239  0.102804
 LDA  0.926966  65  100  0.915493  0.065421
 GNB  0.994382  70  107  0.985915  0.000000,
           ACC  TP   TN       TPR       FPR
 QDA  0.842697  31  119  0.645833  0.084615
 LDA  0.898876  40  120  0.833333  0.076923
 GNB  1.000000  48  130  1.000000  0.000000]
```

* 10 Attributes
```
print(getCMResultsForCls(wineDataCM_N[2]))
> [        ACC  TP   TN       TPR       FPR
 QDA  0.893258  51  108  0.864407  0.092437
 LDA  0.932584  52  114  0.881356  0.042017
 GNB  0.971910  56  117  0.949153  0.016807,
           ACC  TP   TN       TPR       FPR
 QDA  0.887640  62   96  0.873239  0.102804
 LDA  0.893258  62   97  0.873239  0.093458
 GNB  0.960674  68  103  0.957746  0.037383,
           ACC  TP   TN       TPR       FPR
 QDA  0.837079  31  118  0.645833  0.092308
 LDA  0.882022  38  119  0.791667  0.084615
 GNB  0.988764  47  129  0.979167  0.007692]
```

### Limited to the 2 first variables, divide data set to (TRD, VD, TD) 50/25/25 and in this way choose from  _LDA, QDA_ and _NB_

First step is ti limit data to 2 first variables and divide into three groups:
* `TRD` (_training dataset_) - dataset of known data on which training is run
* `VD` (_validation dataset_) - dataset of known data on which is perform class prediction
* `TD` (_test dataset_) - dataset used to provide an unbiased evaluation of a model fit on the training dataset

Then, with given data, perform training on `TRD` and select the method for `VD` with the lowest percentage of wrong data classification.
Based on calculation:
```
LDA:  16.0%
QDA:  13.0%
NB:  16.0%
```
The lowest percentage value is for _QDA_. But when repeat choosing random data for `TRD`, `VD` and `TD`, method with
the lowest percentage has changed, so it can't be chosen base on this data.

### K-fold Cross-validation
Performed method to choose the best classificator didn't get results. Another way for this is to do k-fold cross-validation.
Data set is divided on _k_ groups. For each group - set this group as test dataset and remaining as training dataset. Perform evaluation for each test data set. Calculate probability 
of wrong classification and choose the best method. Last step is to create chosen classificator on whole dataset.

For this cross validation, _k_ is fixed to _5_. To divide dataset is used object `KFold` from _Sklearn_ library.
```
LDA:  24.0 %
QDA:  22.0 %
NB:  22.0 %
```
Similarly to previous point - k-fold cross-validation didn't get the answer for the best classificator.  Depend on random chosing data, method with the lowest error percentage change.

[1]: https://en.wikipedia.org/wiki/Accuracy_and_precision
[2]: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
[3]: https://en.wikipedia.org/wiki/Information_retrieval#Fall-out