# Laboratory 3 - Rating of classifiers
## Exercise 1: "Confusion matrix"
### Description
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

### Data
For this exercises' data, is used normal distribution 2d. Created function `generateGaussData()`
is for generate 2D data from normal distribution and return as _DataFrame_ object. Essential fragment of
 code :

```
...
dataFrame = pd.DataFrame()
for i in range(cls_n):
   X[i] = np.random.multivariate_normal(m[i], S[i], n[i])
   x, y = zip(*X[i])
   temp = pd.DataFrame({"Class": [i+1 for x in range(np.size(X[i], 0))], "x": x, "y": y})
   dataFrame = dataFrame.append(temp, ignore_index=True)
...
```
_Example 1.1. Generate data for 2 classes using 2D normal distribution_
```
data = generateGaussData(S, m, [100,300],2)
print(data)
>      Class         x         y
  0        1  0.978248 -1.371435
  1        1 -3.292490 -0.694666
  2        1 -2.621119 -3.242361
  3        1 -1.982690 -0.663168
  4        1 -3.862287 -1.605063
  ..     ...       ...       ...
  395      2  0.024733  0.767095
```
Where:
 * `S` is array with two covariance matrices
 * `m` is array with two vector means for classes
 * `[100,300]` are numbers of points for first and second class
 * `2` is number of classes
 
### Training dataset
Next step is to train classificators on data, using _LDA, QDA, naiveBayes_. For this task
will be used function `train()` to perform this methods on data and array:

```
def train(data):
    qda = QDA()
    lda = LDA()
    gnb = GaussianNB()

    cls_pred_qda = qda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)
    cls_pred_lda = lda.fit(data[['x', 'y']].values, data['Class'].values).predict(data[['x', 'y']].values)
    cls_pred_gnb = gnb.partial_fit(data[['x', 'y']].values, data['Class'].values, np.unique(data['Class'].values)).predict(data[['x', 'y']].values)

    data["qda_cls"] = cls_pred_qda
    data["lda_cls"] = cls_pred_lda
    data["gnb_cls"] = cls_pred_gnb

    return data

print(train(data))
> [QuadraticDiscriminantAnalysis(...),
      LinearDiscriminantAnalysis(...),
                      GaussianNB(...)]

```
 
 ### Confusion Matrix
 
Next step is to determine confusion matrix. Created function `CM()` return array of
confusion matrices for methods `QDA`, `LDA` and `naive Bayes`. In `CM` - columns are predicted classes
and rows are defined classes.

_Example 1.2.Return of function `CM()` for 2 classes_
```
print(CM(data))
> (array([[67., 13.],
          [14., 46.]]),
   array([[68., 12.],
          [17., 43.]]),
   array([[65., 15.],
          [13., 47.]]))
```

When `CM` matrices are ready, it is time to calculate _Accuracy (ACC),
Sensitivity (TPR)_ and _Information retrieval (FPR)_. For this operation
will be used function `derivationsCM()`.

_Example 1.3. function `derivationsCM()` for QDA method._
```
print(derivationsCM(data_CM[0]))
>   Class       ACC  TP  TN       TPR       FPR
 0      1  0.857143  67  53  0.837500  0.116667
 1      2  0.857143  53  67  0.883333  0.162500
```

The last step to get derivations of confuse matrix is to create _DataFrame_ object with all needed properties.
For this is created function `getCMResultsForCls()`. This function do operations of data and call `derivationsCM()` function:
```
def getCMResultsForCls(data_CM):
    result_CM = [pd.DataFrame({"Class": [],"ACC": [],"TP": [],"TN": [],"TPR": [],"FPR": []}) for x in range(np.size(data_CM,1))]

    for i in range(np.size(data_CM,1)):
        for x in range(np.size(data_CM,0)):
            temp = derivationsCM(data_CM[x])
            temp = temp.loc[temp.Class == i+1]
            result_CM[i] = result_CM[i].append(temp, ignore_index=True)
        result_CM[i] = result_CM[i].rename(index={0:'QDA',1:'LDA',2:'GNB'})
        result_CM[i] = result_CM[i].astype({"TP": 'int64',"TN": 'int64'})
        result_CM[i] = result_CM[i].drop(['Class'],axis=1)
    return result_CM
```
 Passing argument `data_CM` (Three element array with CM for QDA, LDA, GNB) will return for 2 classes:
```
print(getCMResultsForCls(data_CM))
> [          ACC  TP  TN     TPR       FPR
   QDA  0.814286  66  48  0.8250  0.200000
   LDA  0.800000  67  45  0.8375  0.250000
   GNB  0.792857  67  44  0.8375  0.266667,
             ACC  TP  TN       TPR     FPR
   QDA  0.814286  48  66  0.800000  0.1750
   LDA  0.800000  45  67  0.750000  0.1625
   GNB  0.792857  44  67  0.733333  0.1625]
```
Where first element of this array is for class 1 and second is for class 2.

### Testing set
 The winner cannot be determined from the above-mentioned values. Accuracy for every method is almost the same.
 It is because CM prediction of classes was done base on default data, created with defined classes.
 To solve this problem, prediction has to be done on new data:
 ```
# Create new data testing set (TS)
dataTS = generateGaussData(S, m, [30, 30], 2)

## Add predicted classes to data
dataTS["qda_cls"] = TD[0].predict(dataTS[['x', 'y']].values)
dataTS["lda_cls"] = TD[1].predict(dataTS[['x', 'y']].values)
dataTS["gnb_cls"] = TD[2].predict(dataTS[['x', 'y']].values)

# CM
dataTS_CM = CM(dataTS)
```
Results:
```
[           ACC  TP  TN       TPR       FPR
  QDA  0.900000  29  25  0.966667  0.166667
  LDA  0.866667  28  24  0.933333  0.200000
  GNB  0.883333  29  24  0.966667  0.200000,
            ACC  TP  TN       TPR       FPR
  QDA  0.900000  25  29  0.833333  0.033333
  LDA  0.866667  24  28  0.800000  0.066667
  GNB  0.883333  24  29  0.800000  0.033333]
```
From the above data it can be seen that accuracy has increased in this case.

### Example for 3 classes
Doing the same operations, but on data with 3 classes received for training set:
```
[              ACC  TP   TN     TPR       FPR
   QDA  0.838095  61  115  0.762500  0.115385
   LDA  0.842857  60  117  0.750000  0.100000
   GNB  0.842857  61  116  0.762500  0.107692,
             ACC  TP   TN       TPR       FPR
   QDA  0.861905  46  135  0.766667  0.100000
   LDA  0.857143  46  134  0.766667  0.106667
   GNB  0.857143  49  131  0.816667  0.126667,
             ACC  TP   TN       TPR       FPR
   QDA  0.947619  66  133  0.942857  0.050000
   LDA  0.947619  67  132  0.957143  0.057143
   GNB  0.938095  62  135  0.885714  0.035714]
```
And for testing set:
```
[            ACC  TP  TN       TPR       FPR
   QDA  0.866667  27  51  0.900000  0.150000
   LDA  0.866667  28  50  0.933333  0.166667
   GNB  0.888889  29  51  0.966667  0.150000,
             ACC  TP  TN       TPR       FPR
   QDA  0.900000  23  58  0.766667  0.033333
   LDA  0.888889  21  59  0.700000  0.016667
   GNB  0.888889  22  58  0.733333  0.033333, 
             ACC  TP  TN       TPR       FPR
   QDA  0.966667  28  59  0.933333  0.016667
   LDA  0.977778  29  59  0.966667  0.016667
   GNB  0.977778  28  60  0.933333  0.000000]
```
In this situation, also accuracy get better.

## Exercise 2: "ROC curve"

[1]: https://en.wikipedia.org/wiki/Accuracy_and_precision
[2]: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
[3]: https://en.wikipedia.org/wiki/Information_retrieval#Fall-out