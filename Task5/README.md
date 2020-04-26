# Task 5 :chicken:
## The content of the task
Use _Iris_ dataset and perform following points:
* [x] load _Iris_ dataset,
* [x] sample max 80% data to train classifier,
* [x] use _Bagging_ classifier with _LDA_ estimator for number of estimators: 1-50,
* [x] use _Bagging_ classifier with _Decision Tree_ estimator for number of estimators: 1-50.



### _Bagging_ classifier with _LDA_ estimator
Using obejct _BaggingClassifier_ from _Sklearn_ library can be perform _Bagging classifier_ with any estimators.
For this point is used _LDA_ estimator. Created function `get_clf_predicted()` with arguments for object of estimator and number of estimators - return _DataFrame_ with columns for class and predicted class:
```
print(get_clf_predict(LDA(), 5))
>      Class  Predicted
  0        0          0
  1        0          0
  2        0          0
  ..     ...        ...
  148      2          2
  149      2          2
```

This function create object _BaggingClassifier_ and set parameters such as: estimator, number of estimators, part of the sample data.
Then, on created object is used method `fit()` to build a bagging ensemble of estimators from the training. Next task is to predict classes of given data. Last step for this function is to create and return _DataFrame_ object.

Using loop, was created object _DataFrame_ with number of wrong predicted labels in % and number of estimators:
```
print(err_LDA_table)
>    Error [%]  Number of estimators
 0        0.00                   1.0
 1        2.00                   2.0
 2        2.00                   3.0
 ..        ...                   ...
 48       0.00                  49.0
 49       0.67                  50.0
```
Created plot with this data:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task5/LDA.png" height="75%" width="75%">


### _Bagging_ classifier with _Decision Tree_ estimator
To create _Bagging classifier_ with _Decision tree_ estimator - can be followed the instructions as in previous point.

Created plot for Decision Tree:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task5/Tree.png" height="75%" width="75%">


### Conclusion 
As seen in the charts - _Bagging classifier_ is not good chooise to improve quality for _LDA_ estimator. However _Bagging classifier_ returns better results for _Decision Tree_ estimator.