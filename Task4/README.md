# Task 4 :seedling:
## The content of the task
Use wine dataset from _Task 3_ to perform following points:
* [x] load data,
* [x] name columns, using the description,
* [x] create full decision tree (leafs with elements of one class),
* [x] draw full decision tree,
* [x] check accuracy of the full tree by repeated substitution and cross-validation,
* [ ] use Minimal Cost-Complexity Pruning, draw it and compare the results of its
accuracy with the full tree,
* [ ] create tree for the first: 2, 3, 4, etc. variables - determine the most optimal tree each time,
* [ ] plot the accuracy of the tree as a function of the number of used variables, as well 
as differences in the size of the full and optimal tree.

## Solution
### Data
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

### Create full decision tree
To create object classifier is used function from _Sklearn_ - `DecisionTreeClassifier()`.
On created object is used method `fit()` to build decision tree classifier from the _wineData_ set.


### Draw full decision tree
To draw full decision tree is used _Graphviz_. This is open source graph visualization software<sup> _[[1]]_</sup>&nbsp;.
Using function from _Sklearn_ - create data in form customized to _Graphviz_.
```
graph_data = tree.export_graphviz(clf, out_file=None, feature_names=column_names, filled=True,
                                  rounded=True, special_characters=True, class_names=class_names)
graph = graphviz.Source(graph_data)
graph.view()
```
Result of this decision tree:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task4/FullTree.png" height="100%" width="100%">


### Check accuracy of the full decision tree by repeated substitution
Using method `predict()` on _DecisionTreeClassifier_ object with fitted _wineData_ set, get array of predicted class:
```
cls_predict = pd.DataFrame()
cls_predict["Class"] = wineData.Class.values
cls_predict["Predicted"] = clf.predict(wineData.loc[:,wineData.columns != 'Class'].values)
print(cls_predict)
>      Class  Predicted
  0        1          1
  1        1          1
  ..     ...        ...
  175      3          3
  176      3          3
  177      3          3
```
Then, using created functions `CM()` (for calculate confusion matrix) and `derivationsCM()` (to calculate derivations from a confusion matrix) - get _DataFrame_ object:
```
confusion_matrix = CM(cls_predict)
derivations_CM = derivationsCM(confusion_matrix)
print(derivations_CM)
>    Class  ACC  TP   TN  TPR  FPR
  0      1  1.0  59  119  1.0  0.0
  1      2  1.0  71  107  1.0  0.0
  2      3  1.0  48  130  1.0  0.0
```
Thus, it follows that a decision tree that is trained on the same data as it has been checked has 100% accuracy.



### Check accuracy of the full decision tree by _k-fold cross-validation_
Because of small amount of data, constant ___ is set to 5. To divide dataset is used object `KFold` from _Sklearn_ library.
Calculating means for derivations of confusion matrix get result for full decision tree:
```
print(sum(d_CM)/5)
>    Class       ACC    TP     TN       TPR       FPR
  0    1.0  0.991011  58.6  117.8  0.993220  0.010084
  1    2.0  0.982022  69.0  105.8  0.971831  0.011215
  2    3.0  0.988764  47.0  129.0  0.979167  0.007692
```
Mean accuracy for cross-validation method is 1% lower then for repeated substitution. Accuracy is close to _1.00_ and
it is possibility that classifier is overfitting the training data. Good way is to check data for 10-fold cross-validation:
```
print(sum(d_CM)/10)
>    Class       ACC    TP     TN       TPR       FPR
  0    1.0  0.992697  58.3  118.4  0.988136  0.005042
  1    2.0  0.991573  70.3  106.2  0.990141  0.007477
  2    3.0  0.996629  47.7  129.7  0.993750  0.002308
```
Accuracy is even higher then for 5-fold cross-validation. So it can be assumed that classifier is set good for this dataset.

### Minimal Cost-Complexity Pruning
To perform _Minimal Cost-Complexity Pruning_ are used methods on _DecisionTreeClassifier_ object <sup> _[[2]]_</sup>&nbsp;:
* `cost_complexity_pruning_path()` - compute the pruning path during Minimal Cost-Complexity Pruning,
* `decision_path()` - return the decision path in the tree.







[1]: https://www.graphviz.org/
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier