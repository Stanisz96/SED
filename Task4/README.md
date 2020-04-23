# Task 4 :seedling:
## The content of the task
Use wine dataset from _Task 3_ to perform following points:
* [x] load data,
* [x] name columns, using the description,
* [x] create full decision tree (leafs with elements of one class),
* [x] draw full decision tree,
* [x] check accuracy of the full tree by repeated substitution and cross-validation,
* [x] use Minimal Cost-Complexity Pruning, draw it and compare the results of its
accuracy with the full tree,
* [x] create tree for the first: 2, 3, 4, etc. variables - determine the most optimal tree each time,
* [x] plot the accuracy of the tree as a function of the number of used variables, as well 
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
print(train_mean_results)
>   Class  ACC    TP     TN  TPR  FPR
 0    1.0  1.0  47.2   95.2  1.0  0.0
 1    2.0  1.0  56.8   85.6  1.0  0.0
 2    3.0  1.0  38.4  104.0  1.0  0.0
print(test_mean_results)
>   Class       ACC    TP    TN       TPR       FPR
 0    1.0  0.955238  11.4  22.6  0.970330  0.046791
 1    2.0  0.893810  11.6  20.2  0.824603  0.056746
 2    3.0  0.938571   8.8  24.6  0.917778  0.053978
```
Test set mean accuracy for cross-validation method is 1% lower then for repeated substitution. Accuracy is close to _1.00_ and
it is possibility that classifier is overfitting the training data. Good way is to check data for 10-fold cross-validation:
```
print(train_mean_results)
>   Class  ACC    TP     TN  TPR  FPR
 0    1.0  1.0  53.1  107.1  1.0  0.0
 1    2.0  1.0  63.9   96.3  1.0  0.0
 2    3.0  1.0  43.2  117.0  1.0  0.0

print(test_mean_results)
>   Class       ACC   TP    TN       TPR       FPR
 0    1.0  0.943791  5.4  11.4  0.924123  0.038810
 1    2.0  0.887255  6.1   9.7  0.870476  0.094128
 2    3.0  0.943464  4.3  12.5  0.905000  0.042729
```
Accuracy for test set data is similar for 5-fold cross-validation. So it can be assumed that classifier is  not overfitted for this dataset. Accuracy for test sets and training sets doesn't have high difference.

### Minimal Cost-Complexity Pruning
Use _5-fold cross-validation_ to create 5 _wineData test and train sets_. Create `DecisionTreeClassifier()` object and use function
`cost_complexity_pruning_path()` for _k-wineData_Train_. This function return ccp_alphas (cost complexity pruning alphas).
Next for every _ccp_alpha_ fit decision tree - using _k-wineData_Train_ set. On plot below is mean _ccp_alpha_ for k-testset in 5-fold cross-validation.
Mean _ccp_alpha_ is calculated from maximum accuracy for k-testset.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task4/5fold_ccp_alphas.png" height="75%" width="75%">

Calculated mean ccp_alpha from above plot is: 0.02324. Using this value to create decision tree - return:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task4/CcpAlphaTree.png" height="75%" width="75%">


### Create decision tree for the first: 2, 3, 4, etc. variables
Using method of creating most optimal decision tree from point above - can create table with properties of decision
tree with different number of variables. Created _DataFrame_ object `var_decision_tree_prop` contains table for alphas, number of leaves and depth - depend on number of variables in dataset:

```
print(var_decision_tree_prop)
>   Variables  Ccp_alphas  Leaves  Depth  Mean accuracy
0           2      0.0425       3      2         0.8146
1           3      0.0314       4      3         0.8371
2           4      0.0376       5      3         0.8708
3           5      0.0284       5      3         0.8708
4           6      0.0208       7      3         0.9270
5           7      0.0273       5      3         0.9719
6           8      0.0168       5      3         0.9719
7           9      0.0262       5      3         0.9719
8          10      0.0145       7      4         0.9944
9          11      0.0250       4      3         0.9607
10         12      0.0329       4      3         0.9607
11         13      0.0167       8      3         0.9775
```

### Plot the accuracy of the tree as a function of the number of used variables, as well as differences in the size of the full and optimal tree

Using previous _DataFrame_ `var_decision_tree_prop` can create plots as below:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task4/accuracy.png" height="75%" width="75%">

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task4/depth.png" height="75%" width="75%">

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task4/leaves.png" height="75%" width="75%">



[1]: https://www.graphviz.org/
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier