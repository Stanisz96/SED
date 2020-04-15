# Task 4 :seedling:
## The content of the task
Use wine dataset from _Task 3_ to perform following points:
* load data,
* name columns, using the description,
* create full decision tree (leafs with elements of one class),
* draw full decision tree,
* check accuracy of the full tree by repeated substitution and cross-validation,
* use the _cp_ table to select the optimal tree, draw it and compare the results of its
accuracy with the full tree,
* create tree for the first: 2, 3, 4, etc. variables - determine the most optimal tree each time,
* plot the accuracy of the tree as a function of the number of used variables, as well 
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





[1]: https://www.graphviz.org/