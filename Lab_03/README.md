# Laboratory 3 - Rating of classifiers
## Exercise 1: "Confusion matrix"
Confusion matrix contains boolean values, which is used to evaluate accuracy
 of classifiers.

><img src="http://latex.codecogs.com/gif.latex?CM%3D%5Cbegin%7Bpmatrix%7D%20TN%26FP%20%5C%5C%20FN%26TP%20%5Cend%7Bpmatrix%7D" />

<br></br>
Where individual values means:
* `TN` - _true negative_ (data: negative, prediction: negative)
* `FP` - _false positive_ (data: negative, prediction: positive)
* `FN` - _false negative_ (data: positive, prediction: negative)
* `TP` - _true positive_ (data: negative, prediction: negative)

Accuracy for classifiers is defined:
```markdown
ACC = r'$x^2$'
```