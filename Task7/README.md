# Task 7 :palm_tree:

## The content of the task
List to do:
* [x] Interpret animal characteristic data using function `clastermap()` from library _seaborn_,
* [ ] Using _K-means_ method - recognize clasters in the _Iris_ dataset. Test all combinations for attributes.

## Interpret animal characteristic data using function `heatmap()` from library _seaborn_
Read data from _animal.csv_ file using _Pandas_ library. Loaded data are presented as below:
```
     Unnamed: 0  war  fly  ver  end  gro  hai
   0        ant    1    1    1  1.0  2.0    1
   1        bee    1    2    1  1.0  2.0    2
 ...        ...  ...  ...  ...  ...  ...  ...
  17        sal    1    1    2  1.0  NaN    1
  18        spi    1    1    1  NaN  1.0    2
  19        wha    2    1    2  2.0  2.0    1
```
Then add columns names to data, remove value _NaN_ with _1.5_ and change names of indexes to names of animals.

```
     warm-blooded  can fly  vertebrate  endangered  live in groups  have hair
ant             1        1           1         1.0             2.0          1
bee             1        2           1         1.0             2.0          2
...    ...           ...           ...         ...             ...        ...
sal             1        1           2         1.0             1.5          1
spi             1        1           1         1.5             1.0          2
wha             2        1           2         2.0             2.0          1
```
Using `clustermap()` function from _seaborn_ library - create hierarchically-clustered heatmap:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task7/img/heatmap.png" height="75%" width="75%">

<br></br>
From this dendrogram can be read that w"arm-blooded" and "vertebrate" are the closest in terms of the featuers considered.
The least similar attributes are: "can fly" and "vertebrate"
Also it can be said that, the most similar animals are:
* rabbit (rab) and lion (lio)
* spider (spi) and (cpl)
* (sal) and (her)
* ant (ant) and (lob)

The least similar animals are: ant and eagle.

## Using _K-means_ method - recognize clasters in the _Iris_ dataset

### K-Means method
Probably the most popular, non-hierarchical cluster analysis algorithm is
the K-means algorithm. Assignment of n objects to a given number of K clusters
is carried out independently for each K value - not based on previously determined
smaller or larger clusters [[1], 385].

Algorithm for K-Means:
* Randomly initialize _N_ objects in _K_ clusters. Where function _C<sub>K</sub><sup>(i)</sup>_ describe this distribution,
* For each group cluster K - calculate mean vector:
    <img src="https://latex.codecogs.com/gif.latex?%5Cbar%7B%5Cmathbf%7Bx%7D%7D_k%2C%5C%3A%20%5C%3A%20where%5C%3A%20k%3D1%2C2%2C...%2CK"></img>
* Regroup objects in K clusters, so that they meet the condition:
    
    <img src="https://latex.codecogs.com/gif.latex?C%5E%7B%28l%29%7D_K%28i%29%3Darg%5C%2C%20%5Cunderset%7B1%5Cleq%20k%5Cleq%20K%7D%7Bmin%7D%5C%2C%20%5Crho_2%28%5Cmathbf%7Bx_i%7D%2C%5Cmathbf%7B%5Cbar%7Bx%7D%7D_k%29"></img>
* Repeat second and third step until the assignment of objects to clusters remains unchanged.

### Iris data visualization (PCA)
Iris data has 4 attributes, therefore it can't be projected on plot. To enable data visualization, is needed to reduce
the variables to 2. For this is used _Principal component analysis (PCA)_.
First create object `PCA()` from _sklearn.decomposition_. Then use method `fit_transform()`. It will return data with new variables.

One the plot below is Iris dataset - transformed to 2 variables and divided into original classes assigned to data.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task7/img/pca_class.png" height="75%" width="75%">

Data are divided in to three classes. Class 1 and 2 are quite close and it is probably possible to divide dataset to more then 3 classes. It will be presented later.


### K-Means method on Iris dataset
Using `KMeans()` object from _sklearn_ library, with parameter _n_components = 2_ is performed KMeans algorithm.
To visualize data is used as earlier - `PCA()`. 

On the plot below it can be seen representation for 3 clusters assigned using KMeans method for Iris dataset.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task7/img/pca_kmeans_3.png" height="75%" width="75%">

There is slight difference of data assigned to cluster 1 and 2 between KMeans method and original data.


### Search for optimized number of clusters (K)
To find the optimal K was used method based on maximization _gap statistic_.

First - to perform this method - create dataset with the same size as Iris dataset, using uniform distribution - where _min_ and _max_
values for attributes are based on _min_ and _max_ values in Iris dataset. Perform `KMeans()` method on this data for number of cluster from 1 to 10.
From every object _KMeans()_ get value defined as sum of squared distances of samples to their closest cluster center. Then, use logarithm on this value.
Perform this 20 times and then calculate mean for sum of squared distances and calculate standard deviation. Created variables are represented as [[1], 389]:

<img src="https://latex.codecogs.com/gif.latex?W%5E%7B*%7D_%7BK%7D%5C%3A%5C%3A%20and%5C%3A%5C%3A%20s%5E%7B*%7D_%7BK%7D"></img>

Calculated values of sums of squared distances of samples to their closest cluster center for different number of cluster are shown on plot below.
This values are not shown in logarithmic scale.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task7/img/w_value.png" height="75%" width="75%">

<br></br>
_Gap statistic_ is defined as:

<img src="https://latex.codecogs.com/gif.latex?Gap%28K%29%3DW%5E%7B*%7D_%7BK%7D-W_%7BK%7D"></img>

Using calculated values - results are:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task7/img/gap_statistic.png" height="75%" width="75%">

The optimal number of clusters using this method is first K which meets the condition [[2], 2]:

<img src="https://latex.codecogs.com/gif.latex?Gap%28K%29%5Cgeq%20Gap%28k&plus;1%29-s_%7Bk&plus;1%7D%2C%5C%3A%20where%5C%2C%20s_%7Bk&plus;1%7D%3D%5Csqrt%7B1&plus;1/N%7D%5Ccdot%20s%5E*_%7Bk&plus;1%7D"></img>

Calculated optimal number of clusters is 4. It can be seen on visualization using _PCA_:

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task7/img/pca_kmeans_4.png" height="75%" width="75%">


### Optimal number of clusters for every combinations of attributes in Iris dataset
To calculate optimal number of clusters for every combinations of attributes in Iris dataset is used method in point below.
To extract combinations of attributes from array is used function `combinations()` from library_itertools_.

<img src="https://raw.githubusercontent.com/Stanisz96/SED/master/Task7/img/combination_iris.png" height="75%" width="75%">

On the plot above:
* horizontal axis means: number of attributes used multiply by 10 and added combination number,
* vertical axis means: optimal number of cluster, calculated using _gap statistic_ method,
* results that mean optimal number of clusters for 2 attributes is 3, for 3 attributes is 4 and for 4 attributes is 4.

Using few attributes, is harder to divide dataset to the same number of clusters then using more attributes. However, too much attributes may not help to optimally divide dataset into more numbers of clusters.


[1]: https://www.researchgate.net/publication/292148701_Systemy_uczace_sie_Rozpoznawanie_wzorcow_analiza_skupien_i_redukcja_wymiarowosci
[2]: https://core.ac.uk/download/pdf/12172514.pdf