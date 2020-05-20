# Task 7 :palm_tree:

## The content of the task
List to do:
* [x] Interpret animal characteristic data using function `heatmap()` from library _seaborn_,
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


