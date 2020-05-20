import pandas as pd
import seaborn as sb
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

# Read data from file animals.csv
dataAnimal = pd.read_csv("animals.csv")
dataAnimal.columns = ["animal","warm-blooded","can fly","vertebrate","endangered","live in groups","have hair"]
dataAnimal = dataAnimal.fillna(1.5)
dataAnimal = dataAnimal.rename(index=dataAnimal.animal)
dataAnimal = dataAnimal.loc[:,dataAnimal.columns != 'animal']
# imputer = KNNImputer(n_neighbors=2, weights="uniform")
# dataAnimal.loc[:,dataAnimal.columns != 'animal'] = imputer.fit_transform(dataAnimal.loc[:,dataAnimal.columns != 'animal'].values)


ax = sb.clustermap(dataAnimal)
ax.savefig("img/heatmap.png")
print(ax.dendrogram_row.reordered_ind)
print(ax.dendrogram_col.reordered_ind)