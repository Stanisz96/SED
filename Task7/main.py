import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import seaborn as sb
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from itertools import combinations

# Read data from file animals.csv
dataAnimal = pd.read_csv("animals.csv")
dataAnimal.columns = ["animal","warm-blooded","can fly","vertebrate","endangered","live in groups","have hair"]
dataAnimal = dataAnimal.fillna(1.5)
dataAnimal = dataAnimal.rename(index=dataAnimal.animal)
dataAnimal = dataAnimal.loc[:,dataAnimal.columns != 'animal']

ax = sb.clustermap(dataAnimal)
ax.savefig("img/heatmap.png")


# K-Means
dataAnimal = pd.read_csv("animals.csv")
dataAnimal.columns = ["animal","warm-blooded","can fly","vertebrate","endangered","live in groups","have hair"]
dataAnimal = dataAnimal.fillna(1.5)

k_means = KMeans()
## Elbow Method
elbow_value_table = []
elbow_score_table = []
n_table = []
n_elbow_poin = 0
n_combinations = 0

# print(dataAnimal.columns.values[dataAnimal.columns != 'animal'])
for i in range(2,7):
    com_attributes = list(combinations(dataAnimal.columns.values[dataAnimal.columns != 'animal'],i))
    for attributes in com_attributes:
        n_combinations += 1
        elbowMethod = KElbowVisualizer(k_means,k=(1,2*i+1))
        # print(dataAnimal[np.array(attributes)])
        elbowMethod.fit(dataAnimal[np.array(attributes)].values)
        # print(dataAnimal.loc[:,dataAnimal.columns != 'animal'])
        # elbowMethod.fit(dataAnimal.loc[:,dataAnimal.columns != 'animal'])

        if elbowMethod.elbow_score_ != 0:
            n_elbow_poin += 1
            elbow_value_table.append(elbowMethod.elbow_value_)
            elbow_score_table.append(elbowMethod.elbow_score_)
            n_table.append(i)
        plt.close()

print("combinations "+str(n_combinations))
print("finded elbow "+str(n_elbow_poin))

plt.hist2d(x=n_table, y=elbow_value_table,bins=6)
# plt.plot(n_table,elbow_score_table, drawstyle="steps-post")
plt.title("lol")
plt.xlabel("number")
plt.ylabel("value and score")
plt.legend(["Optimal K"])
plt.show()
# elbowMethod.finalize()
#
# plt.savefig('img/elbowtest.png',dpi=150)



## Fit kmeans
k_means.fit(dataAnimal.loc[:,dataAnimal.columns != 'animal'])
pred_k_means = k_means.predict(dataAnimal.loc[:,dataAnimal.columns != 'animal'])
clusters_center = k_means.cluster_centers_


# PCA
reduced_dataAnimal = PCA(n_components=2).fit_transform(dataAnimal.loc[:,dataAnimal.columns != 'animal'])
dataAnimal_PCA = pd.DataFrame(reduced_dataAnimal, columns=['PCA1','PCA2'])

# Join PCA points and pred_k_means to dataAnimal
dataAnimal["PCA1"] = dataAnimal_PCA.PCA1
dataAnimal["PCA2"] = dataAnimal_PCA.PCA2
dataAnimal["predicted"] = pred_k_means


sb.lmplot(data=dataAnimal.loc[:,dataAnimal.columns != 'animal'],x='PCA1', y='PCA2', fit_reg=False, hue = 'predicted', palette = ['#eb6c6a','#ebe76a', '#6aeb6c','#6aebeb', '#6c6aeb','#eb6acf'])
# plt.scatter(clusters_center[:, 0], clusters_center[:, 1], c='black', s=100, alpha=0.5)
plt.savefig("img/test.png",dpi=150)
plt.close()



# Using K-means method - recognize clasters in the Iris dataset. Test all combinations for attributes.

iris = load_iris(return_X_y=True)
features_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
class_names = ['setosa', 'versicolor', 'virginica']
irisData = pd.DataFrame({'Class': iris[1], features_names[0]: iris[0][:,0], features_names[1]: iris[0][:,1],
                         features_names[2]: iris[0][:,2], features_names[3]: iris[0][:,3]})



## K-means

# k_means = KMeans(n_clusters=3)
# k_means.fit(irisData.loc[:,irisData.columns != 'Class'])
# pred_k_means = k_means.predict(irisData.loc[:,irisData.columns != 'Class'])
# clusters_center = k_means.cluster_centers_


# Use KMeans method for (1,10) clusters
K = 10
N = 20
W_table = []
W_uniform_table = []
k_table = []

for n in range(N):
    # generate uniform distribution data
    features = []
    temp_W_uniform = []
    for i in range(len(features_names)):
        temp = np.random.random_integers(np.amin(irisData.iloc[:, i + 1])*10, np.amax(irisData.iloc[:, i + 1])*10, irisData.index.stop)
        features.append(temp/10)

    dataUniform = pd.DataFrame({features_names[0]: features[0], features_names[1]: features[1],
                                features_names[2]: features[2], features_names[3]: features[3]})
    for i in range(1,11):
        k_means_uniform = KMeans(n_clusters=i, init="random")
        k_means_uniform.fit(dataUniform)
        pred_k_means_uniform = k_means_uniform.predict(dataUniform)
        temp_W_uniform.append(k_means_uniform.inertia_)
    W_uniform_table.append(np.log(temp_W_uniform))

W_uniform_table = np.array(W_uniform_table)
W_uniform_table_mean = []
std_uniform_table = []
for i in range(K):
   std_uniform_table.append(2*np.std(W_uniform_table[:,i]))
   W_uniform_table_mean.append(np.sum(W_uniform_table[:,i])/N)
print(std_uniform_table)

for i in range(1,11):
    k_means = KMeans(n_clusters=i, init="random")
    k_means.fit(irisData.loc[:, irisData.columns != 'Class'])
    pred_k_means = k_means.predict(irisData.loc[:, irisData.columns != 'Class'])
    W_table.append(np.log(k_means.inertia_))
    k_table.append(i)

plt.plot(k_table,np.exp(W_table), linestyle='--', lw=1, marker='.', ms=12)
plt.plot(k_table,np.exp(W_uniform_table_mean), linestyle='--',lw=1, marker='.', ms=12)
plt.title("W value")
plt.xlabel("K")
plt.ylabel(r'$W_{K}-W_{K}$')
plt.legend(["data from Iris dataset","data from Uniform dataset"])
plt.grid(True)
plt.savefig("img/w_value.png",dpi=150)
plt.close()

## Gap statistic
W_table = np.array(W_table)
W_uniform_table_mean = np.array(W_uniform_table_mean)

gap = W_uniform_table_mean - W_table
optimal_k = 0
for k in range(K):
    condition = gap[k+1] - std_uniform_table[k+1] * np.sqrt(1+1/N)
    if gap[k] >= condition:
        optimal_k = k+1
        break

print(optimal_k)

plt.plot(k_table,gap,linestyle='--', lw=0.8, color="blue")
plt.errorbar(k_table,gap,yerr=std_uniform_table, fmt='o', color="blue")
plt.title("Gap Statistic")
plt.xlabel("K")
plt.ylabel(r'$W^{*}_{K}-W_{K}$')
# plt.legend([""])
plt.grid(True)
plt.savefig("img/gap_statistic.png",dpi=150)
plt.close()


# PCA
reduced_dataAnimal = PCA(n_components=2).fit_transform(irisData.loc[:,irisData.columns != 'Class'])
dataAnimal_PCA = pd.DataFrame(reduced_dataAnimal, columns=['PCA1','PCA2'])

# KMeans for optimal K
k_means = KMeans(n_clusters=optimal_k, init="random")
k_means.fit(irisData.loc[:, irisData.columns != 'Class'])
pred_k_means = k_means.predict(irisData.loc[:, irisData.columns != 'Class'])

# Join PCA points and pred_k_means to dataAnimal
irisData["PCA1"] = dataAnimal_PCA.PCA1
irisData["PCA2"] = dataAnimal_PCA.PCA2
irisData["clusters"] = pred_k_means


sb.lmplot(data=irisData.loc[:,irisData.columns != 'Class'],x='PCA1', y='PCA2', fit_reg=False, hue = 'clusters', palette = ['#eb6c6a','#ebe76a', '#6aeb6c','#6aebeb', '#6c6aeb','#eb6acf'])
# plt.scatter(clusters_center[:, 0], clusters_center[:, 1], c='black', s=100, alpha=0.5)
plt.savefig("img/pca_kmeans_"+str(optimal_k)+".png",dpi=150)
plt.close()



## combinations for iri

N = 20
K = 10
global_optimal_k = []
table = []
for i in range(2,5):
    print(i)
    com_attributes = list(combinations(irisData.columns.values[irisData.columns != 'Class'],i))
    n_combinations = 0
    for attributes in com_attributes:
        W_table = []
        W_uniform_table = []
        k_table = []
        n_combinations += 1


        for n in range(N):
            # generate uniform distribution data
            features = []
            temp_W_uniform = []
            for j in range(len(features_names)):
                temp = np.random.random_integers(np.amin(irisData.iloc[:, j + 1])*10, np.amax(irisData.iloc[:, j + 1])*10, irisData.index.stop)
                features.append(temp/10)

            dataUniform = pd.DataFrame({features_names[0]: features[0], features_names[1]: features[1],
                                        features_names[2]: features[2], features_names[3]: features[3]})
            for j in range(1,11):
                k_means_uniform = KMeans(n_clusters=j, init="random")
                k_means_uniform.fit(dataUniform[np.array(attributes)].values)
                pred_k_means_uniform = k_means_uniform.predict(dataUniform[np.array(attributes)].values)
                temp_W_uniform.append(k_means_uniform.inertia_)
            W_uniform_table.append(np.log(temp_W_uniform))

        W_uniform_table = np.array(W_uniform_table)
        W_uniform_table_mean = []
        std_uniform_table = []
        for j in range(K):
           std_uniform_table.append(2*np.std(W_uniform_table[:,j]))
           W_uniform_table_mean.append(np.sum(W_uniform_table[:,j])/N)

        for j in range(1,11):
            k_means = KMeans(n_clusters=j, init="random")
            k_means.fit(irisData[np.array(attributes)].values)
            pred_k_means = k_means.predict(irisData[np.array(attributes)].values)
            W_table.append(np.log(k_means.inertia_))
            k_table.append(i)

        ## Gap statistic
        W_table = np.array(W_table)
        W_uniform_table_mean = np.array(W_uniform_table_mean)

        gap = W_uniform_table_mean - W_table
        optimal_k = 0
        for k in range(K):
            condition = gap[k+1] - std_uniform_table[k+1] * np.sqrt(1+1/N)
            if gap[k] >= condition:
                optimal_k = k+1
                global_optimal_k.append(optimal_k)
                table.append(i * 10 + n_combinations)
                break


plt.plot(table,global_optimal_k, marker='o', drawstyle="steps-post")
plt.title("Optimal K for attributes combinations")
plt.xlabel(r'$n_{attr}*10+c$')
plt.ylabel('Optimal K')
plt.grid(True)
plt.savefig("img/combination_iris.png",dpi=150)
plt.close()