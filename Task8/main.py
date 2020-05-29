import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


file = open("wine.data", "r")
wineText = file.readlines()
wineData = pd.DataFrame(
    {"Class": [], "Alcohol": [], "Malic acid": [], "Ash": [], "Alcalinity of ash": [], "Magnesium": [],
     "Total phenols": [], "Flavanoids": [], "Nonflavanoid phenols": [], "Proanthocyanins": [],
     "Color intensity": [], "Hue": [], "OD280/OD315 of diluted wines": [], "Proline": []})
for wineTextLine in wineText:
    wVL = wineTextLine.split(",")  # array of values in wine text line [wine Values Line]
    wVL[-1] = wVL[-1].replace('\n', '')
    tempData = pd.Series(
        {"Class": wVL[0], "Alcohol": wVL[1], "Malic acid": wVL[2], "Ash": wVL[3], "Alcalinity of ash": wVL[4],
         "Magnesium": wVL[5],
         "Total phenols": wVL[6], "Flavanoids": wVL[7], "Nonflavanoid phenols": wVL[8], "Proanthocyanins": wVL[9],
         "Color intensity": wVL[10], "Hue": wVL[11], "OD280/OD315 of diluted wines": wVL[12], "Proline": wVL[13]})
    wineData = wineData.append(tempData, ignore_index=True)
wineData = wineData.astype('float')
wineData = wineData.astype({"Class": "int64", "Proline": "int64"})

options = ["linear" , "poly" , "rbf" , "sigmoid" , "cosine"]
# options = [3]

# for option in options:
#     # PCA
#     reduced_wineData = KernelPCA(n_components=3, kernel=option).fit_transform(wineData.loc[:,wineData.columns != 'Class'])
#     wineData_PCA = pd.DataFrame(reduced_wineData, columns=['PCA1','PCA2','PCA3'])
#
#     # Join PCA points and pred_k_means to dataAnimal
#     wineData["PCA1"] = wineData_PCA.PCA1
#     wineData["PCA2"] = wineData_PCA.PCA2
#     wineData["PCA3"] = wineData_PCA.PCA3
#
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(wineData_PCA.PCA1, wineData_PCA.PCA2,wineData_PCA.PCA3, c=wineData.Class)
#     ax.set_xlabel('PCA1')
#     ax.set_ylabel('PCA2')
#     ax.set_zlabel('PCA3')
#     # ax.legend()
#     # plt.show()
#     plt.savefig("img/pca_3_"+str(option)+".png",dpi=150)
#     plt.close()


columns = ['PCA'+str(i+1) for i in range(13)]

# PCA
reduced_wineData = PCA(n_components=13).fit_transform(wineData.loc[:,wineData.columns != 'Class'])
wineData_PCA = pd.DataFrame(reduced_wineData, columns=columns)

# sd = np.sqrt(reduced_wineData.explained_variance_)
# nr_components = [i+1 for i in range(13)]
# sd_sum = [np.sum(sd[:i+1]) for i in range(sd.size)]

fig, ax = plt.subplots()
ax.scatter(wineData_PCA.PCA2, wineData_PCA.PCA3, c=wineData.Class)
ax.set_xlabel('PCA2')
ax.set_ylabel('PCA3')
# ax.set_xlabel('Number of principal components')
# ax.set_ylabel('Sum of standard deviation')
# ax.legend()
# plt.show()
plt.savefig("img/pca_com_2_3.png",dpi=150)
plt.close()