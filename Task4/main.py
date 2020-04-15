from sklearn import tree
import pandas as pd
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'D:/ProgramFiles/Graphviz/release/bin'

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

### Create full decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(wineData.loc[:,wineData.columns != 'Class'].values, wineData['Class'].values)

### Draw full decision tree
column_names = wineData.loc[:,wineData.columns != 'Class'].columns.values
class_names = ["1","2","3"]
graph_data = tree.export_graphviz(clf, out_file=None, feature_names=column_names, filled=True,
                                  rounded=True, special_characters=True, class_names=class_names)
graph = graphviz.Source(graph_data, format='png')
graph.render("FullTree")

