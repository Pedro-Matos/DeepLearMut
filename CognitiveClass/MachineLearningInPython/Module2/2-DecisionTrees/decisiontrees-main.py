import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

my_data = pandas.read_csv("skulls.csv", delimiter=",")

featureNames = list(my_data.columns.values)[2:6]
# Remove the column containing the target name since it doesn't contain numeric values.
# axis=1 means we are removing columns instead of rows.
X = my_data.drop(my_data.columns[[0,1]], axis=1).values
targetNames = my_data["epoch"].unique().tolist()
y = my_data["epoch"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y,test_size=0.3,random_state=3)

skullsTree = DecisionTreeClassifier(criterion="entropy")
skullsTree.fit(X_trainset, y_trainset)

predTree = skullsTree.predict(X_testset)

print("pred: ", predTree[0:5])
print(y_testset[0:5])
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

dot_data = StringIO()
filename = "skulltree.png"
out=tree.export_graphviz(skullsTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')