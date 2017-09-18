import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values

def target(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    return np.asarray(target)


my_data = pandas.read_csv("skulls.csv", delimiter=",")

X = removeColumns(my_data,0,1)
y = target(my_data,1)

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X,y,test_size=0.3,random_state=7)

neigh = KNeighborsClassifier(n_neighbors=1)
neigh23 = KNeighborsClassifier(n_neighbors=23)
neigh90 = KNeighborsClassifier(n_neighbors=90)

neigh.fit(X_trainset,y_trainset)
neigh23.fit(X_trainset,y_trainset)
neigh90.fit(X_trainset,y_trainset)

pred = neigh.predict(X_testset)
pred23 = neigh23.predict(X_testset)
pred90 = neigh90.predict(X_testset)

print("Neigh's Accuracy: ", metrics.accuracy_score(y_testset, pred))
print("Neigh23's Accuracy: ", metrics.accuracy_score(y_testset, pred23))
print("Neigh90's Accuracy: ", metrics.accuracy_score(y_testset, pred90))