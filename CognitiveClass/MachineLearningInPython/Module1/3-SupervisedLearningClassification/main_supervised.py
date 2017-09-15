import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier

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

neigh = KNeighborsClassifier(n_neighbors=1)
neigh_7 = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X,y)
neigh_7.fit(X,y)

print('Prediction: ', neigh.predict(X[30]))
print('Prediction: ', neigh_7.predict(X[30]))
print('Actual:', y[30])
