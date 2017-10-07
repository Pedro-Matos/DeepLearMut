from IPython.core.display import HTML
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

def targetAndtargetNames(numpyArray, targetColumnIndex):
    target_dict = dict()
    target = list()
    target_names = list()
    count = -1
    for i in range(len(my_data.values)):
        if my_data.values[i][targetColumnIndex] not in target_dict:
            count += 1
            target_dict[my_data.values[i][targetColumnIndex]] = count
        target.append(target_dict[my_data.values[i][targetColumnIndex]])
    # Since a dictionary is not ordered, we need to order it and output it to a list so the
    # target names will match the target.
    for targetName in sorted(target_dict, key=target_dict.get):
        target_names.append(targetName)
    return np.asarray(target), target_names



my_data = pandas.read_csv("skulls.csv", delimiter=",")
#print(my_data.values)

new_data = removeColumns(my_data,0,1)
#print(new_data)

target, target_names = targetAndtargetNames(my_data,1)


X = new_data
y = target
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X,y)
print('Prediction: ', neigh.predict(new_data[10]))
print('Actual:', y[10])