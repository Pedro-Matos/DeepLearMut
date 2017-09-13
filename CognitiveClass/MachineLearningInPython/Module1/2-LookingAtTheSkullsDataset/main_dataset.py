from IPython.core.display import HTML
import numpy as np
import pandas
from sklearn.neighbors import KNeighborsClassifier

HTML("""
<style type="text/css">
    #ans:hover { background-color: black; }
    #ans {padding: 6px; 
        background-color: white; 
        border: green 2px solid; 
        font-weight: bold; }
</style>
""")
# Remove the column containing the target name since it doesn't contain numeric values.
# Also remove the column that contains the row number
# axis=1 means we are removing columns instead of rows.
# Function takes in a pandas array and column numbers and returns a numpy array without
# the stated columns
def removeColumns(pandasArray, *column):
    return pandasArray.drop(pandasArray.columns[[column]], axis=1).values



my_data = pandas.read_csv("skulls.csv", delimiter=",")
print(my_data)

new_data = removeColumns(my_data,0,1)
print(new_data)