# DeepLearMut
Tese's repository. 

## Repositories

**auxiliar** has a script in python that reads the file that contains all the examples and splits it into 2 files.
The first file contais the id of the example and the text(Title + Abstract), the 3 separated by one tab each. One example per line.
The second file contains the results of each example: the mutation and where it is located in the text

**corpus** contains the texts that are being used to feed data into the model and another texts that can be used later.

**model** contais all code related to the deep learning model

## Model Repository

**preprocess.py** treats the input data and returns the text into in a dictionary style {id: text} and another dictionary with the results {id: results[]}

**mutlist.py** it's the main program and contains the pipeline from getting the input to feeding it to the lstm model.
