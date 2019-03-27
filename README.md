# DeepLearMut
Thesis repository about deep learning to mutations in text analysis.

Directory's **MutList** is responsible for the work done with TensorFlow in order to get used to neural networks.


The second directory: **MutList2** is responsible for the final models done with Keras. 

The folder **"Evaluator"** contains a class that is responsible for evaluate the performance (precision, recall and f) of the models. It's done in separate by this class (it compares the gold and silver folders).

The folder **"corpus_char"** contains the tmVar dataset used in this work. 

The folder **"utils"** contains scripts related with the OSCAR4 tokenizer and with the parsing of the tmVar dataset.

The core of this work is in the **"charmodel_tt.py"** and **"wordmodel_tt.py"**. Those classes contain the character and word embeddings models. 
