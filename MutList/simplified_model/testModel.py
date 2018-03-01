import numpy as np
import tensorflow as tf
from preprocessing import PreProcessing
from sklearn.model_selection import train_test_split
import datetime
from random import randint

class Teste:
    def __init__(self):
        self.maxSeqLength = 1  # Maximum length of sentence
        self.types = {}
        self.batchSize = 24
        self.numClasses = -1
        self.lstmUnits = 64
        self.iterations = 100000


    def main(self):
        preprocess = PreProcessing()
        data, labels, types = preprocess.load_mutations()
        self.numClasses = len(types)

        # create dictionary of type and it's respective value in int
        count = 0
        for i in types:
            dic = {i: count}
            self.types.update(dic)
            count = count + 1

        train_seqs, test_seqs, train_labels, test_labels = self.normalize_data(data, labels)

        # Spit out details about data
        classes = np.sort(np.unique(train_labels))
        print("\n=================================\nData details:")
        print("- Training-set:\t{}".format(len(train_seqs)))
        print("- Test-set:\t\t{}".format(len(test_seqs)))
        print("- Classes:\t\t{}".format(classes))
        print("=================================\n\n")


if __name__ == "__main__":
    mutlist = Teste()
    mutlist.main()
