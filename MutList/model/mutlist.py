import numpy as np
import tensorflow as tf
from preprocess import PreProcess
import matplotlib.pyplot as plt
import math

class MutList:
    def __init__(self):
        self.model = "lstm"
        self.wordsList = None
        self.wordVectors = None
        self.dic_text = None
        self.list_id = None
        self.dic_results = None
        self.list_types = None
        self.maxSeqLength = -1  # Maximum length of sentence
        self.numDimensions = 300  # Dimensions for each word vector

    def main(self):
        print("Starting")
        prep = PreProcess()
        self.wordsList, self.wordVectors = prep.load_glove()
        self.dic_text, self.list_id, self.dic_results, self.list_types = prep.load_mutations()
        self.maxSeqLength = self.round_int(int(prep.average_words))
        print(self.maxSeqLength)


    def round_int(self, x):
        return 10 * ((x + 5) // 10)




if __name__ == "__main__":
    mutlist = MutList()
    mutlist.main()
