from preprocess import PreProcess
import numpy as np

class Mutlist:
    def __init__(self):
        self.wordsList = None
        self.wordVectors = None

        self.numClasses = -1
        self.types = {}
        self.maxSeqLength = -1  # Maximum length of sentence

        self.numDimensions = 200  # Dimensions for each word vector
        self.batchSize = 24
        self.lstmUnits = 64
        self.iterations = 100000

    def create_matrix(self, dic_text, numFiles):
        ids = np.zeros((numFiles, self.maxSeqLength), dtype='int32')

        for k, v in dic_text.items():
            print(k)


    def main(self):
        preprocess = PreProcess()
        self.wordsList, self.wordVectors = preprocess.load_word2vec()
        dic_text, list_id, dic_results, self.types, self.maxSeqLength = preprocess.load_mutations()
        self.numClasses = len(self.types)

        self.create_matrix(dic_text, len(list_id))








if __name__ == "__main__":
    mutlist = Mutlist()
    mutlist.main()
