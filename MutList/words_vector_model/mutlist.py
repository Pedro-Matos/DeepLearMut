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
        dic_counter = {}
        file_counter = 0

        for k, v in dic_text.items():
            indexCounter = 0
            dic_tmp = {file_counter: k}
            dic_counter.update(dic_tmp)

            for token in v:
                try:
                    ids[file_counter][indexCounter] = self.wordsList.index(token)
                except ValueError:
                    ids[file_counter][indexCounter] = 999999  # Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= self.maxSeqLength:
                    break
            file_counter = file_counter + 1
        print(dic_counter)
        return ids

    def main(self):
        preprocess = PreProcess()
        self.wordsList, self.wordVectors = preprocess.load_word2vec()
        dic_text, list_id, dic_results, self.types, self.maxSeqLength = preprocess.load_mutations()
        self.numClasses = len(self.types)

        ids = self.create_matrix(dic_text, len(list_id))









if __name__ == "__main__":
    mutlist = Mutlist()
    mutlist.main()
