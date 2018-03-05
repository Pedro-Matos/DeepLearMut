from preprocess import PreProcess
from sklearn.model_selection import train_test_split
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
        return ids

    def split_data(self, data, labels):
        # split the data to train and to test
        train_df, test_df = train_test_split(data, test_size=0.1, shuffle=False)
        labels_train, labels_test = train_test_split(labels, test_size=0.1, shuffle=False)

        return train_df, test_df, labels_train, labels_test


    def main(self):
        preprocess = PreProcess()
        self.wordsList, self.wordVectors = preprocess.load_word2vec()
        data, list_id, labels, types, self.maxSeqLength = preprocess.load_mutations()
        self.numClasses = len(self.types)

        # create dictionary of type and it's respective value in int
        count = 0
        for i in types:
            dic = {i: count}
            self.types.update(dic)
            count = count + 1

        # ids = self.create_matrix(dic_text, len(list_id))

        train_df, test_df, labels_train, labels_test = self.split_data(data, labels)

        # Spit out details about data
        print("\n=================================\nData details:")
        print("- Training-set:\t{}".format(len(train_df)))
        print("- Test-set:\t\t{}".format(len(test_df)))
        print("- Classes:\t\t{}".format(self.types))
        print("=================================\n\n")

if __name__ == "__main__":
    mutlist = Mutlist()
    mutlist.main()
