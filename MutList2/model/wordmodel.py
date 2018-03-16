from utils import wordUtils
import numpy as np

class WordModel:
    def __init__(self):
        self.maxSeqLength = -1  # Maximum length of sentence
        self.numDimensions = 200  # Dimensions for each word vector
        self.words_list = None
        self.embedding_matrix = None

    def create_matrix(self, train, test):
        num_train = len(train)  # number of sentences
        num_test = len(test)

        ids_train = np.zeros((num_train, self.maxSeqLength), dtype='int32')
        file_counter = 0

        for sentence in train:
            words = sentence.split()
            token_counter = 0
            for word in words:
                try:
                    ids_train[file_counter][token_counter] = self.words_list.index(word)
                except ValueError:
                    ids_train[file_counter][token_counter] = 499999  # full of zeros
                token_counter = token_counter + 1

                if token_counter >= self.maxSeqLength:
                    break

            file_counter = file_counter + 1


        ids_test = np.zeros((num_test, self.maxSeqLength), dtype='int32')
        file_counter = 0

        for sentence in test:
            words = sentence.split()
            token_counter = 0
            for word in words:
                try:
                    ids_test[file_counter][token_counter] = self.words_list.index(word)
                except ValueError:
                    ids_test[file_counter][token_counter] = 499999  # full of zeros
                token_counter = token_counter + 1

                if token_counter >= self.maxSeqLength:
                    break

            file_counter = file_counter + 1

        print(ids_train.shape)
        print(ids_test.shape)


    def main(self):
        utils = wordUtils.Utils()
        self.words_list, self.embedding_matrix = utils.load_word2vec()
        sentences, labels, self.maxSeqLength = utils.load_seq()
        train_d, test_d, train_lab, test_lab = utils.split_data(sentences, labels)
        print(self.maxSeqLength)
        print(len(train_d))
        print(len(test_d))
        print("-----\n")

        self.create_matrix(train_d, test_d)




if __name__ == "__main__":
    model = WordModel()
    model.main()
