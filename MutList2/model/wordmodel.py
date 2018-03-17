from utils import wordUtils
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import KFold
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

class WordModel:
    def __init__(self):
        self.maxSeqLength = 150  # Maximum length of sentence
        self.numDimensions = 200  # Dimensions for each word vector
        self.words_list = None
        self.embedding_matrix = None

    def create_matrix(self, train, test):
        num_train = len(train)  # number of sentences
        num_test = len(test)

        ids_train = np.zeros((num_train, self.maxSeqLength), dtype=object)
        file_counter = 0

        for sentence in train:
            words = sentence.split()
            token_counter = 0
            for word in words:
                try:
                    id = self.words_list.index(word)
                    ids_train[file_counter][token_counter] = self.embedding_matrix[id]
                except ValueError:
                    ids_train[file_counter][token_counter] = self.embedding_matrix[499999]  # full of zeros
                token_counter = token_counter + 1

                if token_counter >= self.maxSeqLength:
                    break

            file_counter = file_counter + 1

        print(ids_train.shape)

        return ids_train

    def create_matrix_words(self, words_set):
        # create a weight matrix for words in training docs
        vocab_size = len(words_set)

        embedding_matrix = np.zeros((vocab_size, self.numDimensions))
        i = 0
        for word in words_set:
            try:
                id = self.words_list.index(word)
                embedding_vector = self.embedding_matrix[id]
            except ValueError:
                embedding_vector = self.embedding_matrix[499999]  # full of zeros
            embedding_matrix[i] = embedding_vector

            i = i + 1

        return embedding_matrix

    def main(self):
        utils = wordUtils.Utils()
        self.words_list, self.embedding_matrix = utils.load_word2vec()
        sentences, labels, maxSeqLength, words_set = utils.load_seq_string()
        #print(len(words_set))

        t = Tokenizer(filters='',split=" ")
        t.fit_on_texts(sentences)
        sequences_1 = t.texts_to_sequences(sentences)
        data = pad_sequences(sequences_1, maxlen=self.maxSeqLength, padding='post')



        #ids_train = self.create_matrix_words(words_set)

        # define the model
        #model = Sequential()
        #model.add(Embedding(len(words_set), self.numDimensions, weights=[ids_train], input_length=self.maxSeqLength,
        #                   trainable=False))



if __name__ == "__main__":
    model = WordModel()
    model.main()
