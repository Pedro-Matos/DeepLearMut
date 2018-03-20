from utils import wordUtils
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF


class WordModel:
    def __init__(self):
        self.maxSeqLength = -1  # Maximum length of sentence
        self.numDimensions = 200  # Dimensions for each word vector
        self.words_list = None
        self.embedding_matrix = None
        self.seed = 42
        self.k = 5  # for Kfold
        self.tags = ['O', 'Mut']

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

    def create_matrix_words(self, t, vocab_size):
        # create a weight matrix for words in training docs

        embedding_matrix = np.zeros((vocab_size, self.numDimensions))
        for word, i in t.word_index.items():
            try:
                id = self.words_list.index(word)
                embedding_vector = self.embedding_matrix[id]
            except ValueError:
                embedding_vector = self.embedding_matrix[499999]  # full of zeros

            embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def main(self):
        utils = wordUtils.Utils()
        #self.words_list, self.embedding_matrix = utils.load_word2vec()
        sentences, labels = utils.load_seq_string()

        t = Tokenizer(filters='',split=" ")
        t.fit_on_texts(sentences)
        sequences_1 = t.texts_to_sequences(sentences)

        sum = 0
        i = 0
        for seq in sequences_1:
            l = len(seq)
            sum = sum + l
            i = i+1

        self.maxSeqLength = int(sum / i)

        words = []
        for word, i in t.word_index.items():
            words.append(word)

        X = pad_sequences(sequences_1, maxlen=self.maxSeqLength, padding='post')
        labs = pad_sequences(labels, maxlen=self.maxSeqLength, padding='post')
        n_tags = len(self.tags)
        y = [to_categorical(i, num_classes=n_tags) for i in labs]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)

        input = Input(shape=(self.maxSeqLength,))
        model = Embedding(input_dim=len(t.word_index) + 1, output_dim=20,
                          input_length=self.maxSeqLength, mask_zero=True)(input)  # 20-dim embedding
        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(model)  # variational biLSTM
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(n_tags)  # CRF layer
        out = crf(model)  # output

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

        file = open("test_pred.txt", "w")
        for i in range(len(X_te)):
            p = model.predict(np.array([X_te[i]]))
            p = np.argmax(p, axis=-1)
            true = np.argmax(y_te[i], -1)
            file.write("{:15}||{:5}||{}".format("Word", "True", "Pred"))
            file.write("\n")
            #print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
            file.write(30 * "=")
            file.write("\n")
            #print(30 * "=")
            for w, t, pred in zip(X_te[i], true, p[0]):
                if w != 0:
                    #print("{:15}: {:5} {}".format(words[w - 1], self.tags[t], self.tags[pred]))
                    file.write("{:15}: {:5} {}".format(words[w - 1], self.tags[t], self.tags[pred]))
                    file.write("\n")

    def main_no_tokenizer(self):
        utils = wordUtils.Utils()
        # self.words_list, self.embedding_matrix = utils.load_word2vec()
        sentences, labels = utils.load_seq_all()

        sum = 0
        i = 0
        words = []
        sent_arr = []
        for seq in sentences:
            s = seq.split()
            sent_arr.append(s)
            l = len(s)
            sum = sum + l
            i = i + 1

            for idx in s:
                words.append(idx)

        words = list(set(words))
        n_words = len(words)
        self.maxSeqLength = int(sum / i)

        word2idx = {w: i + 1 for i, w in enumerate(words)}

        X = [[word2idx[w] for w in s] for s in sent_arr]
        X = pad_sequences(maxlen=self.maxSeqLength, sequences=X, padding="post", value=0)

        labs = pad_sequences(labels, maxlen=self.maxSeqLength, padding='post', value=0)
        n_tags = len(self.tags)
        y = [to_categorical(i, num_classes=n_tags) for i in labs]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)

        input = Input(shape=(self.maxSeqLength,))
        model = Embedding(input_dim=n_words + 1, output_dim=20,
                          input_length=self.maxSeqLength, mask_zero=True)(input)  # 20-dim embedding
        model = Bidirectional(LSTM(units=1000, return_sequences=True,
                                   recurrent_dropout=0.1))(model)  # variational biLSTM
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(n_tags)  # CRF layer
        out = crf(model)  # output

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

        file = open("test_pred.txt", "w")
        for i in range(len(X_te)):
            p = model.predict(np.array([X_te[i]]))
            p = np.argmax(p, axis=-1)
            true = np.argmax(y_te[i], -1)
            file.write("{:15}||{:5}||{}".format("Word", "True", "Pred"))
            file.write("\n")
            # print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
            file.write(30 * "=")
            file.write("\n")
            # print(30 * "=")
            for w, t, pred in zip(X_te[i], true, p[0]):
                if w != 0:
                    # print("{:15}: {:5} {}".format(words[w - 1], self.tags[t], self.tags[pred]))
                    file.write("{:15}: {:5} {}".format(words[w - 1], self.tags[t], self.tags[pred]))
                    file.write("\n")


if __name__ == "__main__":
    model = WordModel()
    model.main_no_tokenizer()

# vocab_size = len(t.word_index) + 1
# print('Found %s unique tokens' % vocab_size)
# ids_train = self.create_matrix_words(t, vocab_size)

# define the model
# model = Sequential()
# model.add(Embedding(vocab_size, self.numDimensions, weights=[ids_train], input_length=self.maxSeqLength,
#                     trainable=False))
