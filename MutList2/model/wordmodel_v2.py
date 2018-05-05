from utils import wordUtils
from utils import corpusreader
from datetime import datetime
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import random
from keras_contrib.utils import save_load_utils
from numpy import zeros




class WordModel:
    def __init__(self):
        self.numDimensions = 200  # Dimensions for each word vector
        self.words_list = None
        self.embedding_matrix = None


        self.textfile = '../corpus_char/tmVarCorpus/treated/train_data.txt'
        self.annotfile = '../corpus_char/tmVarCorpus/treated/train_labels.tsv'

        self.lablist = {'O': 0, 'B-E': 1, 'I-E': 2, 'E-E': 3, 'S-E': 4}
        self.lab_len = 5

        self.epochsN = 10


    def main(self):
        # get word embeddings
        utils = wordUtils.Utils()
        self.words_list, self.embedding_matrix = utils.load_word2vec()
        unword_n = len(self.words_list)

        '''
        print(self.words_list[0])
        print(self.words_list[1])
        print(self.words_list[10])
        print(self.words_list[100])
        '''
        '''
        </s>
        the
        a
        most
        '''
        # get the training corpus
        cr = corpusreader.CorpusReader(self.textfile, self.annotfile)
        corpus = cr.trainseqs

        train = []
        vocab = []
        print("Processing training data", datetime.now())
        for doc in corpus:
            tmp_dic = {}

            # create vocab size
            words = doc['tokens']

            for i in words:
                vocab.append(i)

            tmp_dic['tokens'] = doc['tokens']

            # convert SOBIE tags to numbers
            tags = doc['bio']
            tags = [self.lablist[i] for i in tags]
            tmp_dic['bion'] = tags
            train.append(tmp_dic)


        # get vocab size
        vocab = list(set(vocab))
        vocab_size = len(vocab)
        print(vocab_size)


        # create a weight matrix for words in training docs
        embedding_matrix = zeros((vocab_size+1, 200))
        cont = 0
        for word in vocab:

            # get the number of the embedding
            try:
                # the index of the word in the embedding matrix
                index = self.words_list.index(word)
            except ValueError:
                # use the embedding full of zeros to identify an unknown word
                index = unword_n

            embedding_vector = self.embedding_matrix[index]
            embedding_matrix[cont] = embedding_vector
            cont = cont +1

        # get the number of the embedding
        for idx in range(len(train)):
            words = train[idx]['tokens']
            words_id = []
            for i in words:
                # the index of the word in the embedding matrix
                words_id.append(vocab.index(i))

            train[idx]['tokens'] = words_id

        # get all sizes from the sequences with training data
        train_l_d = {}
        train_l_labels = {}
        for seq in train:
            # corpus
            l = len(seq['tokens'])
            if l not in train_l_d: train_l_d[l] = []
            train_l_d[l].append(seq['tokens'])

            # labels
            l1 = len(seq['bion'])
            if l1 not in train_l_labels: train_l_labels[l1] = []
            train_l_labels[l1].append(seq['bion'])

        sizes = list(train_l_d.keys())
        for i in sizes:
            if len(train_l_d[i]) != len(train_l_labels[i]):
                print("merda")

            for m in range(len(train_l_d[i])):
                if len(train_l_d[i][m]) != len(train_l_labels[i][m]):
                    print("XXX")



        input = Input(shape=(None,))
        model = Embedding(vocab_size+1, 200, weights=[embedding_matrix], trainable=False)(input)
        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(model)  # variational biLSTM
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(self.lab_len)  # CRF layer
        out = crf(model)  # output

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()

        # OK, start actually training
        for epoch in range(self.epochsN):
            print("Epoch", epoch, "start at", datetime.now())
            # Train in batches of different sizes - randomize the order of sizes
            # Except for the first few epochs
            if epoch > 2:
                random.shuffle(sizes)
            for size in sizes:
                batch = train_l_d[size]
                labs = train_l_labels[size]

                tx = np.array([seq for seq in batch])
                y = [seq for seq in labs]

                ty = [to_categorical(i, num_classes=self.lab_len) for i in y]

                # This trains in mini-batches
                model.fit(tx, np.array(ty), verbose=0, epochs=1)
            print("Trained at", datetime.now())

        save_load_utils.save_all_weights(model, 'words_10.h5')




if __name__ == "__main__":
    model = WordModel()
    model.main()
