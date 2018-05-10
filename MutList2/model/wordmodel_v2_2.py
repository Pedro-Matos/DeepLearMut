import glob
from collections import defaultdict

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

        self.epochsN = 20

    def main(self):
        # get word embeddings
        utils = wordUtils.Utils()
        self.words_list, self.embedding_matrix = utils.load_word2vec()
        unword_n = len(self.words_list)

        # get the training corpus
        cr = corpusreader.CorpusReader(self.textfile, self.annotfile)
        corpus = cr.trainseqs

        train = []
        print("Processing training data", datetime.now())
        for doc in corpus:
            tmp_dic = {}


            tmp_dic['tokens'] = doc['tokens']

            # convert SOBIE tags to numbers
            tags = doc['bio']
            tags = [self.lablist[i] for i in tags]
            tmp_dic['bion'] = tags
            train.append(tmp_dic)


        # get the number of the embedding
        for idx in range(len(train)):
            words = train[idx]['tokens']
            words_id = []
            for i in words:
                # get the number of the embedding
                try:
                    # the index of the word in the embedding matrix
                    index = self.words_list.index(i)
                except ValueError:
                    # use the embedding full of zeros to identify an unknown word
                    index = unword_n

                # the index of the word in the embedding matrix
                words_id.append(index)

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
        model = Embedding(len(self.words_list) + 1, 200, weights=[self.embedding_matrix], trainable=False)(input)
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

        save_load_utils.save_all_weights(model, 'words_20.h5')

    def test_model(self, test_data, test_labels):
        # get word embeddings
        utils = wordUtils.Utils()
        self.words_list, self.embedding_matrix = utils.load_word2vec()
        unword_n = len(self.words_list)


        input = Input(shape=(None,))
        model = Embedding(unword_n + 1, 200, weights=[self.embedding_matrix], trainable=False)(input)
        model = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(model)  # variational biLSTM
        model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
        crf = CRF(self.lab_len)  # CRF layer
        out = crf(model)  # output

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()

        save_load_utils.load_all_weights(model, 'words_20.h5')

        # get the training corpus
        cr = corpusreader.CorpusReader(test_data, test_labels)
        corpus = cr.trainseqs

        # get the number of the embedding
        for idx in range(len(corpus)):
            words = corpus[idx]['tokens']
            words_id = []
            for i in words:

                # get the number of the embedding
                try:
                    # the index of the word in the embedding matrix
                    index = self.words_list.index(i)
                except ValueError:
                    # use the embedding full of zeros to identify an unknown word
                    index = unword_n

                # the index of the word in the embedding matrix
                words_id.append(index)

            corpus[idx]['embs'] = words_id


        for doc in corpus:
            doc_arr = doc['embs']
            p = model.predict(np.array([doc_arr]))
            p = np.argmax(p, axis=-1)

            position = 0
            offsets = defaultdict(list)
            counter = 0
            # check if there are any mutations identified
            # {'O': 0, 'B-E': 1, 'I-E': 2, 'E-E': 3, 'S-E': 4}
            B = False
            last = 0
            for idx in p[0]:
                if idx == 1 and last == 1:
                    counter = counter + 1
                    offsets[counter].append(position)
                    B = True
                elif idx == 1:
                    B = True
                    offsets[counter].append(position)
                    last = 1
                elif idx == 2 and B:
                    offsets[counter].append(position)
                    last = 2
                elif idx == 3 and B:
                    offsets[counter].append(position)
                    last = 3
                    B = False
                    counter = counter + 1
                elif idx == 4:
                    offsets[counter].append(position)
                    counter = counter + 1
                    last = 4
                else:
                    B = False

                position = position + 1

            print(p)
            print(doc)
            # open file to write
            textid = str(doc['textid'])
            abstract = open("silver_words_20epoch/" + textid + ".a1", 'w')
            for i in offsets:
                word = offsets.get(i)
                print(len(word))
                print(word)
                size = len(word)
                if size == 1:
                    s = word[0]     # just one; singleton

                    abstract.write(str(doc['tokstart'][s]) + "\t")
                    abstract.write(str(doc['tokend'][s]) + "\t")
                    abstract.write(str(doc['tokens'][s]) + "\n")


                    print(doc['tokstart'][s])
                    print(doc['tokend'][s])
                    print(doc['tokens'][s])

                elif size > 1:
                    s = word[0]     # start of token
                    e = word[-1]    # end of token

                    abstract.write(str(doc['tokstart'][s]) + "\t")
                    abstract.write(str(doc['tokend'][e]) + "\t")


                    print(doc['tokstart'][s])
                    print(doc['tokend'][e])
                    print("just test:")

                    token = ""
                    for c in word:
                        token = token + doc['tokens'][c]

                        print(doc['tokstart'][c])
                        print(doc['tokend'][c])


                    print("test ended:")
                    print(token)

                    abstract.write(str(token) + "\n")

                print("---")
            print("\n\n---\n\n")


if __name__ == "__main__":
    model = WordModel()
    #model.main()
    test_data = '../corpus_char/tmVarCorpus/treated/test_data.txt'
    test_labels = '../corpus_char/tmVarCorpus/treated/test_labels.tsv'
    model.test_model(test_data, test_labels)
