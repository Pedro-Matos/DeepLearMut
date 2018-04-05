from utils import seq_to_char
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from collections import defaultdict
from datetime import datetime
import sys
import time
import random
import collections
import matplotlib.pyplot as plt
import click


class CharModel:
    def __init__(self, model):
        self.maxSeqLength = -1  # Maximum length of sentence
        self.w_arit_mean = -1
        self.dic_corpus_char = defaultdict(list)
        self.dic_chars = defaultdict(list)
        self.data = []
        self.alphabet = []
        self.labdict = {'O': 1, 'B': 2, 'I': 3, 'X': 0}
        self.lab_len = 4
        self.dict_labs_nopad = {'O': 0, 'B': 1, 'I': 2}
        self.num_labs = 3
        self.epochsN = 5

        self.model = model  # choose between padding or no padding
                            # 1 for padding
                            # 0 for no padding

    def creat_alphabet(self):
        # every character in corpus will count

        chars = []
        for idx in range(len(self.data)):
            corp = self.data[idx]['corpus']
            [chars.append(c) for c in corp]

        self.alphabet = list(set(chars))

    def convert_tags(self):
        for idx in range(len(self.data)):
            labs = self.data[idx]['labels']
            # convert SOBIE tags to numbers
            self.data[idx]["bion"] = [self.dict_labs_nopad[i] for i in labs]

    def get_seq(self, DICT):
        sequences = []
        labels = []

        for i in range(len(self.data)):
            corp = self.data[i]['corpus']
            labs = self.data[i]['labels']

            corp_num = []
            for c in corp:
                corp_num.append(DICT.get(c))
            sequences.append(corp_num)

            # pass labels to its numeric value
            labs_num = []
            for l in labs:
                labs_num.append(self.labdict.get(l))
            labels.append(labs_num)

        return sequences, labels

    def model_with_padding(self, DICT, n_char):

        # get sequences and labels separated.
        # convert BIO tags to numbers
        sequences, labels = self.get_seq(DICT)

        X = pad_sequences(sequences, maxlen=self.maxSeqLength, padding='post')
        y_pad = pad_sequences(labels, maxlen=self.maxSeqLength, padding='post')

        y = [to_categorical(i, num_classes=self.lab_len) for i in y_pad]

        tr_x, te_x, tr_y, te_y = train_test_split(X, y, test_size=0.3)

        # Set up the keras model
        input = Input(shape=(self.maxSeqLength,))
        el = Embedding(n_char + 1, 200, name="embed")(input)
        bl1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm1")(el)
        bl2 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm2")(bl1)
        bl3 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm3")(bl2)
        model = TimeDistributed(Dense(50, activation="relu"))(bl3)
        crf = CRF(self.lab_len)  # CRF layer
        out = crf(model)  # output

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        history = model.fit(tr_x, np.array(tr_y), batch_size=32, epochs=self.epochsN, validation_split=0.1, verbose=1)

    def model_no_padding(self, DICT, n_char):

        # convert BIO tags to numbers
        self.convert_tags()

        for i in range(len(self.data)):
            corp = self.data[i]['corpus']
            labs = self.data[i]['labels']

            corp_num = []
            for c in corp:
                corp_num.append(DICT.get(c))
            self.data[i]['corpus'] = corp_num

        # get all sizes from the sequences with training data
        train_l_d = {}
        train_l_labels = {}
        for seq in self.data:
            # corpus
            l = len(seq['corpus'])
            if l not in train_l_d: train_l_d[l] = []
            train_l_d[l].append(seq['corpus'])

            # labels
            l1 = len(seq['bion'])
            if l1 not in train_l_labels: train_l_labels[l1] = []
            train_l_labels[l1].append(seq['bion'])

        sizes = list(train_l_d.keys())

        # Set up the keras model
        il = Input(shape=(None,), dtype='int32')
        el = Embedding(n_char + 1, 200, name="embed")(il)
        bl1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm1")(el)
        bl2 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm2")(bl1)
        bl3 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm3")(bl2)
        dl = TimeDistributed(Dense(self.num_labs, activation="softmax"), name="output")(bl3)
        model = Model(inputs=il, outputs=dl)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        model.summary()

        # OK, start actually training
        for epoch in range(5):
            print("Epoch", epoch, "start at", datetime.now())
            # Train in batches of different sizes - randomize the order of sizes
            # Except for the first few epochs - train on the smallest examples first
            if epoch > 2:
                random.shuffle(sizes)  # For unknown reasons we can't train on a single token (i.e. character)
            for size in sizes:
                if size == 1: continue
                batch = train_l_d[size]
                labs = train_l_labels[size]

                tx = np.array([seq for seq in batch])
                ty = np.array([[to_categorical(i, num_classes=self.num_labs) for i in seq] for seq in labs])

                # This trains in mini-batches
                model.fit(tx, ty, verbose=0, epochs=1)
            print("Trained at", datetime.now())

    def seqs_distribution(self):
        dimensions = {}
        for seq in self.data:
            l = len(seq['corpus'])

            if l not in dimensions: dimensions[l] = 0
            value = dimensions.get(l)
            value = value + 1
            dimensions[l] = value

        od = collections.OrderedDict(sorted(dimensions.items()))

        df = pd.DataFrame.from_dict(od, orient='index')
        df.plot(kind='bar', figsize=(50,10))
        plt.show()

        # WEIGHTED ARITHMETIC MEAN
        numerator = 0
        denominator = 0
        for item in od:
            value = od.get(item)
            prod = item * value
            numerator = numerator + prod
            denominator = denominator + value

        w_arit_mean = numerator/denominator
        return w_arit_mean

    def main(self):
        reader = seq_to_char.CorpusReader()
        self.data = reader.read()
        self.creat_alphabet()

        # we associate every character in our alphabet to a number:
        # e.g. b => 1 d => 3 etc.
        DICT = {ch: ix + 1 for ix, ch in enumerate(self.alphabet)}
        n_char = len(self.alphabet)

        # model with padding
        if self.model == 1:
            # biggest sequence
            self.maxSeqLength = reader.get_length(self.data)
            # sequences length distribution
            self.w_arit_mean = self.seqs_distribution()

            # model with padding on the sequences
            self.model_with_padding(DICT, n_char)

        # model without padding - similar to chemlistem
        else:
            self.model_no_padding(DICT, n_char)


if __name__ == "__main__":
    model = CharModel(0)
    model.main()
