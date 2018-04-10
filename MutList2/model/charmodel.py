from utils import seq_to_char
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Input, Sequential
from keras.models import load_model
from keras_contrib.utils import save_load_utils
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Masking
from keras_contrib.layers import CRF
from collections import defaultdict
from datetime import datetime
import sys
import time
import random
import collections
import matplotlib.pyplot as plt

from keras.layers import Dense, Masking
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras_contrib.layers import CRF

from keras.models import load_model
from keras_contrib.layers.advanced_activations import PELU


class CharModel:
    def __init__(self, model):
        self.maxSeqLength = -1  # Maximum length of sentence
        self.w_arit_mean = -1
        self.dic_corpus_char = defaultdict(list)
        self.dic_chars = defaultdict(list)
        self.train_data = []
        self.test_data = []
        self.alphabet = []
        self.labdict = {'O': 1, 'B': 2, 'I': 3, 'X': 0}
        self.lab_len = 4
        self.dict_labs_nopad = {'O': 0, 'B': 1, 'I': 2}
        self.num_labs = 3
        self.epochsN = 8

        self.model = model  # choose between padding or no padding
                            # 1 for padding
                            # 0 for no padding

    def creat_alphabet(self):
        # every character in corpus will count

        chars = []
        for idx in range(len(self.train_data)):
            corp = self.train_data[idx]['corpus']
            [chars.append(c) for c in corp]

        self.alphabet = list(set(chars))

    def convert_tags(self):
        for idx in range(len(self.train_data)):
            labs = self.train_data[idx]['labels']
            # convert SOBIE tags to numbers
            self.train_data[idx]["bion"] = [self.dict_labs_nopad[i] for i in labs]

    def get_seq(self, DICT):
        sequences = []
        labels = []

        for i in range(len(self.train_data)):
            corp = self.train_data[i]['corpus']
            labs = self.train_data[i]['labels']

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

        #sequences = sequences[:100]
        #labels = labels[:100]

        X = pad_sequences(sequences, maxlen=self.maxSeqLength, padding='post')
        y_pad = pad_sequences(labels, maxlen=self.maxSeqLength, padding='post')

        #X = pad_sequences(sequences, maxlen=self.w_arit_mean, padding='post', truncating='post')
        #y_pad = pad_sequences(labels, maxlen=self.w_arit_mean, padding='post', truncating='post')

        y = [to_categorical(i, num_classes=self.lab_len) for i in y_pad]

        # Set up the keras model
        input = Input(shape=(self.maxSeqLength,))
        el = Embedding(n_char + 1, 200, name="embed")(input)
        bl1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm1")(el)
        bl2 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm2")(bl1)
        bl3 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                            name="lstm3")(bl2)
        model = TimeDistributed(Dense(self.lab_len, activation="relu"))(bl3)
        crf = CRF(self.lab_len)  # CRF layer
        out = crf(model)  # output

        model = Model(input, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        history = model.fit(X, np.array(y), batch_size=32, epochs=self.epochsN, validation_split=0.1, verbose=0)

        save_load_utils.save_all_weights(model, 'char_max_seq.h5')

    def model_no_padding(self, DICT, n_char):

        # convert BIO tags to numbers
        self.convert_tags()

        for i in range(len(self.train_data)):
            corp = self.train_data[i]['corpus']
            labs = self.train_data[i]['labels']

            corp_num = []
            for c in corp:
                corp_num.append(DICT.get(c))
            self.train_data[i]['corpus'] = corp_num

        # get all sizes from the sequences with training data
        train_l_d = {}
        train_l_labels = {}
        for seq in self.train_data:
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
        model = TimeDistributed(Dense(self.num_labs, activation="relu"))(bl3)
        crf = CRF(self.num_labs)  # CRF layer
        out = crf(model)  # output

        model = Model(il, out)
        model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()

        # OK, start actually training
        for epoch in range(self.epochsN):
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

        save_load_utils.save_all_weights(model, 'char_no_pad.h5')

    def seqs_distribution(self):
        dimensions = {}
        for seq in self.train_data:
            l = len(seq['corpus'])

            if l not in dimensions: dimensions[l] = 0
            value = dimensions.get(l)
            value = value + 1
            dimensions[l] = value

        od = collections.OrderedDict(sorted(dimensions.items()))

        df = pd.DataFrame.from_dict(od, orient='index')
        df.plot(kind='bar', figsize=(50,10))
        #plt.show()

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
        self.train_data = reader.read(1)
        self.test_data = reader.read(0)

        self.creat_alphabet()

        # we associate every character in our alphabet to a number:
        # e.g. b => 1 d => 3 etc.
        DICT = {ch: ix + 1 for ix, ch in enumerate(self.alphabet)}
        n_char = len(self.alphabet)

        # model with padding
        if self.model == 1:
            # biggest sequence
            self.maxSeqLength = reader.get_length(self.train_data)
            # sequences length distribution
            self.w_arit_mean = int(self.seqs_distribution())

            # model with padding on the sequences
            self.model_with_padding(DICT, n_char)

        # model without padding - similar to chemlistem
        elif self.model == 0:
            self.model_no_padding(DICT, n_char)

        elif self.model == 2:
            # biggest sequence
            self.maxSeqLength = reader.get_length(self.train_data)
            # sequences length distribution
            self.w_arit_mean = int(self.seqs_distribution())

            model.load_model(DICT, n_char, "normal")

    def load_model(self, DICT, n_char, type):

        if type == "normal":
            # Load Character Model without padding
            il = Input(shape=(None,), dtype='int32')
            el = Embedding(n_char + 1, 200, name="embed")(il)
            bl1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                                name="lstm1")(el)
            bl2 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                                name="lstm2")(bl1)
            bl3 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5), merge_mode="concat",
                                name="lstm3")(bl2)
            model = TimeDistributed(Dense(self.num_labs, activation="relu"))(bl3)
            crf = CRF(self.num_labs)  # CRF layer
            out = crf(model)  # output
            model = Model(il, out)
            model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

            save_load_utils.load_all_weights(model, 'char_no_pad.h5')


            # get sequences and labels separated.
            # convert BIO tags to numbers
            test = []
            test_labels = []
            for i in range(len(self.test_data)):
                corp = self.test_data[i]['corpus']
                labs = self.test_data[i]['labels']

                corp_num = []
                for c in corp:
                    corp_num.append(DICT.get(c))
                test.append(corp_num)

                # pass labels to its numeric value
                labs_num = []
                for l in labs:
                    labs_num.append(self.dict_labs_nopad.get(l))
                test_labels.append(labs_num)

            for i in range(len(test)):
                p = model.predict(np.array([test[i]]))
                p = np.argmax(p, axis=-1)
                true = test_labels[i]
                print(len(p[0]))
                print(len(true))
                print("\n\n")

        elif type == "max_seq":
            # Set up the keras model
            input = Input(shape=(self.maxSeqLength,))
            el = Embedding(n_char + 1, 200, name="embed")(input)
            bl1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5, dropout=0.5),
                                merge_mode="concat",
                                name="lstm1")(el)
            bl2 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5),
                                merge_mode="concat",
                                name="lstm2")(bl1)
            bl3 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5),
                                merge_mode="concat",
                                name="lstm3")(bl2)
            model = TimeDistributed(Dense(self.lab_len, activation="relu"))(bl3)
            crf = CRF(self.lab_len)  # CRF layer
            out = crf(model)  # output

            model = Model(input, out)
            model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
            save_load_utils.load_all_weights(model, 'char_max_seq.h5')

            test = []
            test_labels = []
            for i in range(len(self.test_data)):
                corp = self.test_data[i]['corpus']
                labs = self.test_data[i]['labels']

                corp_num = []
                for c in corp:
                    corp_num.append(DICT.get(c))
                test.append(corp_num)

                # pass labels to its numeric value
                labs_num = []
                for l in labs:
                    labs_num.append(self.labdict.get(l))
                test_labels.append(labs_num)

            test = pad_sequences(test, maxlen=self.maxSeqLength, padding='post')

            for i in range(len(test)):
                p = model.predict(np.array([test[i]]))
                p = np.argmax(p, axis=-1)
                true = test_labels[i]
                print(p[0])
                #print(true)
                print("\n\n")

        elif type == "w_arit_mean":
            # Set up the keras model
            input = Input(shape=(self.w_arit_mean,))
            el = Embedding(n_char + 1, 200, name="embed")(input)
            bl1 = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.5, dropout=0.5),
                                merge_mode="concat",
                                name="lstm1")(el)
            bl2 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5),
                                merge_mode="concat",
                                name="lstm2")(bl1)
            bl3 = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.5, dropout=0.5),
                                merge_mode="concat",
                                name="lstm3")(bl2)
            model = TimeDistributed(Dense(self.lab_len, activation="relu"))(bl3)
            crf = CRF(self.lab_len)  # CRF layer
            out = crf(model)  # output

            model = Model(input, out)
            model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

            save_load_utils.load_all_weights(model, 'char_w_arit_mean.h5')

            test = []
            test_labels = []
            for i in range(len(self.test_data)):
                corp = self.test_data[i]['corpus']
                labs = self.test_data[i]['labels']

                corp_num = []
                for c in corp:
                    corp_num.append(DICT.get(c))
                test.append(corp_num)

                # pass labels to its numeric value
                labs_num = []
                for l in labs:
                    labs_num.append(self.labdict.get(l))
                test_labels.append(labs_num)

            test = pad_sequences(test, maxlen=self.w_arit_mean, padding='post', truncating='post')

            for i in range(len(test)):
                p = model.predict(np.array([test[i]]))
                p = np.argmax(p, axis=-1)
                true = test_labels[i]
                print(p[0])
                #print(true)
                print("\n\n")


if __name__ == "__main__":
    model = CharModel(2)
    model.main()

