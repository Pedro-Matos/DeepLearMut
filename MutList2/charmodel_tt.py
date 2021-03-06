import glob
import shutil

import keras
from keras.callbacks import ModelCheckpoint

from utils import seq_to_char
from utils import test_to_char
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
        self.test_data = defaultdict(list)
        self.test_all = defaultdict(list)
        self.alphabet = ['~', 'Z', 'l', 'u', 'c', '\n', '2', 'k', '{', ']', 'R', '%', '+', '.', 'B', 'g', '\\', 'a',
                         'p', '3', ';',
                         '}', 'r', 'Q', '>', 'J', 'V', 'D', '-', '0', 'i', 'F', '6', '#', 'x', '<', 'Y', ',', "'", 'y',
                         '[', 'U', '8',
                         'd', 'T', '"', ' ', 'O', 't', 'N', 'C', 'K', 'o', 'X', '1', 'f', 'v', 'h', 'n', '9', '4', 'G',
                         'L', 'e', 'A',
                         '=', '(', 'E', 'S', 'P', 'W', ')', '?', 'm', '/', 'j', '5', 's', 'I', ':', 'w', 'b', 'q', '7',
                         'M', '_', 'z', 'H', '*']

        self.labdict = {'O': 1, 'B': 2, 'I': 3, 'X': 0}
        self.lab_len = 4
        self.dict_labs_nopad = {'O': 0, 'B': 1, 'I': 2}
        self.num_labs = 3
        self.epochsN = 15

        self.model = model  # choose between padding or no padding
        # 1 for padding
        # 0 for no padding

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

        # sequences = sequences[:100]
        # labels = labels[:100]

        # X = pad_sequences(sequences, maxlen=self.w_arit_mean, padding='post', truncating='post')
        # y_pad = pad_sequences(labels, maxlen=self.w_arit_mean, padding='post', truncating='post')

        X = pad_sequences(sequences, maxlen=self.maxSeqLength, padding='post')
        y_pad = pad_sequences(labels, maxlen=self.maxSeqLength, padding='post')

        y = [to_categorical(i, num_classes=self.lab_len) for i in y_pad]

        # early stopping and best epoch
        #early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')
        #filepath = "max-seq.h5"
        #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max')
        #callbacks_list = [checkpoint, early_stop]

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

        #treinar com 32, 147, 245, 735
        history = model.fit(X, np.array(y), batch_size=32, epochs=self.epochsN, validation_split=0.0, verbose=1)
        # save all epochs
        save_load_utils.save_all_weights(model, 'max_seq_%s_32b.h5' % self.epochsN)

        #

    def model_no_padding(self, DICT, n_char):

        # convert BIO tags to numbers
        self.convert_tags()

        '''
        check if bion contains 'B' and 'I'
        for i in self.train_data:
            print(i['bion'])
        '''

        for i in range(len(self.train_data)):
            corp = self.train_data[i]['corpus']

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

        '''
        for i in range(len(train_l_d[110])):
            print(len(train_l_d[110][i]) == len(train_l_labels[110][i]))
            print()
        print("\n\n")

        for i in range(len(train_l_d[31])):
            print(len(train_l_d[31][i]) == len(train_l_labels[31][i]))
        print("\n\n")

        for i in range(len(train_l_d[103])):
            print(len(train_l_d[103][i]) == len(train_l_labels[103][i]))
        print("\n\n")
        exit()
        '''
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

        f_best = -1
        f_index = -1
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

                ty = [to_categorical(i, num_classes=self.num_labs) for i in y]

                # This trains in mini-batches
                model.fit(tx, np.array(ty), verbose=0, epochs=1)
            print("Trained at", datetime.now())

            # save all epochs
            save_load_utils.save_all_weights(model, 'mini-batch-results/epoch_%s.h5' % epoch)
            # test the results
            self.test_minibatch(DICT, model)
            f = self.eval()

            if f > f_best:
                f_best = f
                f_index = epoch


        # Pick the best model, and save it with a useful name
        print("Choosing the best epoch")
        shutil.copyfile("mini-batch-results/epoch_%s.h5" % f_index, "minibatch_%s.h5" % f_index)


    def main(self):
        reader = seq_to_char.CorpusReader()
        test_data = test_to_char.DataToTest()
        self.train_data = reader.read()
        self.test_data, self.test_all = test_data.get_testset()

        # we associate every character in our alphabet to a number:
        # e.g. b => 1 d => 3 etc.
        DICT = {ch: ix + 1 for ix, ch in enumerate(self.alphabet)}
        n_char = len(self.alphabet)

        # model with padding
        if self.model == 1:
            # biggest sequence
            self.maxSeqLength = reader.get_length(self.train_data)

            # model with padding on the sequences
            self.model_with_padding(DICT, n_char)

        # model without padding - similar to chemlistem
        elif self.model == 0:
            self.model_no_padding(DICT, n_char)

    def test_minibatch(self, DICT, model):
        print("Getting accuracy of this batch:", datetime.now())
        # get sequences and labels separated.
        # convert BIO tags to numbers
        keys = self.test_data.keys()

        for key in keys:
            # getting all sequences from a document/corpus
            seqs = self.test_data.get(key)
            # print(key)
            abstract = open("mini-batch-silver/" + key + ".a1", 'w')
            position = 0
            offsets = defaultdict(list)
            counter = 0
            for seq in seqs:
                # pass to the number representation
                char_seq = []
                for c in seq:
                    char_seq.append(DICT.get(c))

                p = model.predict(np.array([char_seq]))
                p = np.argmax(p, axis=-1)

                # check if there are any mutations identified
                B = False
                for idx in p[0]:
                    if idx == 1:
                        B = True
                        offsets[counter].append(position)
                    elif idx == 2 and B:
                        offsets[counter].append(position)
                    elif idx == 0 and B:
                        B = False
                        counter = counter + 1
                    else:
                        B = False

                    position = position + 1

                # print(p[0])

            corpus = self.test_all.get(key)
            all_words = []
            for i in offsets:
                chunk = ""
                word = offsets.get(i)
                for c in word:
                    chunk = chunk + corpus[c]
                all_words.append(chunk)

            for i in range(len(all_words)):
                abstract.write(str(offsets.get(i)[0]) + "\t")
                abstract.write(str(offsets.get(i)[-1] + 1) + "\t")
                abstract.write(str(all_words[i]) + "\n")

    def eval(self):
        keys = []
        gold = defaultdict(list)
        silver = defaultdict(list)

        path_gold = 'corpus_char/tmVarCorpus/treated/gold_results/'
        docs = glob.glob(path_gold + "*.a1")

        path_silver = 'mini-batch-silver/'
        docs_silver = glob.glob(path_silver + "*.a1")

        if len(docs) != len(docs_silver):
            print("The folders do not contain the same size.")
            exit()

        print("Creating silver docs", datetime.now())

        for i in range(len(docs)):
            file = docs[i].split("/")[-1]
            keys.append(file)

        for key in keys:
            path = path_gold + key
            with open(path) as reading:
                sentences = reading.readlines()
                if sentences != []:
                    id = key.split(".")[0]
                    for sent in sentences:
                        sent = sent.rstrip()
                        sent = sent.split("\t")
                        t = (sent[0], sent[1], sent[2])
                        gold[id].append(t)

            path = path_silver + key
            with open(path) as reading:
                sentences = reading.readlines()
                if sentences != []:
                    id = key.split(".")[0]
                    for sent in sentences:
                        sent = sent.rstrip()
                        sent = sent.split("\t")
                        t = (sent[0], sent[1], sent[2])
                        silver[id].append(t)

        for i in range(len(keys)):
            keys[i] = keys[i].split(".")[0]

        print("Evaluating", datetime.now())
        # evaluate

        tp_all = 0
        fp_all = 0
        fn_all = 0

        for i in range(len(keys)):
            id = keys[i]

            tp = 0
            fp = 0
            fn = 0

            correct_ents = gold[id]
            pred_ents = silver[id]
            for ent in pred_ents:
                if ent in correct_ents:
                    tp += 1
                else:
                    fp += 1
            fn = len(correct_ents) - tp

            tp_all += tp
            fp_all += fp
            fn_all += fn



        if tp_all == 0:
            f = 0
        else:
            f = (2 * tp_all / (tp_all + tp_all + fp_all + fn_all))

        if tp_all == 0:
            precision = 0
        else:
            precision = tp_all / (tp_all + fp_all)

        if tp_all == 0:
            recal = 0
        else:
            recal = tp_all / (tp_all + fn_all)


        print("TP: " + str(tp_all))
        print("FP: " + str(fp_all))
        print("FN: " + str(fn_all))
        print("F: " + str(f))
        print("Precision: " + str(precision))
        print("Recal: " + str(recal))

        return f



if __name__ == "__main__":
    model = CharModel(0)
    model.main()
