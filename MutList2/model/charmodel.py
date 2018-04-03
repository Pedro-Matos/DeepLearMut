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


class CharModel:
    def __init__(self):
        self.maxSeqLength = -1  # Maximum length of sentence
        self.dic_corpus_char = defaultdict(list)
        self.dic_chars = defaultdict(list)
        self.data = []
        self.alphabet = []
        self.labdict = {'O': 0, 'B': 1, 'I': 2}
        self.lab_len = 3

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
            self.data[idx]["bion"] = [self.labdict[i] for i in labs]

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

    def main(self):
        reader = seq_to_char.CorpusReader()
        self.data = reader.read()
        self.creat_alphabet()
        self.maxSeqLength = reader.get_length(self.data)

        # we associate every character in our alphabet to a number:
        # e.g. b => 1 d => 3 etc.
        DICT = {ch: ix+1 for ix, ch in enumerate(self.alphabet)}

        # convert BIO tags to numbers
        self.convert_tags()

        # get sequences and labels separated
        sequences, labels = self.get_seq(DICT)
        
        X = pad_sequences(sequences, maxlen=self.maxSeqLength, padding='post')
        y_pad = pad_sequences(labels, maxlen=self.maxSeqLength, padding='post')
        y = [to_categorical(i, num_classes=self.lab_len) for i in y_pad]

        tr_x, te_x, tr_y, te_y = train_test_split(X, y, test_size=0.3)





if __name__ == "__main__":
    model = CharModel()
    model.main()