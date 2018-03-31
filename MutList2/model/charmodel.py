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
        self.labdict = {'O': 1, 'B': 2, 'I': 3}

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

    def main(self):
        reader = seq_to_char.CorpusReader()
        self.data = reader.read()
        self.creat_alphabet()
        self.maxSeqLength = reader.get_length(self.data)

        # we associate every character in our alphabet to a number:
        # e.g. b => 1 d => 3 etc.
        DICT = {ch: ix for ix, ch in enumerate(self.alphabet)}

        # convert BIO tags to numbers
        self.convert_tags()


if __name__ == "__main__":
    model = CharModel()
    model.main()