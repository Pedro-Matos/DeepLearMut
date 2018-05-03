from utils import wordUtils
from utils import corpusreader
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
        self.numDimensions = 200  # Dimensions for each word vector
        self.words_list = None
        self.embedding_matrix = None

        self.textfile = '../corpus_char/tmVarCorpus/treated/train_data.txt'
        self.annotfile = '../corpus_char/tmVarCorpus/treated/train_labels.tsv'

        self.lablist = {'O': 0, 'B-E': 1, 'I-E': 2, 'E-E': 3, 'S-E': 4}
        self.lab_len = 5


    def main(self):
        # get word embeddings
        utils = wordUtils.Utils()
        self.words_list, self.embedding_matrix = utils.load_word2vec()

        # get the training corpus
        cr = corpusreader.CorpusReader(self.textfile, self.annotfile)
        corpus = cr.trainseqs

        seqs = []
        labels = []
        for doc in corpus:
            seqs.append(doc['tokens'])

            # convert SOBIE tags to numbers
            tags = doc['bio']
            tags = [self.lablist[i] for i in tags]
            labels.append(tags)



        # convert word tokens to the word embeddings



if __name__ == "__main__":
    model = WordModel()
    model.main()
