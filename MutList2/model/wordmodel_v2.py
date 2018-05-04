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
            words_id = []
            for i in words:
                vocab.append(i)

                # get the number of the embedding
                try:
                    # the index of the word in the embedding matrix
                    words_id.append(self.words_list.index(i))
                except ValueError:
                    # use the embedding full of zeros to identify an unknown word
                    words_id.append(unword_n)

            tmp_dic['tokens'] = words_id

            # convert SOBIE tags to numbers
            tags = doc['bio']
            tags = [self.lablist[i] for i in tags]
            tmp_dic['bion'] = tags
            train.append(tmp_dic)


        # get vocab size
        vocab = list(set(vocab))
        vocab_size = len(vocab)

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



if __name__ == "__main__":
    model = WordModel()
    model.main()
