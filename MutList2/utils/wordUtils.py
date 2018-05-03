import numpy as np
from gensim.models import KeyedVectors
import os
from sklearn.model_selection import train_test_split

class Utils:
    def __init__(self):
        # Load GLOVE vectors
        self.corpus_dir = '../corpus/Doc_corpus/'
        self.labels_dir = '../corpus/Doc_label/'
        self.corpus_all = '/home/eldiablo/Documents/DeepLearMut/MutList2/utils/corpus_test/'
        self.word2vec_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wikipedia-pubmed-and-PMC-w2v.bin'

    # function to load pre-processed words in word2vec from a combination of PubMed and PMC texts
    def load_word2vec(self):
        word_vectors = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True,
                                                        limit=499999)  # limit just by now to speed up the run time

        print('Found %s word vectors of word2vec' % len(word_vectors.vocab))

        words_list = word_vectors.index2word
        embedding_matrix = np.zeros((500000, 200), dtype='float32')
        for i in range(len(words_list)):
            word = words_list[i]
            embedding_matrix[i] = word_vectors.word_vec(word)

        return words_list, embedding_matrix

