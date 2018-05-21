from numpy import array
from numpy import asarray
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
        self.GLOVE_DIR = "/Users/pmatos9/Desktop/pedrinho/tese/glove/glove.6B.200d.txt"

    # function to load pre-processed words in word2vec from a combination of PubMed and PMC texts
    def load_word2vec(self):
        word_vectors = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)


        print('Found %s word vectors of word2vec' % len(word_vectors.vocab))

        words_list = word_vectors.index2word
        embedding_matrix = np.zeros((len(word_vectors.vocab)+1, 200), dtype='float32')
        for i in range(len(words_list)):
            word = words_list[i]
            embedding_matrix[i] = word_vectors.word_vec(word)

        return words_list, embedding_matrix


    def load_glove(self):
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(self.GLOVE_DIR, encoding="utf8")
        words = []
        for line in f:
            values = line.split()
            word = values[0]
            words.append(word)
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        # create a weight matrix for words
        embedding_matrix = np.zeros((len(embeddings_index)+1, 200))
        for i in range(len(words)):
            embedding_vector = embeddings_index.get(words[i])
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


        print('Loaded %s word vectors.' % len(embedding_matrix))
        return words, embedding_matrix
