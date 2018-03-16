import numpy as np
from gensim.models import KeyedVectors
import os

class Utils:
    def __init__(self):
        # Load GLOVE vectors
        self.corpus_dir = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Doc_corpus/'
        self.labels_dir = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Doc_label/'
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

    def load_seq(self):
        all_corpus = os.listdir(self.corpus_dir)
        sentences = []
        labels = []

        for file in all_corpus:

            # abrir o ficheiro do corpus
            corpus_path = self.corpus_dir + file
            with open(corpus_path) as reading:
                results = reading.readlines()
                for r in results:
                    sentences.append(r)

            # abrir o ficheiro das labels
            labels_path = self.labels_dir + file
            with open(labels_path) as reading:
                results = reading.readlines()
                for r in results:
                    labels.append(r)

        print("Sentences and labels read!")
        return sentences, labels
