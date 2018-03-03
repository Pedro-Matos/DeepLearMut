import numpy as np
from gensim.models import KeyedVectors
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


class PreProcess:
    def __init__(self):
        # Load GLOVE vectors
        self.wordslist_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordsList.npy'
        self.wordsvector_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordVectors.npy'
        self.text_path = '../corpus/mycorpus/train_final.txt'
        self.results_path = '../corpus/mycorpus/mut.tsv'
        self.average_words = 0
        self.word2vec_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wikipedia-pubmed-and-PMC-w2v.bin'

    # function to load the pre-processed words in glove dataset
    def load_glove(self):
        wordsList = np.load(self.wordslist_path)
        print('Loaded the word list!')
        wordsList = wordsList.tolist()  # Originally loaded as numpy array
        wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
        wordVectors = np.load(self.wordsvector_path)
        print('Loaded the word vectors!')
        return wordsList, wordVectors

    # function to load the information about the mutations
    def load_mutations(self):
        # create dictionary with identifier and the text(t + a) as 1st step
        list_id = []
        numwords = []

        tokenizer = RegexpTokenizer(r'\w+')
        regex_dic = {}

        with open(self.text_path) as fp:
            lines = fp.readlines()
            for line in lines:
                content = line.split('\t')
                id = content[0]
                text = content[1] + " " + content[2]

                list_id.append(id)

                regex_text = tokenizer.tokenize(text)
                tmp = {id: regex_text}
                regex_dic.update(tmp)

                counter = len(regex_text)
                numwords.append(counter)

        self.average_words = max(numwords)

        print("Loaded the mutations texts!")


        list_types = []
        res_dict = defaultdict(list)

        with open(self.results_path) as rp:
            results = rp.readlines()
            for result in results:
                content = result.split('\t')
                id = content[0]
                list_types.append(content[5])
                res_dict[id].append(content[5])

        print("Loaded the mutations results")
        list_types = set(list_types)
        list_types = list(list_types)

        return regex_dic, list_id, res_dict, list_types, self.average_words

    # function to load pre-processed words in word2vec from a combination of PubMed and PMC texts
    def load_word2vec(self):
        wordVectors = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True, limit=500000)  # limit just by now to speed up the run time

        print('Found %s word vectors of word2vec' % len(wordVectors.vocab))

        wordsList = wordVectors.index2word
        embedding_matrix = np.zeros((500000, 200))
        for i in range(len(wordsList)):
            word = wordsList[i]
            embedding_matrix[i] = wordVectors.word_vec(word)

        return wordsList, embedding_matrix
