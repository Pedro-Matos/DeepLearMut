import numpy as np


class PreProcessing:
    def __init__(self):
        # Load GLOVE vectors
        self.wordslist_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordsList.npy'
        self.wordsvector_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordVectors.npy'
        self.results_path = '../corpus/mycorpus/mut.tsv'
        self.data = []
        self.labels = []
        self.types = []

    # function to load the pre-processed words in glove dataset
    def load_glove(self):
        wordsList = np.load(self.wordslist_path)
        print('Loaded the word list!')
        wordsList = wordsList.tolist()  # Originally loaded as numpy array
        wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
        wordVectors = np.load(self.wordsvector_path)
        print('Loaded the word vectors!')
        return wordsList, wordVectors

    # function to load information about the mutations
    def load_mutations(self):
        # types = []
        # with open(self.results_path) as rp:
        #     results = rp.readlines()
        #     for result in results:
        #         content = result.split('\t')
        #         self.data.append(content[4])
        #         self.labels.append(content[5])
        #         types.append(content[5])
        #
        # types = set(types)
        # self.types = list(types)
        # print("Loaded the data!")
        #
        # return self.data, self.labels, self.types

        types = []
        tmp_dic = {}
        with open(self.results_path) as rp:
            results = rp.readlines()
            for result in results:
                content = result.split('\t')
                x = {content[4]:content[5]}
                tmp_dic.update(x)
                types.append(content[5])

        for k, v in tmp_dic.items():
            self.data.append(k)
            self.labels.append(v)

        types = set(types)
        self.types = list(types)
        print("Loaded the data!")

        return self.data, self.labels, self.types

