import numpy as np



class PreProcess:
    def __init__(self):
        # Load GLOVE vectors
        self.wordslist_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordsList.npy'
        self.wordsvector_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordVectors.npy'
        self.text_path = '../corpus/mycorpus/train_final.txt'
        self.results_path = '../corpus/mycorpus/mut.tsv'
        self.average_words = 0

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
        dic_text = {}
        list_id = []
        numwords = []

        with open(self.text_path) as fp:
            lines = fp.readlines()
            for line in lines:
                content = line.split('\t')
                id = content[0]
                text = content[1] + "\t" + content[2]
                dict = {id: text}
                dic_text.update(dict)
                list_id.append(id)

                counter = len(content[1].split(" ")) + len(content[2].split(" "))
                numwords.append(counter)

        self.average_words = sum(numwords)/len(numwords)

        print("Loaded the mutations texts!")

        dic_results = {}
        list_types = []

        with open(self.results_path) as rp:
            results = rp.readlines()
            for result in results:
                content = result.split('\t')
                id = content[0]
                r_list = content[1:]
                list_types.append(content[5])
                dic = {id: r_list}
                dic_results.update(dic)

        print("Loaded the mutations results")
        list_types = set(list_types)
        list_types = list(list_types)

        return dic_text, list_id, dic_results, list_types

#pre = PreProcess()
#pre.load_mutations()
