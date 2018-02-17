import numpy as np


class PreProcess:
    def __init__(self):
        # Load GLOVE vectors
        self.wordslist_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordsList.npy'
        self.wordsvector_path = '/Users/pmatos9/Desktop/pedrinho/tese/glove/wordVectors.npy'
        self.wordsList = [] #list of all the words in the Glove. This way, we have a list and then, with the index we can
                            # access the vector and retrieve the information
        self.wordVectors = []
        self.text_path = '../corpus/mycorpus/train_final.txt'
        self.dic_text = {}
        self.list_id = []

    def main(self):
        self.load_glove()
        # print(len(self.wordsList))
        # print(self.wordVectors.shape)
        self.load_mutations()

    # function to load the pre-processed words in glove dataset
    def load_glove(self):
        self.wordsList = np.load(self.wordslist_path)
        print('Loaded the word list!')
        self.wordsList = self.wordsList.tolist()  # Originally loaded as numpy array
        self.wordsList = [word.decode('UTF-8') for word in self.wordsList]  # Encode words as UTF-8
        self.wordVectors = np.load(self.wordsvector_path)
        print('Loaded the word vectors!')

    # function to load the information about the mutations
    def load_mutations(self):
        # create dictionary with identifier and the text(t + a) as 1st step
        with open(self.text_path) as fp:
            lines = fp.readlines()
            for line in lines:
                content = line.split('\t')
                id = content[0]
                text = content[1] + "\t" + content[2]
                dict = {id : text}
                self.dic_text.update(dict)
                self.list_id.append(id)

        print("Loaded the mutations texts!")





if __name__ == "__main__":
    pre = PreProcess()
    pre.main()

