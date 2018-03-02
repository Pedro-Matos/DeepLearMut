from preprocess import PreProcess

class Mutlist:
    def __init__(self):
        self.words2vec = None
        self.maxSeqLength = 1  # Maximum length of sentence
        self.numDimensions = 300  # Dimensions for each word vector
        self.types = {}
        self.batchSize = 24
        self.numClasses = -1
        self.lstmUnits = 64
        self.iterations = 100000

    def main(self):
        preprocess = PreProcess()
        self.words2vec = preprocess.load_word2vec()
        wordsList, wordVectors = preprocess.load_glove()
        print(wordVectors.shape)
        print(wordVectors[20])


if __name__ == "__main__":
    mutlist = Mutlist()
    mutlist.main()
