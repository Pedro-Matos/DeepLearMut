from utils import wordUtils

class WordModel:
    def __init__(self):
        self.maxSeqLength = -1  # Maximum length of sentence
        self.numDimensions = 200  # Dimensions for each word vector



    def main(self):
        utils = wordUtils.Utils()
        words_list, embedding_matrix = utils.load_word2vec()
        sentences, labels = utils.load_seq()

if __name__ == "__main__":
    model = WordModel()
    model.main()
