import glob
from collections import defaultdict

class DataToTest:
    def __init__(self):
        self.path_corpus = '../corpus_char/tmVarCorpus/treated/test_data2/'
        self.dic_corpus = defaultdict(list)

    def split_seqs(self, sentence):
            seqs = []

            c = list(sentence)

            #size_doc = len(c)
            #print(corp)

            # divide by sentences
            dot = False
            split_off = []
            count = 0
            for char in c:
                if char == ".":
                    dot = True

                elif char == " " and dot:
                    dot = False
                    split_off.append(count)
                else:
                    dot = False

                count = count + 1

            #print(split_off)
            # splitting the character labels

            first = c[:split_off[0]]
            i = len(split_off) - 1
            last = c[split_off[i]:]

            for idx in split_off:
                tmp_a = [(split_off[i], split_off[i + 1]) for i in range(len(split_off) - 1)]
            middle = []

            for tup in tmp_a:
                b = tup[0]
                e = tup[1]
                middle.append(c[b:e])

            seqs.append(first)
            for i in middle:
                seqs.append(i)
            seqs.append(last)

            return seqs


    def get_content(self):
        docs = glob.glob(self.path_corpus + "*.txt")

        for i in docs:
            with open(i) as reading:
                sentences = reading.read()

                id = i.split("/")[-1]
                id = id.split(".")[0]
                seqs = self.split_seqs(sentences)
                self.dic_corpus[id] = seqs


    def get_testset(self):
        self.get_content()
        return self.dic_corpus

if __name__ == "__main__":
    reader = DataToTest()
    reader.get_testset()
