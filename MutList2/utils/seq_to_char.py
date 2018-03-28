from collections import defaultdict





'''
for i in dic_chars:
    print(dic_chars.get(i)[0])
'''

class CorpusReader:
    def __init__(self):
        self.corpus_path = '../corpus_char/train_small.txt'
        self.labels_file = '../corpus_char/mut_small.tsv'

        self.ids = []
        self.dic_corpus = {}
        self.dic_labels = defaultdict(list)
        self.dic_labels_chars = defaultdict(list)
        self.dic_corpus_char = defaultdict(list)

    def readcorpus(self):
        with open(self.corpus_path) as reading:
            sentences = reading.readlines()
            for sent in sentences:
                sp = sent.split("\t")
                id = int(sp[0])
                dic = {id: sent}
                self.dic_corpus.update(dic)
                self.ids.append(id)

        self.ids = list(set(self.ids))

    def readlabels(self):
        with open(self.labels_file) as reading:
            sentences = reading.readlines()
            for sent in sentences:
                sp = sent.split("\t")
                id = int(sp[0])
                self.dic_labels[id].append(sent)

    def create_char_seqs(self):
        for id in self.ids:
            corp = self.dic_corpus[id]
            arr_off_start = []
            arr_off_end = []
            result = []

            # get all labels for a corpus
            labels = self.dic_labels.get(id)
            if labels != None:
                for lab in labels:
                    # get offset
                    s = lab.split("\t")
                    arr_off_start.append(int(s[1]))
                    arr_off_end.append(int(s[2]))
                    result.append(s[3])

                # get the corpus
                corp_split = corp.split("\t")
                corp_split = corp_split[1:]
                corp = corp_split[0] + " " + corp_split[1]

                c = list(corp)
                class_chars = []

                # inicializar com tudo a 'O'
                for count in range(len(c)):
                    class_chars.append('O')

                # meter 'B' e 'I' nos sitios corretos. Assim o resto já está preenchido
                for i in range(len(arr_off_start)):
                    start = arr_off_start[i]
                    end = arr_off_end[i]

                    for counter in range(start, end + 1):
                        if counter == start:
                            #class_chars.insert(counter, 'B')

                            arr1 = class_chars[:counter]
                            arr1.append('B')
                            arr1 = arr1 + class_chars[counter + 1:]

                            class_chars = arr1

                        elif counter > start and counter <= end:
                            #class_chars.insert(counter, 'I')

                            arr1 = class_chars[:counter]
                            arr1.append('I')
                            arr1 = arr1 + class_chars[counter + 1:]

                            class_chars = arr1

                    # print(result[i])
                    # print(corp[start:end])
                    # print(c[start:end])
                    # print(class_chars[start:end])

                #print(len(c) == len(class_chars))
                self.dic_labels_chars[id].append(class_chars)

            else:
                # get the corpus
                corp_split = corp.split("\t")
                corp_split = corp_split[1:]
                corp = corp_split[0] + " " + corp_split[1]

                # corpus não tem exemplos de mutações, é tudo characteres a '0'
                c = list(corp)
                class_chars = []

                for count in range(len(c)):
                    class_chars.append('O')

                self.dic_labels_chars[id].append(class_chars)

    def split_seqs(self):
        for id in self.ids:
            #print("------")
            # get the corpus
            corp = self.dic_corpus[id]
            corp_split = corp.split("\t")
            corp_split = corp_split[1:]
            corp = corp_split[0] + " " + corp_split[1]
            c = list(corp)
            #print("corpus:" + str(len(c)))
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
                count = count + 1

            # splitting the character labels
            class_chars = self.dic_labels_chars[id][0]
            #print("labels: " + str(len(class_chars)))
            first = class_chars[:split_off[0]]
            i = len(split_off) - 1
            last = class_chars[split_off[i]:]

            for idx in split_off:
                tmp_a = [(split_off[i], split_off[i + 1]) for i in range(len(split_off) - 1)]
            middle = []

            for tup in tmp_a:
                b = tup[0]
                e = tup[1]
                middle.append(class_chars[b:e])

            all_chars = []
            all_chars.append(first)
            for i in middle:
                all_chars.append(i)
            all_chars.append(last)

            del self.dic_labels_chars[id]
            for chars in all_chars:
                self.dic_labels_chars[id].append(chars)

            # splitting the corpus by character
            first = c[:split_off[0]]
            i = len(split_off) - 1
            last = c[split_off[i]:]

            tmp_a = [(split_off[i], split_off[i + 1]) for i in range(len(split_off) - 1)]
            middle = []


            for tup in tmp_a:
                b = tup[0]
                e = tup[1]
                middle.append(c[b:e])

            all_chars = []
            all_chars.append(first)
            for i in middle:
                all_chars.append(i)
            all_chars.append(last)

            for chars in all_chars:
                self.dic_corpus_char[id].append(chars)

    def merged_data(self):
        merged = []
        count = 0

        for id in self.dic_corpus_char:
            corpus = self.dic_corpus_char[id]
            labels = self.dic_labels_chars[id]
            for idx in range(len(corpus)):
                if len(corpus[idx]) == len(labels[idx]):
                    new_dic = {"corpus":corpus[idx], "labels":labels[idx]}
                    merged.append(new_dic)

                count = count + 1

        print("\n\n\n")
        print(count)
        print(len(merged))


    def read(self):
        self.readcorpus()
        self.readlabels()
        self.create_char_seqs()
        self.split_seqs()
        self.merged_data()
        return self.dic_corpus_char, self.dic_labels_chars



if __name__ == "__main__":
    reader = CorpusReader()
    reader.read()
