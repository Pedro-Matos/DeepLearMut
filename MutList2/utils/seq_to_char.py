from collections import defaultdict


corpus_path = '../corpus_char/train_small.txt'
labels_file = '../corpus_char/mut_small.tsv'

ids = []

dic_corpus = {}

with open(corpus_path) as reading:
    sentences = reading.readlines()
    for sent in sentences:
        sp = sent.split("\t")
        id = int(sp[0])
        dic = {id: sent}
        dic_corpus.update(dic)
        ids.append(id)

ids = list(set(ids))


dic_labels = defaultdict(list)
with open(labels_file) as reading:
    sentences = reading.readlines()
    for sent in sentences:
        sp = sent.split("\t")
        id = int(sp[0])
        dic_labels[id].append(sent)
'''
dic_chars = defaultdict(list)
for i in ids:
    all_labels = dic_labels[i]
    for lab in all_labels:
        # get offsets
        s = lab.split("\t")
        off_s = int(s[1])
        off_e = int(s[2])
        r = lab.split("\t")

        # get the corpus
        corp = dic_corpus.get(i)
        corp = corp.rstrip()
        corp_split = corp.split("\t")
        id = int(corp_split[0])
        corp_split = corp_split[1:]
        corp = corp_split[0] + " " + corp_split[1]

        # create array of characters and their respective classification
        c = list(corp)
        class_chars = []
        count = 0

        for count in range(len(c)-1):
            if count == off_s:
                class_chars.append('B')
            elif count > off_s and count <=off_e:
                class_chars.append('I')
            else:
                class_chars.append('O')

        dic_chars[id].append(class_chars)

        # divide by sentences
        # sent = []
        # dot = False
        # split_off = []
        # count = 0
        # for char in c:
        #     if char == ".":
        #         dot = True
        #
        #     elif char == " " and dot:
        #         dot = False
        #         split_off.append(count)
        #     count = count + 1
'''

dic_chars = defaultdict(list)
for i in ids:
    corp = dic_corpus[i]
    arr_off_start = []
    arr_off_end = []

    # get all labels for a corpus
    labels = dic_labels.get(i)
    for lab in labels:
        # get offset
        s = lab.split("\t")
        arr_off_start.append(int(s[1]))
        arr_off_end.append(int(s[2]))

    # get the corpus
    corp_split = corp.split("\t")
    corp_split = corp_split[1:]
    corp = corp_split[0] + " " + corp_split[1]
    #print(corp)

    # corpus não tem exemplos de mutações, é tudo characteres a '0'
    if arr_off_start == []:
       print("sem exemplos")

    else:
        print("com exemplos")
        # inicializar com tudo a 'O'

        #print(corp[arr_off_start[0]:arr_off_end[0]])

        # meter 'B' e 'I' nos sitios corretos. Assim o resto já está preenchido