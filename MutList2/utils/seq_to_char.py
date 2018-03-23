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

for i in ids:
    all_labels = dic_labels[i]
    for lab in all_labels:
        # get offsets
        s = lab.split("\t")
        off_s = int(s[1])
        off_e = int(s[2])
        print(off_s)
        print(off_e)
        print(lab.rstrip())

        #get the corpus
        corp = dic_corpus.get(i)
        corp = corp.rstrip()
        corp_split = corp.split("\t")
        corp_split = corp_split[1:]
        corp = corp_split[0] + " " + corp_split[1]
        print(corp[off_s:off_e])
        print("\n")
