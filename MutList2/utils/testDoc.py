import os
from collections import defaultdict
from keras.preprocessing.text import text_to_word_sequence

corpus_path = '../corpus/Doc_corpus'
labels_path = '../corpus/Doc_label'
write_path = 'corpus_test'

# array of files
all_files = os.listdir(corpus_path)
files = []
for file in all_files:
    files.append(file)

corpus = []
# load all corpus
for file in files:
    file_path = corpus_path + "/" + file
    with open(file_path) as reading:
        sentences = reading.readlines()
        for sent in sentences:
            corpus.append(sent)


# load all labels
labels = []
for file in files:
    file_path = labels_path + "/" + file
    with open(file_path) as reading:
        sentences = reading.readlines()
        for sent in sentences:
            labels.append(sent)

# see if the lengths are correct at each
for i in range(len(corpus)):
    # split com keras das frases por whitespace
    corpus[i] = corpus[i].rstrip()
    s = corpus[i].split()

    # split com das labels por ","
    labels[i] = labels[i].rstrip()
    l = labels[i].split(",")

    if len(s) != len(l):
        print("error")

m_classes = []
# get all the classes
for i in range(len(labels)):
    l = labels[i].split(",")
    for idx in l:
        m_classes.append(idx)

m_classes = list(set(m_classes))

print(m_classes)

# criar o ficheiro para ter as labels
X = write_path + "/" + "data.txt"
y = write_path + "/" + "labels.txt"
data = open(X, 'w')
lab = open(y, 'w')
count = 0
for i in range(len(corpus)):
    # split com keras das frases por whitespace
    corpus[i] = corpus[i].rstrip()
    s = corpus[i].split()

    # split com das labels por ","
    labels[i] = labels[i].rstrip()
    l = labels[i].split(",")
    check = False
    for tmp in l:
        if tmp != '0':
            check = True



    if check:
        count = count + 1
        l = labels[i].split(",")
        data.write(corpus[i] + "\n")
        lab.write(labels[i] + "\n")

