import os
from collections import defaultdict


docs_path = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Documents'
corpus_path = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Doc_corpus'
labels_path = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Doc_label'

all_files = os.listdir(docs_path)

# dictionaries to save the data
dic_corpus = {}
dic_labels = defaultdict(list)

# read one by one
for file in all_files:
    file_type = file.split(".")
    if file_type[1] == "txt":
        file_path = docs_path + "/" + file
        count_line = 0

        with open(file_path) as reading:
            results = reading.readlines()
            id = -1
            for result in results:
                if count_line == 0:
                    corpus = result.split("\t")
                    id = corpus[0]
                    text = corpus[1]+" "+corpus[2]

                    # dividi by sentence
                    sentences = text.split(". ")

                    # create the file
                    p = corpus_path + "/" + str(id) + ".txt"
                    tmp_doc_corpus = open(p, 'w')

                    tmp_dic = {id:sentences}
                    dic_corpus.update(tmp_dic)

                    #print the sentences; one by line
                    for i in sentences:
                        i = i.rstrip()
                        tmp_doc_corpus.write(i+"."+"\n")

                else:
                    id = int(id)
                    dic_labels[id].append(result)
                count_line = count_line + 1

dic_token = defaultdict(list)
# ir documento a documento e obter a lista dos tokens que são mutações
for idx, label in dic_labels.items():
    # ir token a token
    for l in label:
        split_l = l.split("\t")
        dic_token[idx].append(split_l[3])

# now the hardest part. split each sentence from corpus by whitespaces to get all the words
# then see if that word is one of the tokens
# create the file with the labels

for idx, sentences in dic_corpus.items():
    # ir buscar o os tokens
    idx = int(idx)
    tokens = dic_token.get(idx)

    # criar o ficheiro para ter as labels
    p = labels_path + "/" + str(idx) + ".txt"
    tmp_doc_labels = open(p, 'w')


    # ir frase a frase; em cada frase fazer o split por whitespace
    for sent in sentences:
        word = sent.split()
        arr = []
        for w in word:
            if w in tokens:
                arr.append(1)
            else:
                arr.append(0)


        # escrever no ficheiro
        max = len(arr)
        i = 0
        for label in arr:
            if i < (max -1):
                tmp_doc_labels.write(str(label)+ ",")
            else:
                tmp_doc_labels.write(str(label))
            i = i + 1
        tmp_doc_labels.write("\n")






