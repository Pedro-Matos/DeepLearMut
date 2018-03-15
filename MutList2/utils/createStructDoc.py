import os

docs_path = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Documents'
corpus_path = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Doc_corpus'
all_files = os.listdir(docs_path)

# read one by one
for file in all_files:
    file_type = file.split(".")
    if file_type[1] == "txt":
        file_path = docs_path + "/" + file
        count_line = 0

        with open(file_path) as reading:
            results = reading.readlines()
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

                    #print the sentences; one by line
                    for i in sentences:
                        tmp_doc_corpus.write(i+"\n")

                else:
                    print(result)
                count_line = count_line + 1

