import glob

path_corpus = '../corpus_char/tmVarCorpus/treated/test_a1_gold/'
docs = glob.glob(path_corpus + "*.a1")
print(len(docs))


path_corpus_silver = '../model/silver_minibatch_10epoch/'
docs_silver = glob.glob(path_corpus_silver + "*.a1")
print(len(docs_silver))


for i in range(len(docs_silver)):
    print(docs[i].split("/")[-1])
    print(docs_silver[i].split("/")[-1])
    print("----")