from utils import corpusreader

textfile = '../cc/train_data.txt'
annotfile = '../cc/train_labels.tsv'

cr = corpusreader.CorpusReader(textfile, annotfile)
seqs = cr.trainseqs

for i in seqs:
    print(i)