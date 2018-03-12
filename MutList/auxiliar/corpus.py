# path_read = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/mycorpus/all.txt'
# path_write = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/mycorpus/train_final.txt'
# path_write_tsv= '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/mycorpus/mut.tsv'


path_read = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/mycorpus_novo/all.txt'
path_write = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/mycorpus_novo/abstract.txt'
path_write_tsv= '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/mycorpus_novo/mut.tsv'

days_file = open(path_read, 'r')
abstract = open(path_write, 'w')
mut = open(path_write_tsv, 'w')




with open(path_read) as reading:
    results = reading.readlines()
    breaker = 1
    for result in results:
        if result is "\n":  # new id
            breaker = 1
        else:
            if breaker == 1:
                split = result.split("|")
                id = split[0]
                text = split[2]
                text = text.rstrip()
                abstract.write(id+"\t"+text)
                breaker = breaker + 1

            elif breaker == 2:
                split = result.split("|")
                id = split[0]
                text = split[2]
                text = text.rstrip()
                abstract.write("\t" + text+"\n")
                breaker = breaker + 1

            else:
                mut.write(result)
                breaker = breaker + 1



days_file.close()
abstract.close()
mut.close()