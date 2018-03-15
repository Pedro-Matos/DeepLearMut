from collections import defaultdict

path_read = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/auxiliar/example.txt'

data_file = open(path_read, 'r')
count = 0
data_abstracts = []
data_results = defaultdict(list)
corpus = ""

with data_file as reading:
    results = reading.readlines()
    breaker = 1
    for result in results:
        if result is "\n":  # new id
            breaker = 1
            count = count + 1
            corpus = ""
        else:
            if breaker == 1:
                split = result.split("|")
                id = split[0]
                text = split[2]
                text = text.rstrip()
                corpus = corpus + str(count) + "\t"+text
                breaker = breaker + 1

            elif breaker == 2:
                split = result.split("|")
                id = split[0]
                text = split[2]
                text = text.rstrip()
                corpus = corpus + "\t" + text + "\n"
                breaker = breaker + 1
                data_abstracts.append(corpus)

            else:
                #print(result)
                data_results[count].append(result)
                breaker = breaker + 1


for i in range(len(data_abstracts)):
    # get the corpus from this id
    tmp_corpus = data_abstracts[i]
    # get all the results from each abstract
    tmp_list_abs = data_results.get(i)
    for j in range(len(tmp_list_abs)):
        split_tabs = tmp_list_abs[j].split("\t")
        offset_1 = split_tabs[1]
        offset_2 = split_tabs[2]
        print("%s   %s" % (str(offset_1), str(offset_2)))



data_file.close()