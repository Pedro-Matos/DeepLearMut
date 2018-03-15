all_path = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/all.txt'
path_doc = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList2/corpus/Documents/'

days_file = open(all_path, 'r')
count = 0

with open(all_path) as reading:
    tmp_doc = path_doc + "doc" + str(count) + ".txt"
    doc = open(tmp_doc, 'w')
    results = reading.readlines()
    breaker = 1

    for result in results:
        if result is "\n":  # new document
            breaker = 1
            count = count + 1
            tmp_doc = path_doc + "doc" + str(count) + ".txt"
            doc = open(tmp_doc, 'w')

        else:
            if breaker == 1:

                split = result.split("|")
                id = split[0]

                try:
                    text = split[2]
                except :
                    print(split)
                text = text.rstrip()
                doc.write(id+"\t"+text)
                breaker = breaker + 1

            elif breaker == 2:
                split = result.split("|")
                id = split[0]
                text = split[2]
                text = text.rstrip()
                doc.write("\t" + text+"\n")
                breaker = breaker + 1

            else:
                doc.write(result)
                breaker = breaker + 1



