
# train dataset

path_read = 'test.PubTator.txt'
path_write = 'treated/test_data.txt'
path_write_tsv= 'treated/test_labels.tsv'

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

                try:
                    text = split[2]
                except :
                    print(split)
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
