
# train dataset
'''
path_read = 'train.PubTator.txt'
path_write = 'treated/train_data.txt'
path_write_tsv= 'treated/train_labels.tsv'

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
'''

# test dataset
from collections import defaultdict

path_read = 'test.PubTator.txt'
path_write = 'treated/test_data/'
path_write_tsv = 'treated/test_a1/'


days_file = open(path_read, 'r')
#abstract = open(path_write, 'w')
#mut = open(path_write_tsv, 'w')

dic_labels = defaultdict(list)
all_ids = []
with open(path_read) as reading:
    results = reading.readlines()
    breaker = 1
    data = []


    for result in results:
        if result is "\n":  # new id
            breaker = 1
            data = []

        else:
            if breaker == 1:

                split = result.split("|")
                id = split[0]

                try:
                    text = split[2]
                except :
                    print(split)
                text = text.rstrip()
                data.append(text)

                breaker = breaker + 1

            elif breaker == 2:
                split = result.split("|")
                id = split[0]
                text = split[2]
                text = text.rstrip()

                data.append(text)

                path = path_write + id + ".txt"
                all_ids.append(id)
                abstract = open(path, 'w')
                str = data[0] + " " + data[1]
                abstract.write(str)
                breaker = breaker + 1

            else:
                #mut.write(result)

                split = result.split("\t")
                id = split[0]
                dic_labels[id].append(split)
                breaker = breaker + 1


for key in dic_labels:
    id = key
    path = path_write_tsv + id + ".a1"
    a1 = open(path, 'w')
    array = dic_labels.get(key)

    for label in array:
        mut = label[1:4]
        mut_str = mut[0] + "\t" + mut[1] + "\t" + mut[2] + "\n"
        a1.write(mut_str)


ids_with_labels = dic_labels.keys()

left2 = [i for i in all_ids if i not in ids_with_labels]
left2 = list(set(left2))

for i in left2:
    path = path_write_tsv + i + ".a1"
    a1 = open(path, 'w')
    a1.write("")

days_file.close()

