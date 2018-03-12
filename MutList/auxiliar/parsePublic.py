from collections import defaultdict


path_emu_abs = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/PublicCorpus/corpus[EMU]_abstracts.txt'
path_emu_labels = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/PublicCorpus/corpus[EMU]_answers.txt'


dic_emu_abs = {}
with open(path_emu_abs) as emu_abs:
    results = emu_abs.readlines()
    for result in results:
        content = result.split('\t')
        text = content[1] + "\t" + content[2]
        tmp_d = {content[0]:text}
        dic_emu_abs.update(tmp_d)

dic_emu_labels = defaultdict(list)
with open(path_emu_labels) as emu_labels:
    results = emu_labels.readlines()
    for result in results:
        content = result.split('\t')
        dic_emu_labels[content[0]].append(content[1:])


path_tmVar_abs = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/PublicCorpus/corpus[tmVar]_abstracts.txt'
path_tmVar_labels = '/Users/pmatos9/Desktop/pedrinho/tese/DeepLearMut/MutList/corpus/PublicCorpus/corpus[tmVar]_answers.txt'


dic_abs = {}
with open(path_tmVar_abs) as mf_abs:
    results = mf_abs.readlines()
    for result in results:
        content = result.split('\t')
        text = content[1] + "\t" + content[2]
        tmp_d = {content[0]:text}
        dic_abs.update(tmp_d)

dic_labels = defaultdict(list)
with open(path_tmVar_labels) as tm_labels:
    results = tm_labels.readlines()
    for result in results:
        content = result.split('\t')
        dic_labels[content[0]].append(content[1:])

#join both
with open('tmVar.txt', 'w') as file: # the_file.write('Hello\n')
    for id in dic_abs.keys():
        value_abs = dic_abs.get(id)
        value_labels = dic_labels.get(id)
        if value_labels is not None:
            file.write(id)
            split = value_abs.split('\t')
            file.write("|t|")
            file.write(split[0])
            file.write("\n")
            file.write(id)
            file.write("|a|")
            file.write(split[1])
            for l in value_labels:
                file.write(id)
                for i in l:
                    file.write("\t"+i)

            file.write("\n")

#join both
with open('EMU.txt', 'w') as the_file: # the_file.write('Hello\n')
    for id in dic_emu_abs.keys():
        value_abs = dic_emu_abs.get(id)
        value_labels = dic_emu_labels.get(id)
        the_file.write(id)
        split = value_abs.split('\t')
        the_file.write("|t|")
        the_file.write(split[0])
        the_file.write("\n")
        the_file.write(id)
        the_file.write("|a|")
        the_file.write(split[1])
        for l in value_labels:
            the_file.write(id)
            for i in l:
                the_file.write("\t"+i)

        the_file.write("\n")


#both of them
with open('final.txt', 'w') as the_file: # the_file.write('Hello\n')
    for id in dic_emu_abs.keys():
        value_abs = dic_emu_abs.get(id)
        value_labels = dic_emu_labels.get(id)
        the_file.write(id)
        split = value_abs.split('\t')
        the_file.write("|t|")
        the_file.write(split[0])
        the_file.write("\n")
        the_file.write(id)
        the_file.write("|a|")
        the_file.write(split[1])
        for l in value_labels:
            the_file.write(id)
            for i in l:
                the_file.write("\t"+i)

        the_file.write("\n")

    for id in dic_abs.keys():
        value_abs = dic_abs.get(id)
        value_labels = dic_labels.get(id)
        if value_labels is not None:
            the_file.write(id)
            split = value_abs.split('\t')
            the_file.write("|t|")
            the_file.write(split[0])
            the_file.write("\n")
            the_file.write(id)
            the_file.write("|a|")
            the_file.write(split[1])
            for l in value_labels:
                the_file.write(id)
                for i in l:
                    the_file.write("\t"+i)

            the_file.write("\n")


