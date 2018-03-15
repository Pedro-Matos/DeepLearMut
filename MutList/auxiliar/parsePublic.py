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
        text = [content[1],content[2],content[3],content[5],content[6],content[7]]
        dic_emu_labels[content[0]].append(text)


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
        text = [content[1], content[2], content[3], content[5], content[6], content[7]]
        dic_labels[content[0]].append(text)



#both of them
with open('final.txt', 'w') as the_file: # the_file.write('Hello\n')
    for id in dic_emu_abs.keys():
        value_abs = dic_emu_abs.get(id)
        value_labels = dic_emu_labels.get(id)
        the_file.write(id)
        split = value_abs.split('\t')
        the_file.write("|t|")
        text = split[0].rstrip()
        the_file.write(text)
        the_file.write("\n")
        the_file.write(id)
        the_file.write("|a|")
        text = split[1].rstrip()
        the_file.write(text)
        the_file.write("\n")
        for l in value_labels:
            the_file.write(id)
            for i in l:
                the_file.write("\t"+i)
            the_file.write("\n")
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
            text = split[1].rstrip()
            the_file.write(text)
            the_file.write("\n")
            for l in value_labels:
                the_file.write(id)
                for i in l:
                    the_file.write("\t"+i)
                the_file.write("\n")
            the_file.write("\n")


