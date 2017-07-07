

path_read = 'all_2.txt'
path_write = 'train_final.txt'
path_write_tsv= 'CEMP_mut.tsv'

days_file = open(path_read, 'r')
final = open(path_write, 'w')
final_tsv = open(path_write_tsv, 'w')

id_lido = 0
idx_family = 0
id_lido_antigo = -1

for line in days_file:


    line_splitted = line.split('|', 3)
    id_lido = line_splitted[0]

    if idx_family >= 2:
        cemp_splitted = line.split('\t', 5)
        id_lido = cemp_splitted[0]

    if id_lido != id_lido_antigo:
        idx_family = 0


    #print(id_lido)
    #print(idx_family)

    if idx_family >= 2:
        list_line = line.split('\t', 5)
        str_tmp = list_line[0]+"\t"+"A"+"\t"+list_line[1]+"\t"+list_line[2]+"\t"+list_line[3]+"\t"+list_line[4]+"\n"
        final_tsv.write(str_tmp)
    else:
        if len(line_splitted) == 3:
            if idx_family == 0:
                tmp = line_splitted[2].split("\n",1)
                final.write(id_lido+"\t"+tmp[0])
                idx_family = idx_family+1
            else:
                final.write("\t"+line_splitted[2])
                idx_family = idx_family + 1

    id_lido_antigo = id_lido

days_file.close()
final.close()
final_tsv.close()