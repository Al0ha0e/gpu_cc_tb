with open('tpcc_no_batchsize_re.txt','r') as f:
    lines = f.readlines()
    olines = []
    for i in range(0,6):
        oline = str(10 + i * 2)
        for j in range(0,6):
            line = lines[i  + j*6].split(',')
            oline += ' ' + str(int(line[-1]) / 1048576)
        olines.append(oline + '\n')

    with open('tpcc_no_batchsize_abr.txt','a') as f1:
        f1.writelines(olines)