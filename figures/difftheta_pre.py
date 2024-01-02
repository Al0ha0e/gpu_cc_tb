ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]

jump = [0,1,2,3,8,9,6,7]
xlabel = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

with open('./ycsb_difftheta.txt','r') as f:
    lines = f.readlines()
    olines = []
    oabrlines = []
    for i in range(0,9):
        oline = str(xlabel[i])
        oabrline = str(xlabel[i])
        for j in range(0,len(ccs)):
            line = lines[i * 10 + jump[j]]
            oline += ' ' + str(float(line.split(',')[4]))
        for j in range(2,len(ccs)):
            line = lines[i * 10 + jump[j]]
            oabrline += ' ' + str(float(line.split(',')[5]) / 1048576)
        olines.append(oline+'\n')
        oabrlines.append(oabrline+'\n')
    with open('./ycsb_difftheta_throughput.txt', 'w') as f1:
        f1.writelines(olines)
    with open('./ycsb_difftheta_abr.txt', 'w') as f1:
        f1.writelines(oabrlines)