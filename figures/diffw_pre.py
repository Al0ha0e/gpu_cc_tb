ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]

xlabel = [0,0.1,0.2,0.3,0.4,0.5]

with open('./ycsb_diffw.txt','r') as f:
    lines = f.readlines()
    olines = []
    oabrlines = []
    for i in range(0,6):
        oline = str(xlabel[i])
        oabrline = str(xlabel[i])
        for j in range(0,len(ccs)):
            line = lines[i * 10 + j]
            oline += ' ' + str(float(line.split(',')[4]))
        for j in range(2,len(ccs)):
            line = lines[i * 10 + j]
            oabrline += ' ' + str(float(line.split(',')[5]) / 1048576)
        olines.append(oline+'\n')
        oabrlines.append(oabrline+'\n')
    with open('./ycsb_diffw_throughput.txt', 'w') as f1:
        f1.writelines(olines)
    with open('./ycsb_diffw_abr.txt', 'w') as f1:
        f1.writelines(oabrlines)