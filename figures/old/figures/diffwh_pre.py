ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]

jump = [0,1,2,3,8,9,6,7]
xlabel = [8,16,32,64,128]

with open('./tpcc_no_diffwh.txt','r') as f:
    lines = f.readlines()
    olines = []
    oabrlines = []
    for i in range(0,5):
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
    with open('./tpcc_no_diffwh_throughput.txt', 'w') as f1:
        f1.writelines(olines)
    with open('./tpcc_no_diffwh_abr.txt', 'w') as f1:
        f1.writelines(oabrlines)

with open('./tpcc_pm_diffwh.txt','r') as f:
    lines = f.readlines()
    olines = []
    oabrlines = []
    for i in range(0,5):
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
    with open('./tpcc_pm_diffwh_throughput.txt', 'w') as f1:
        f1.writelines(olines)
    with open('./tpcc_pm_diffwh_abr.txt', 'w') as f1:
        f1.writelines(oabrlines)