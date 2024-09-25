ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]

def process(pref):
    othrough_lines = []
    oabr_lines = []


    for i in range(0,10):
        cur_through = f'0.{i}'
        cur_abr = f'0.{i}'
        with open(f"./dataset/ycsb_diff{pref}_{i}.txs.txt",'r') as f: 
            lines = f.readlines()
            for j in range(0,len(ccs)):
                print(ccs[j])
                line = lines[j].split(',')
                cur_through += ' ' + str(float(line[4]))
                if j > 1:
                    cur_abr += ' ' + str(float(line[5]) / 1048576)
        othrough_lines += cur_through + '\n'
        if pref == 'w' and i == 0:
            pass
        else:
            oabr_lines += cur_abr + '\n'
    with open(f'./figures/mid/ycsb_diff{pref}_throughput.txt', 'w') as f1:
        f1.writelines(othrough_lines)
    with open(f'./figures/mid/ycsb_diff{pref}_abr.txt', 'w') as f1:
        f1.writelines(oabr_lines)
    

process('w')
process('t')