ccs = [
    "tpl_nw",
    "tpl_wd",
    "to",
    "mvcc",
    "silo",
    "tictoc",

    "slow_tpl_nw",
    "slow_tpl_wd",
    "slow_to",
    "slow_mvcc",
    "slow_silo",
    "slow_tictoc",
]

def process(name):
    othrough_lines = ['free latch\n']
    with open(f'dataset/ycsb_{name}_latch.txt', 'r') as f:
        lines = f.readlines()
        for i in range(0,len(ccs) // 2):
            line1 = lines[i].split(',')
            line2 = lines[i + len(ccs) // 2].split(',')
            cur_through = str(float(line1[4])) + ' ' + str(float(line2[4]))
            othrough_lines.append(cur_through + '\n')
    
    with open(f'./figures/mid/ycsb_latch_{name}.txt', 'w') as f1:
        f1.writelines(othrough_lines)



process('ro')
process('mc')
process('hc')