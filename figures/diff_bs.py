ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]

def process(name):
    othrough_lines = []
    oabr_lines = []
    with open(f'dataset/ycsb_{name}.txs.txt', 'r') as f:
        lines = f.readlines()
        
        for i in range(1,33):
            cur_through = str(i)
            cur_abr = str(i)
            for j in range(0, len(ccs)):
                print(ccs[j])
                lineid = (j * 6 + 5) * 32 + i - 1
                line = lines[lineid].split(',')
                cur_through += ' ' + str(float(line[4]))
                if j > 1:
                    cur_abr += ' ' + str(float(line[5]) / 1048576)
            othrough_lines += cur_through + '\n'
            oabr_lines += cur_abr + '\n'

    with open(f'./figures/mid/ycsb_diffbs_{name}_throughput.txt', 'w') as f1:
        f1.writelines(othrough_lines)
    if name != 'ro':
        with open(f'./figures/mid/ycsb_diffbs_{name}_abr.txt', 'w') as f1:
            f1.writelines(oabr_lines)



process('ro')
process('mc')
process('hc')