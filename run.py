import os
import subprocess

ccs = [
    "gacco",
    "gputx",
    "tpl_nw",
    "tpl_wd",
    "to",
    "mvcc",
    "silo",
    "tictoc",
    "slowto",
    "slowmvcc",
]

for cc_scheme in ccs:
    for k in range(10,21,2):
        for i in range(0,6):
            for j in range(1,33):
                    os.system(f"./test_ycsb ycsb_hc.txs 10485760 1048576 {cc_scheme} {i} {j} {1<<k} 0")
                    os.system(f"./test_tpcc 0 40 65536 {cc_scheme} {i} {j} {1<<k} 0")
                    os.system(f"./test_tpcc 1 40 65536 {cc_scheme} {i} {j} {1<<k} 0")

