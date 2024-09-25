import os
import subprocess

ccs = [
    # "gacco",
    # "gputx",
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

ycsb_rownum = 10485760
ycsb_txcnt = 1048576
warp_density = 0
block_size = 32
batch_size = ycsb_txcnt


# for k in range(10, 21, 2):
k = 20
for cc_scheme in ccs:
    # for i in range(2, 6):
    #     for j in range(1, 33):
    i = 5
    j = 32
    os.system(
        f"./out/test_ycsb ./dataset/ycsb_hc.txs 10485760 1048576 {cc_scheme} {i} {j} {1<<k} 0"
    )

# for w in range(0,10):
#     # for k in range(10, 21, 2):
#     k = 20
#     # for i in range(0, 6):
#     #     for j in range(1, 33):
#     i = 5
#     j = 32
#     for cc_scheme in ccs:
#         os.system(
#             f"./out/test_ycsb ./dataset/ycsb_difft_{w}.txs 10485760 1048576 {cc_scheme} {i} {j} {1<<k} 0"
#         )
#         os.system(
#             f"./out/test_ycsb ./dataset/ycsb_diffw_{w}.txs 10485760 1048576 {cc_scheme} {i} {j} {1<<k} 0"
#         )


# SB = [
#     # ['silo',1,32],
#     # ['tictoc',0,19],
#     # ['tictoc',1,6],
#     # ['tictoc',1,7],
#     # ['tictoc',2,8],
#     # ['tictoc',2,18],
#     ['tictoc',2,20],
#     # ['tictoc',3,10],
#     # ['slowto',3,25],
# ]

# for sb in SB:
#     os.system(f"./out/test_ycsb ./dataset/ycsb_hc.txs 10485760 1048576 {sb[0]} {sb[1]} {sb[2]} {1<<20} 0")