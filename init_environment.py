import os

# whcnt = 128
# tpcc_txcnt = 1048576

ycsb_rownum = 10485760
ycsb_txcnt = 1048576


# tpcc_txcnt = 65536
# ycsb_rownum = 65536
# ycsb_txcnt = 65536 * 4
# whcnt = 16

cmds = [
    # "nvcc ./generation/gen_tpcc.cu -I ./include/ -o ./out/gen_tpcc",
    # "nvcc ./generation/gen_ycsb.cu -I ./include/ -o ./out/gen_ycsb",
    # "nvcc ./generation/gen_tpcc_table.cu -I ./include/ -o ./out/gen_tpcc_table",
    # "nvcc -lnvrtc -lcuda -arch=sm_89 ./benchmark_cpu/tpcc.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o ./out/test_tpcc -I ./include",
    # "nvcc -lnvrtc -lcuda -arch=sm_89 ./benchmark_cpu/ycsb.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o ./out/test_ycsb -I ./include",
    # f"./out/gen_tpcc_table 0 {whcnt} {tpcc_txcnt}",
    # f"./out/gen_tpcc {whcnt} {tpcc_txcnt} 0 0.1",
    # f"./out/gen_tpcc {whcnt} {tpcc_txcnt} 1 0.1",
    # f"./out/gen_ycsb {ycsb_rownum} {ycsb_txcnt} 0 1 ./dataset/ycsb_ro.txs",
    # f"./out/gen_ycsb {ycsb_rownum} {ycsb_txcnt} 0.6 0.9 ./dataset/ycsb_mc.txs",
    # f"./out/gen_ycsb {ycsb_rownum} {ycsb_txcnt} 0.8 0.5 ./dataset/ycsb_hc.txs",

]

for t in range(0,10):
    cmds.append(f"./out/gen_ycsb {ycsb_rownum} {ycsb_txcnt} {t/10} 0.9 ./dataset/ycsb_difft_{t}.txs")

for w in range(0,10):
    cmds.append(f"./out/gen_ycsb {ycsb_rownum} {ycsb_txcnt} 0.6 {1 - w/10} ./dataset/ycsb_diffw_{w}.txs")

for cmd in cmds:
    print(cmd)
    os.system(cmd)
