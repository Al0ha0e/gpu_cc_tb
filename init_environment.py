import os

print("nvcc ./generation/gen_tpcc.cu -I ./include/ -o gen_tpcc")
os.system("nvcc ./generation/gen_tpcc.cu -I ./include/ -o gen_tpcc")
print("nvcc ./generation/gen_ycsb.cu -I ./include/ -o gen_ycsb")
os.system("nvcc ./generation/gen_ycsb.cu -I ./include/ -o gen_ycsb")
print("nvcc ./generation/gen_tpcc_table.cu -I ./include/ -o gen_tpcc_table")
os.system("nvcc ./generation/gen_tpcc_table.cu -I ./include/ -o gen_tpcc_table")

print(
    "nvcc -lnvrtc -lcuda -arch=sm_80 ./benchmark_cpu/tpcc.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o test_tpcc -I ./include"
)
os.system(
    "nvcc -lnvrtc -lcuda -arch=sm_80 ./benchmark_cpu/tpcc.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o test_tpcc -I ./include"
)

print(
    "nvcc -lnvrtc -lcuda -arch=sm_80 ./benchmark_cpu/ycsb.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o test_ycsb -I ./include"
)
os.system(
    "nvcc -lnvrtc -lcuda -arch=sm_80 ./benchmark_cpu/ycsb.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o test_ycsb -I ./include"
)

print("./gen_tpcc_table 0 128 1048576")
os.system("./gen_tpcc_table 0 128 1048576")
print("./gen_tpcc 128 1048576 0 0.1")
os.system("./gen_tpcc 128 1048576 0 0.1")
print("./gen_tpcc 128 1048576 1 0.1")
os.system("./gen_tpcc 128 1048576 1 0.1")
print("./gen_ycsb 10485760 1048576 0 1 ycsb_ro.txs")
os.system("./gen_ycsb 10485760 1048576 0 1 ycsb_ro.txs")
print("./gen_ycsb 10485760 1048576 0.6 0.9 ycsb_mc.txs")
os.system("./gen_ycsb 10485760 1048576 0.6 0.9 ycsb_mc.txs")
print("./gen_ycsb 10485760 1048576 0.8 0.9 ycsb_hc.txs")
os.system("./gen_ycsb 10485760 1048576 0.8 0.9 ycsb_hc.txs")
