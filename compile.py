cmds = [
    "nvcc ./generation/gen_tpcc.cu -I ./include/ -o ./out/gen_tpcc",
    "nvcc ./generation/gen_ycsb.cu -I ./include/ -o ./out/gen_ycsb",
    "nvcc ./generation/gen_tpcc_table.cu -I ./include/ -o ./out/gen_tpcc_table",
    "nvcc -lnvrtc -lcuda -arch=sm_89 ./benchmark_cpu/tpcc.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o ./out/test_tpcc -I ./include",
    "nvcc -lnvrtc -lcuda -arch=sm_89 ./benchmark_cpu/ycsb.cu ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu -o ./out/test_ycsb -I ./include",
]

for cmd in cmds:
    print(cmd)
    os.system(cmd)