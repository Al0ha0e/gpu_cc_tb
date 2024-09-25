import os
import subprocess


for i in range(0,1):
    os.system(
        # f"./out/test_ycsb ./dataset/ycsb_hc.txs 10485760 {1048576} gputx {0} {32} {1048576} 1"
        f"./out/test_ycsb ./dataset/ycsb_hc.txs 10485760 {1048576} tictoc {5} {32} {1048576} 0"
    )

## compute-sanitizer --tool memcheck 