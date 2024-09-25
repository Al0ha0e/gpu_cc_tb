import math

ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]

file_exts = ["hc"]

batch_density = 3  # 5

for file_ext in file_exts:
    with open(f"./ycsb_{file_ext}.txs.txt", "r") as f1:
        lines = f1.readlines()
        st = 0
        result = []
        result_abr = []
        for cc_scheme in ccs:
            sbb = []
            abrs = []
            for k in range(1, 33):
                line_items = lines[st + 6 * 5 * 32 + batch_density * 32 + k - 1].split(
                    ","
                )
                sbb.append(float(line_items[4]))
                abrs.append(int(line_items[5]))
            print(sbb)
            result.append(sbb)
            result_abr.append(abrs)
            st += 32 * 6 * 6

        with open(f"./ycsb_{file_ext}_warpnum.txt", "w") as f2:
            wlines = []
            for i in range(0, 32):
                wline = str(i + 1)
                for j in range(0, len(ccs)):
                    wline += " " + str(result[j][i] / 1e6)
                wlines.append(wline + "\n")
            f2.writelines(wlines)

        with open(f"./ycsb_{file_ext}_warpnum_abr.txt", "w") as f2:
            wlines = []
            for i in range(0, 32):
                wline = str(i + 1)
                for j in range(2, len(ccs)):
                    abr = float(result_abr[j][i] / 1048576)
                    wline += " " + str(abr)
                wlines.append(wline + "\n")
            f2.writelines(wlines)

# file_exts = ["pm", "no"]

# batch_density = 3  # 5

# for file_ext in file_exts:
#     with open(f"./tpcc_{file_ext}.txt", "r") as f1:
#         lines = f1.readlines()
#         st = 0
#         result = []
#         result_abr = []
#         for cc_scheme in ccs:
#             sbb = []
#             abrs = []
#             for k in range(1, 33):
#                 line_items = lines[st + 6 * 5 * 32 + batch_density * 32 + k - 1].split(
#                     ","
#                 )
#                 sbb.append(float(line_items[4]))
#                 abrs.append(int(line_items[5]))
#             print(sbb)
#             result.append(sbb)
#             result_abr.append(abrs)
#             st += 32 * 6 * 6

#         with open(f"./tpcc_{file_ext}_warpnum.txt", "w") as f2:
#             wlines = []
#             for i in range(0, 32):
#                 wline = str(i + 1)
#                 for j in range(0, len(ccs)):
#                     wline += " " + str(result[j][i] / 1e6)
#                 wlines.append(wline + "\n")
#             f2.writelines(wlines)

#         with open(f"./tpcc_{file_ext}_warpnum_abr.txt", "w") as f2:
#             wlines = []
#             for i in range(0, 32):
#                 wline = str(i + 1)
#                 for j in range(2, len(ccs)):
#                     abr = float(result_abr[j][i] / 1048576)
#                     wline += " " + str(abr)
#                 wlines.append(wline + "\n")
#             f2.writelines(wlines)
