import math

ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]


# filenames = ['ycsb_ro','ycsb_mc','ycsb_hc']

# for filename in filenames:
#     with open(f"./{filename}.txs.txt", "r") as f1:
#         lines = f1.readlines()
#         print("???")
#         st = 0
#         result = []
#         result_abr = []
#         for cc_scheme in ccs:
#             sbb = []
#             abrs = []
#             for i in range(0, 6):
#                 max_t = 0
#                 max_abr = 0
#                 for j in range(0, 6):
#                     for k in range(1, 33):
#                         line_items = lines[st + i * 6 * 32 + j * 32 + k - 1].split(",")
#                         through = float(line_items[4])
#                         if through > max_t:
#                             max_t = through
#                             max_abr = int(line_items[5])
#                 sbb.append(max_t)
#                 abrs.append(max_abr)
#             print(sbb)
#             result.append(sbb)
#             result_abr.append(abrs)
#             st += 32 * 6 * 6

#         with open(f"./{filename}_batchsize.txt", "w") as f2:
#             wlines = []
#             for i in range(0, 6):
#                 wline = str(i * 2 + 10)
#                 for j in range(0, len(ccs)):
#                     wline += " " + str(result[j][i] / 1e6)
#                 wlines.append(wline + "\n")
#             f2.writelines(wlines)

#         with open(f"./{filename}_batchsize_abr.txt", "w") as f2:
#             wlines = []
#             for i in range(0, 6):
#                 wline = str(i * 2 + 10)
#                 for j in range(2, len(ccs)):
#                     abr = float(result_abr[j][i] / 1048576)
#                     wline += " " + str(abr)
#                 wlines.append(wline + "\n")
#             f2.writelines(wlines)

filenames = ['tpcc_no','tpcc_pm']

for filename in filenames:
    print('----------------------------')
    with open(f"./{filename}.txt", "r") as f1:
        lines = f1.readlines()
        st = 0
        result = []
        result_abr = []
        for cc_scheme in ccs:
            sbb = []
            abrs = []
            params = []
            for i in range(0, 6):
                max_t = 0
                max_abr = 0
                max_param = ''
                for j in range(0, 6):
                    for k in range(1, 33):
                        line_items = lines[st + i * 6 * 32 + j * 32 + k - 1].split(",")
                        through = float(line_items[4])
                        if through > max_t:
                            max_t = through
                            max_abr = int(line_items[5])
                            max_param = line_items[1] + ' ' + line_items[2] + ' ' + line_items[3]  
                sbb.append(max_t)
                abrs.append(max_abr)
                params.append(max_param)
            for sb in params:
                print(f"os.system(\'./test_tpcc {1 if filename == 'tpcc_no' else 0} 128 1048576 {cc_scheme} {sb} 0\')")
            #print(cc_scheme,params)
            result.append(sbb)
            result_abr.append(abrs)
            st += 32 * 6 * 6

        with open(f"./{filename}_batchsize.txt", "w") as f2:
            wlines = []
            for i in range(0, 6):
                wline = str(i * 2 + 10)
                for j in range(0, len(ccs)):
                    wline += " " + str(result[j][i] / 1e6)
                wlines.append(wline + "\n")
            f2.writelines(wlines)

        with open(f"./{filename}_batchsize_abr.txt", "w") as f2:
            wlines = []
            for i in range(0, 6):
                wline = str(i * 2 + 10)
                for j in range(2, len(ccs)):
                    abr = float(result_abr[j][i] / 1048576)
                    wline += " " + str(abr)
                wlines.append(wline + "\n")
            f2.writelines(wlines)
