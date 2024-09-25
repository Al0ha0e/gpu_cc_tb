ccscheme = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]
# ccscheme = ['gputx']

# file_exts = ["ro", "mc", "hc"]

# for file_ext in file_exts:
#     with open(f"./ycsb_{file_ext}.txs.txt", "r") as f:
#         lines = f.readlines()
#         st = 32 * 6 * 5
#         for cc in ccscheme:
#             with open("./matrices/" + cc + f"_{file_ext}.txt", "w") as f2:
#                 line0 = ""
#                 for i in range(1, 33):
#                     line0 += "," + (str(i) if i % 4 == 0 else "")
#                 newlines = [line0 + "\n"]
#                 for i in range(5, -1, -1):
#                     newline = str(5 - i)
#                     for j in range(1, 33):
#                         print(st + i * 32 + j, lines[st + i * 32 + j - 1])
#                         line_items = lines[st + i * 32 + j - 1].split(",")
#                         newline += "," + str(float(line_items[4]) / 1e6)
#                     newline += "\n"
#                     newlines.append(newline)
#                 f2.writelines(newlines)
#             st += 32 * 6 * 6

file_exts = ["no", "pm"]

for file_ext in file_exts:
    with open(f"./tpcc_{file_ext}.txt", "r") as f:
        lines = f.readlines()
        st = 32 * 6 * 5
        for cc in ccscheme:
            with open("./matrices/" + cc + f"_{file_ext}.txt", "w") as f2:
                line0 = ""
                for i in range(1, 33):
                    line0 += "," + (str(i) if i % 4 == 0 else "")
                newlines = [line0 + "\n"]
                for i in range(5, -1, -1):
                    newline = str(5 - i)
                    for j in range(1, 33):
                        print(st + i * 32 + j, lines[st + i * 32 + j - 1])
                        line_items = lines[st + i * 32 + j - 1].split(",")
                        newline += "," + str(float(line_items[4]) / 1e6)
                    newline += "\n"
                    newlines.append(newline)
                f2.writelines(newlines)
            st += 32 * 6 * 6
