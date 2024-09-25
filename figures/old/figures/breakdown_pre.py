ccs = ["gacco", "gputx", "tpl_nw", "tpl_wd", "to", "mvcc", "silo", "tictoc"]
ccs1 = [
    "gacco",
    "gputx",
    "tpl\\\\\\_nw",
    "tpl\\\\\\_wd",
    "to",
    "mvcc",
    "silo",
    "tictoc",
]

#setnames = ["ycsb_ro", "ycsb_mc", "ycsb_hc"]
setnames = ["tpcc_no", "tpcc_pm"]

for setname in setnames:
    with open(f"./{setname}_breakdown.txt", "r") as f1:
        lines = f1.readlines()
        datas = []
        max_tot = 0
        for i, cc in enumerate(ccs):
            oline = cc
            items = lines[i].split(",")
            data = [
                float(items[10]),  # 0 gpu total
                float(items[6]),  # 1 ts
                float(items[7]),  # 2 wait
                float(items[8]),  # 3 abort
                float(items[9]),  # 4 manager
                float(items[11]),  # 5 index
                0,  # 6 useful
                float(items[12]),  # 7 pre
                float(items[13]),  # 8 gpu
                float(items[14]),  # 9 tot
            ]
            data[4] -= data[2]
            # max_tot = max(max_tot, float(items[14]))
            data[-3] = data[-3] / data[-1]
            print("???", data[-2], data[0])
            data[-2] = 1 - data[-3]
            sum = 0
            sum1 = 0
            for i in range(1, 6):
                sum1 += data[i]
                data[i] = data[i] / data[0] * data[-2]
                sum += data[i]
            print("-------", sum1, data[0])
            data[6] = data[-2] - sum
            datas.append(data)
        # line0 = "# ts wait abort manager index useful pre\n"
        # olines = [line0]
        olines = []
        for i, cc in enumerate(ccs1):
            data = datas[i]
            line = cc
            frac = 1  # data[-1] / max_tot
            for i in range(1, 7):
                line += " " + str(data[i] * frac)
            line += " " + str(data[-3] * frac) + "\n"
            olines.append(line)
        with open(f"./{setname}_breakdown1.txt", "w") as f2:
            f2.writelines(olines)
