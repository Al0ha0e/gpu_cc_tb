set terminal pngcairo size 800,600 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体

# set xlabel " " offset 0,-2
set ylabel "throughput(million txn/s)"
set grid ytics linewidth 4

set key font "Roman,16,Bold"
set key under box center

set style data histogram
set yrange [0:*]

set style histogram clustered gap 1 title offset -0,0.25
set style fill solid noborder
set boxwidth 0.95

set xtics font "Roman,16,Bold"
set ytics font "Roman,16,Bold"
# set key font "mbfont:SimSun,18"

# factors = "tpl\\\_wd to mvcc silo tictoc"
# set xtic ("tpl\\\_nw" 0)
# NF = words(factors)
# set for [i=1:NF] xtics add (word(factors,i) i)

set xtics ("tpl\\\_nw" 0, "tpl\\\_wd" 1, "to" 2, "mvcc" 3, "silo" 4, "tictoc" 5)

set output "../png/ycsb_ro_latch.png"  ###输出的文件名
throu_pth = "../mid/ycsb_latch_ro.txt"
plot throu_pth u 1 ti col , '' u 2 ti col

set output "../png/ycsb_mc_latch.png"  ###输出的文件名
throu_pth = "../mid/ycsb_latch_mc.txt"
plot throu_pth u 1 ti col , '' u 2 ti col

set output "../png/ycsb_hc_latch.png"  ###输出的文件名
throu_pth = "../mid/ycsb_latch_hc.txt"
plot throu_pth u 1 ti col , '' u 2 ti col