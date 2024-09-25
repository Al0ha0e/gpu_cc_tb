set terminal pngcairo size 1200,500 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体
set output "../png/ycsb_ro_diffwd.png"  ###输出的文件名

set xlabel "warp density"     # X轴标签
set ylabel "throughput(million txn/s)"     # Y轴标签
set grid ytics linewidth 4

set key font "Roman,16,Bold"
set key under box center

throu_pth = "../mid/ycsb_diffwd_ro_throughput.txt"

plot throu_pth using 1:2 with lp pt 1 ps 2 lc rgb 'red' lw 4 title "gacco", \
     throu_pth using 1:3 with lp pt 2 ps 2 lc rgb 'green' lw 4 title "gputx", \
     throu_pth using 1:4 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     throu_pth using 1:5 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     throu_pth using 1:6 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     throu_pth using 1:7 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     throu_pth using 1:8 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     throu_pth using 1:9 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"

####################################################################################

set output "../png/ycsb_mc_diffwd.png"  ###输出的文件名

# set xlabel "warp density"     # X轴标签
set ylabel "throughput(million txn/s)"     # Y轴标签
set grid ytics linewidth 4


throu_pth = "../mid/ycsb_diffwd_mc_throughput.txt"
abr_pth = "../mid/ycsb_diffwd_mc_abr.txt"

plot throu_pth using 1:2 with lp pt 1 ps 2 lc rgb 'red' lw 4 title "gacco", \
     throu_pth using 1:3 with lp pt 2 ps 2 lc rgb 'green' lw 4 title "gputx", \
     throu_pth using 1:4 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     throu_pth using 1:5 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     throu_pth using 1:6 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     throu_pth using 1:7 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     throu_pth using 1:8 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     throu_pth using 1:9 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"

set output "../png/ycsb_mc_diffwd_abr.png"  ###输出的文件名

set logscale y 2
#set yrange [0.0156:128]
set ylabel "abort rate"     # Y轴标签

plot abr_pth using 1:2 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     abr_pth using 1:3 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     abr_pth using 1:4 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     abr_pth using 1:5 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     abr_pth using 1:6 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     abr_pth using 1:7 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"

####################################################################################

set output "../png/ycsb_hc_diffwd.png"  ###输出的文件名

# set xlabel "warp density"     # X轴标签
set ylabel "throughput(million txn/s)"     # Y轴标签
set grid ytics linewidth 4

throu_pth = "../mid/ycsb_diffwd_hc_throughput.txt"
abr_pth = "../mid/ycsb_diffwd_hc_abr.txt"

unset logscale y

plot throu_pth using 1:2 with lp pt 1 ps 2 lc rgb 'red' lw 4 title "gacco", \
     throu_pth using 1:3 with lp pt 2 ps 2 lc rgb 'green' lw 4 title "gputx", \
     throu_pth using 1:4 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     throu_pth using 1:5 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     throu_pth using 1:6 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     throu_pth using 1:7 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     throu_pth using 1:8 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     throu_pth using 1:9 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"

set output "../png/ycsb_hc_diffwd_abr.png"  ###输出的文件名

set logscale y 2
#set yrange [0.0156:128]
set ylabel "abort rate"     # Y轴标签

plot abr_pth using 1:2 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     abr_pth using 1:3 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     abr_pth using 1:4 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     abr_pth using 1:5 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     abr_pth using 1:6 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     abr_pth using 1:7 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"
