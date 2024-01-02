set terminal pngcairo size 800,600 enhanced font 'C:/Windows/Fonts/roman.fon,20,Bold'   ## 格式，大小和字体
set output "../png/tpcc_pm_batchsize.png"  ###输出的文件名

set xlabel "log_2(batch size)"     # X轴标签
set ylabel "throughput(million txn/s)"     # Y轴标签
set grid ytics linewidth 4

set key font "Roman,16,Bold"
set key box
set key right outside

plot "../tpcc_pm_batchsize.txt" using 1:2 with lp pt 1 ps 2 lc rgb 'red' lw 4 title "gacco", \
     "../tpcc_pm_batchsize.txt" using 1:3 with lp pt 2 ps 2 lc rgb 'green' lw 4 title "gputx", \
     "../tpcc_pm_batchsize.txt" using 1:4 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     "../tpcc_pm_batchsize.txt" using 1:5 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     "../tpcc_pm_batchsize.txt" using 1:6 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     "../tpcc_pm_batchsize.txt" using 1:7 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     "../tpcc_pm_batchsize.txt" using 1:8 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     "../tpcc_pm_batchsize.txt" using 1:9 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"

set output "../png/tpcc_no_batchsize.png"  ###输出的文件名

plot "../tpcc_no_batchsize.txt" using 1:2 with lp pt 1 ps 2 lc rgb 'red' lw 4 title "gacco", \
     "../tpcc_no_batchsize.txt" using 1:3 with lp pt 2 ps 2 lc rgb 'green' lw 4 title "gputx", \
     "../tpcc_no_batchsize.txt" using 1:4 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     "../tpcc_no_batchsize.txt" using 1:5 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     "../tpcc_no_batchsize.txt" using 1:6 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     "../tpcc_no_batchsize.txt" using 1:7 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     "../tpcc_no_batchsize.txt" using 1:8 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     "../tpcc_no_batchsize.txt" using 1:9 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"

set output "../png/tpcc_pm_batchsize_abr.png"  ###输出的文件名

set logscale y 2
#set yrange [0.0156:128]
set ylabel "abort rate"     # Y轴标签

plot "../tpcc_pm_batchsize_abr.txt" using 1:2 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     "../tpcc_pm_batchsize_abr.txt" using 1:3 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     "../tpcc_pm_batchsize_abr.txt" using 1:4 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     "../tpcc_pm_batchsize_abr.txt" using 1:5 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     "../tpcc_pm_batchsize_abr.txt" using 1:6 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     "../tpcc_pm_batchsize_abr.txt" using 1:7 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"

set output "../png/tpcc_no_batchsize_abr.png"  ###输出的文件名

plot "../tpcc_no_batchsize_abr.txt" using 1:2 with lp pt 4 ps 2 lc rgb 'blue' lw 4 title "tpl\\\_nw", \
     "../tpcc_no_batchsize_abr.txt" using 1:3 with lp pt 6 ps 2 lc rgb 'purple' lw 4 title "tpl\\\_wd", \
     "../tpcc_no_batchsize_abr.txt" using 1:4 with lp pt 8 ps 2 lc rgb 'orange' lw 4 title "to", \
     "../tpcc_no_batchsize_abr.txt" using 1:5 with lp pt 7 ps 2 lc rgb 'cyan' lw 4 title "mvcc", \
     "../tpcc_no_batchsize_abr.txt" using 1:6 with lp pt 'x' ps 2 lc rgb 'black' lw 4 title "silo", \
     "../tpcc_no_batchsize_abr.txt" using 1:7 with lp pt '+' ps 2 lc rgb 'pink' lw 4 title "tictoc"