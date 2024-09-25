set terminal pngcairo size 1100,600 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体
set output "../png/ycsb_ro_breakdown.png"  ###输出的文件名

set grid ytics linewidth 4

set key font "Roman,16,Bold"
set key box
set key right outside

set style data histogram
set style histogram rowstacked
set style fill pattern border -1

plot '../ycsb_ro_breakdown1.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"


set output "../png/ycsb_mc_breakdown.png"  ###输出的文件名

plot '../ycsb_mc_breakdown1.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"

set output "../png/ycsb_hc_breakdown.png"  ###输出的文件名

plot '../ycsb_hc_breakdown1.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"