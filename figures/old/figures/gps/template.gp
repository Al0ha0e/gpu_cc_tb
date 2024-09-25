############  几十行代码急速入门Gnuplot数据绘图   ###################################
### "#" 后边是注释
### 首先设置输出的格式，支持pdf,png,eps等常用格式
set terminal pngcairo size 1000,1000 font 'Times New Roman,10'   ## 格式，大小和字体
set output "plot.png"  ###输出的文件名

#set terminal pdfcairo size 20cm,20cm font 'Times New Roman,12'  ## 
#set output "plot.pdf"

#set terminal epscairo size 20cm,20cm font 'Times New Roman,12'  ## 
#set output "plot.eps"

### 可以定义变量和宏，便于后边重复使用
sx = "set xrange "  ## 例如后边当成宏来引用：@sx ，而不是使用 "set xrange"，你可缩减代码量

### 定义变量，用来设置上下左右的边缘和子图间距离
left=0.1           
right=0.95
bottom=0.1
top="0.95"
hspace="0.1"
wspace="0.15"

###因为是要一张图里4个子图，所以启用了多图模式：
set multiplot layout 1,4 spacing @hspace,@wspace margins left,right,bottom,@top

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set label "(1)" at graph 0.02,0.03 font ',20' textcolor rgb 'red'
set title "example"
set xlabel "This is xlabel with {/Symbol a}=0.1 to 100"
set ylabel "This is ylabel with X^2_3"
set key top right Left reverse font 'Times New Roman,15'  ###设置图例格式：位置、字体等
f(x)=sin(x)/x          ###定义函数
set xrange [0:100]     ###设置x轴范围
set yrange [-0.5:1.0]  ###设置y轴范围
set xtics scale 3      ###设置x轴的刻度的长度，是默认的3倍长
set mxtics 10          ###x轴子刻度的数目
set mytics 5           ###y轴子刻度的数目
set log x              ###x轴设置成log
set style fill pattern 1
                       ### plot命令开始绘图并设置参数：
plot f(x) w line linetype 1 pointtype 5 ps 1.0 lc 2 lw 4,\
     f(x) w filledcurves y=0.5 lc rgb 'blue'
###上边用到了缩写ps=pointsize.其他也有缩写：如line=l, linetype=lt, pointtype=pt等等

######### 子图 (2): 数据文件绘图和拟合

reset           ###Gnuplot会继承上边的命令，所以需要reset取消之前所有的设置
set title "fitting functions"
set label "(2)" at graph 0.02,0.03 font ',20'  textcolor rgb 'red'
@sx [0:11]      ###这里引用了宏
set yrange [0:11]
set key bottom right font ",14" spacing 2
g(x) = a*x**2 + b*x + c             ###定义函数并拟合，参数为a,b,c
fit g(x) "data.txt" u 1:2 via a,b,c ###定义函数并拟合
key_g= sprintf("fits without yerror:\ng(x) = %5.3f*x^2 + %5.3f*x + %5.3f",a,b,c)
h(x) = d*x**2 + e*x + f
fit h(x) "data.txt" u 1:2:3 yerror via d,e,f
key_h= sprintf("fits with yerror:\nh(x) = %5.3f*x^2 + %5.3f*x + %5.3f",d,e,f)
set xlabel 'xxxx' rotate by 45
plot "data.txt" u 1:2:3 w yerror pt 5 ps 1.0 lc rgb 'blue' title "data",\
     g(x) w line linecolor rgb 'red' title key_g,\
     h(x) w l lc rgb 'green' title key_h

######### 子图 (3)：统计和填充
reset
set key left top box
set label "(3)" at graph 0.02,0.03 font ',20'  textcolor rgb 'red'
df='data.txt'                ###这里使用文件里的数据绘图
stats df u 1:2 name "A"      ###统计数据的1和2列，并将统计结果存入A中
print A_min_x, A_min_y       ###可以打印出A中统计的x的最大和最小值
@sx [A_min_x-1: A_max_x + 1] ###根据统计数据设置x轴范围
set yr [A_min_y-1: A_max_y + 1]
set arrow from A_min_x,A_min_y+2 to A_min_x,A_min_y ###绘制箭头
set label 'Min.' at A_min_x,A_min_y+2.3 center
set arrow from A_max_x,A_max_y-2 to A_max_x,A_max_y
set label 'Max.' at A_max_x,A_max_y-2.3 center front
set xlabel 'xxxx' textcolor rgb 'red'
set style fill solid 0.5 
#set style fill pattern 3
plot df u 1:2 w filledcurves y=2 lc rgb 'seagreen' title 'fill with y=2',\
     df u 1:2 w lp pt 4 ps 0.9 lc rgb 'red' lw 2,\
     [3:6] A_mean_y w l dt '--' lw 2 lc rgb 'black' title "mean"

######## 子图 (4):直方图
reset
set label "(4)" at graph 0.02,0.03 font ',20'  textcolor rgb 'red'
set style data histogram
set style fill solid 1.0 border lt -1 
set boxwidth 1.0
set key at 1.5,7  box font ',15' reverse
set yr [0:10]
set xtics ("A" 0, "B" 1, "C" 2, "{/Symbol b}" 3, "{/Symbol S}" 4)
set xlabel 'xxxx' offset 0,-1
plot "data.txt" u 1 title 'histogram' lc rgb 'seagreen',\
     "data.txt" u 0:($1+0.5):1 w labels title ""

unset multiplot  ###退出多图模式，完成绘图并保存