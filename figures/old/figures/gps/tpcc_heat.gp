set terminal pngcairo size 1500,750 enhanced font 'Verdana,12'   ## 格式，大小和字体
set output "../png/tpcc_heat_no.png"  ###输出的文件名

### 定义变量，用来设置上下左右的边缘和子图间距离
left=0.1           
right=0.95
bottom=0.1
top=0.95
hspace=0.1
wspace=0.05

###因为是要一张图里4个子图，所以启用了多图模式：
set multiplot layout 2,4 
set xrange [-0.5:31.5]
set yrange [-0.5:5.5]
set xlabel 'block size'
set ylabel 'warp density'
set mxtics 2
set mytics 4
set datafile separator ","

unset key
set cblabel "throughput(million txn/s)"
set tics nomirror out scale 0.75

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "gacco"
plot '../matrices/gacco_no.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "gputx"
plot '../matrices/gputx_no.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_nw"
plot '../matrices/tpl_nw_no.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_wd"
plot '../matrices/tpl_wd_no.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "to"
plot '../matrices/to_no.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "mvcc"
plot '../matrices/mvcc_no.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "silo"
plot '../matrices/silo_no.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tictoc"
plot '../matrices/tictoc_no.txt' matrix rowheaders columnheaders with image


unset multiplot  ###退出多图模式，完成绘图并保存

###########################################################################################

set output "../png/tpcc_heat_pm.png"  ###输出的文件名

### 定义变量，用来设置上下左右的边缘和子图间距离
left=0.1           
right=0.95
bottom=0.1
top=0.95
hspace=0.1
wspace=0.05

###因为是要一张图里4个子图，所以启用了多图模式：
set multiplot layout 2,4 
set xrange [-0.5:31.5]
set yrange [-0.5:5.5]
set xlabel 'block size'
set ylabel 'warp density'
set mxtics 2
set mytics 4
set datafile separator ","

unset key
set cblabel "throughput(million txn/s)"
set tics nomirror out scale 0.75

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "gacco"
plot '../matrices/gacco_pm.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "gputx"
plot '../matrices/gputx_pm.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_nw"
plot '../matrices/tpl_nw_pm.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_wd"
plot '../matrices/tpl_wd_pm.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "to"
plot '../matrices/to_pm.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "mvcc"
plot '../matrices/mvcc_pm.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "silo"
plot '../matrices/silo_pm.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tictoc"
plot '../matrices/tictoc_pm.txt' matrix rowheaders columnheaders with image


unset multiplot  ###退出多图模式，完成绘图并保存
