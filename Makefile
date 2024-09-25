
SOURCES =  ./src/runtime.cu ./src/generator.cu ./src/index.cu ./src/gacco.cu ./src/gputx.cu

CC = nvcc
CXXFLAGS = -O3 -std=c++17
LDFLAGS = -lnvrtc -lcuda
CUDAFLAGS = -arch=sm_89
INCDIR = ./include

ycsb : $(SOURCES)
	$(CC) $(CXXFLAGS) $(LDFLAGS) $(CUDAFLAGS) -I $(INCDIR) -o out/test_ycsb benchmark_cpu/ycsb.cu $(SOURCES)

.PHONY : clean
clean :
	rm out/*