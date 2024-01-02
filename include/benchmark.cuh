#ifndef BENCHMARK_H
#define BENCHMARK_H

#ifndef NVRTC_COMPILE
#include <string>
#endif

namespace common
{
#ifdef NVRTC_COMPILE

#else
    std::string GetBenchmarkName(std::string bench_name, unsigned int rsize, unsigned int wsize, std::string cc_name);
#endif
}

#endif