#include <benchmark.cuh>
#include <sstream>
namespace common
{
    std::string GetBenchmarkName(std::string bench_name, unsigned int rsize, unsigned int wsize, std::string cc_name)
    {
        auto buffer = std::stringstream();
        buffer << bench_name << "<" << rsize << ", " << wsize << ", " << cc_name << ">";
        std::string bench_instance_name(buffer.str());
        return bench_instance_name;
    }
}