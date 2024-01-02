#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#ifndef NVRTC_COMPILE
#include <vector>
#include <string>
#endif

namespace common
{

#ifndef TS_ALLOCATOR_RUN
#define TS_ALLOCATOR_TYPE NaiveTSAllocator
#define TS_ALLOCATOR 0
#endif

    class NaiveTSAllocator
    {
    public:
        __device__ void Init(size_t tid) {}

        __device__ unsigned long long Alloc()
        {
            return atomicAdd(&ts, 1);
        }

    private:
        unsigned long long ts;
    };

    __shared__ unsigned long long sb;

    class BatchedTSAllocator
    {
    public:
        __device__ void Init(size_t tid) {}

        __device__ unsigned long long Alloc()
        {
            return atomicAdd(&sb, 1);
        }

    private:
    };

#ifndef NVRTC_COMPILE

    class TSAllocatorPlaceholder
    {
    public:
        __device__ void Init(size_t tid) {}
        __device__ unsigned long long Alloc() { return 0; }
    };

    class TSAllocator_CPU
    {
    public:
        virtual void GetCompileOptions(std::vector<std::string> &opts) = 0;
        virtual void Init(int batch_id, int batch_st) = 0;
    };

    class NaiveTSAllocator_CPU : public TSAllocator_CPU
    {
    public:
        NaiveTSAllocator_CPU()
        {
            cudaMalloc(&allocator, sizeof(common::NaiveTSAllocator));
        }
        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D TS_ALLOCATOR_RUN");
            opts.push_back("-D TS_ALLOCATOR_TYPE=NaiveTSAllocator");
            opts.push_back("-D TS_ALLOCATOR=" + std::to_string((unsigned long long)allocator));
        }

        void Init(int batch_id, int batch_st) override
        {
            unsigned long long ori = 1;
            cudaMemcpy(allocator, &ori, sizeof(long long), cudaMemcpyHostToDevice);
        }

    private:
        NaiveTSAllocator *allocator;
    };

    class BatchedTSAllocator_CPU : public TSAllocator_CPU
    {
    public:
        BatchedTSAllocator_CPU()
        {
            cudaMalloc(&allocator, sizeof(common::BatchedTSAllocator));
        }
        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D TS_ALLOCATOR_RUN");
            opts.push_back("-D TS_ALLOCATOR_TYPE=BatchedTSAllocator");
            opts.push_back("-D TS_ALLOCATOR=" + std::to_string((unsigned long long)allocator));
        }

        void Init(int batch_id, int batch_st) override
        {
            unsigned long long ori = 1;
            cudaMemcpy(allocator, &ori, sizeof(long long), cudaMemcpyHostToDevice);
        }

    private:
        BatchedTSAllocator *allocator;
    };

#endif

}

#endif