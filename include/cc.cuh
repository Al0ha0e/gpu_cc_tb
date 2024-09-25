#ifndef CC_H
#define CC_H

#ifndef NVRTC_COMPILE

#include <string>
#include <env.cuh>

#endif

#include <transaction.cuh>
#include <db.cuh>

namespace common
{
#ifndef NVRTC_COMPILE

    class ConcurrencyControlCPUBase
    {
    public:
        ConcurrencyControlCPUBase() {}

        ConcurrencyControlCPUBase(size_t batch_size,
                                  size_t tx_cnt,
                                  size_t obj_cnt) : batch_size(batch_size),
                                                    tx_cnt(tx_cnt),
                                                    obj_cnt(obj_cnt) {}

        virtual void Init0()
        {
            for (size_t st = 0; st < tx_cnt; st += batch_size)
                batches.push_back(min(batch_size, tx_cnt - st));

            streams.resize(batches.size());
        }
        virtual void Init(int batch_id, int batch_st) = 0;
        virtual void *ToGPU() = 0;
        virtual void GetCompileOptions(std::vector<std::string> &opts) = 0;
        virtual size_t GetMemSize() = 0;
        virtual void Explain(unsigned long long self_info, unsigned long long target_info)
        {
            std::cout << "self_info " << self_info << " target_info " << target_info << "\n";
        }

        void Sync()
        {
            for (auto &stream : streams)
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        }

        size_t batch_size;
        size_t tx_cnt;
        size_t obj_cnt;
        std::vector<size_t> batches;
        std::vector<cudaStream_t> streams;
    };

    class ConcurrencyControlPlaceholder
    {
    public:
        size_t self_tid;
        __device__ ConcurrencyControlPlaceholder(void *txs_info, void *info, size_t tid);
        __device__ bool TxStart(void *info);
        __device__ bool TxEnd(void *info);

        __device__ bool Read(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size);

        __device__ bool ReadForUpdate(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size);

        __device__ bool Write(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size);

        __device__ void Finalize();
    };

#define CC_TYPE common::ConcurrencyControlPlaceholder
#define THREAD_TX_SHIFT 0
#define GLOBAL_METRICS 0

#endif

}

#endif