#ifndef SLOW_MVCC_H
#define SLOW_MVCC_H

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>
#include <timestamp.cuh>

namespace cc
{
    struct mvcc_full_timestamp_t
    {
        int uncommited;
        unsigned int prev;
        unsigned long long rts;
        unsigned long long wts;
    };

    struct slow_mvcc_write_info
    {
        mvcc_full_timestamp_t *verp;
        void *srcdata;
        void *dstdata;
        size_t size;
    };

    struct DynamicSlowMVCCInfo
    {
        slow_mvcc_write_info *write_info;
        char *has_wts;

        __host__ __device__ DynamicSlowMVCCInfo() {}
        __host__ __device__ DynamicSlowMVCCInfo(
            slow_mvcc_write_info *write_info,
            char *has_wts) : write_info(write_info),
                             has_wts(has_wts)
        {
        }
    };

#ifndef SLOW_MVCC_RUN
#define SLOW_VERSION_TABLE 0
#define SLOW_VERSION_NODES 0
#define MVCC_LATCH_TABLE 0
#endif

    class Slow_MVCC_GPU
    {
    public:
        common::Metrics self_metrics;
        mvcc_full_timestamp_t *self_nodes;
        size_t self_ts;
        size_t self_tid;
        unsigned long long st_time;

#ifdef DYNAMIC_RW_COUNT
        common::DynamicTransactionSet_GPU *txset_info;
        slow_mvcc_write_info *write_info;
        char *has_wts;
        int rcnt;
        int wcnt;
#else
        slow_mvcc_write_info write_info[WCNT];
        char has_wts[WCNT];
#endif

        __device__ Slow_MVCC_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[tid];
            wcnt = txset_info->tx_wcnt[tid];
            size_t wst = txset_info->tx_wcnt_st[tid];
            DynamicSlowMVCCInfo *tinfo = (DynamicSlowMVCCInfo *)info;
            write_info = tinfo->write_info + wst;
            has_wts = tinfo->has_wts + wst;
            self_nodes = ((mvcc_full_timestamp_t *)SLOW_VERSION_NODES) + wst;
#else
            self_nodes = ((mvcc_full_timestamp_t *)SLOW_VERSION_NODES) + tid * WCNT;
#endif
        }

        __device__ bool TxStart(void *info)
        {
            unsigned long long st_time2 = clock64();
            self_ts = ((common::TS_ALLOCATOR_TYPE *)TS_ALLOCATOR)->Alloc();
            st_time = clock64();
            self_metrics.ts_duration += st_time - st_time2;
            self_metrics.wait_duration = 0;
            memset(has_wts, 0, sizeof(char) * WCNT);
            self_metrics.manager_duration = clock64() - st_time;
            return true;
        }

        __device__ bool TxEnd(void *info)
        {
            unsigned long long manager_st_time = clock64();
#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                slow_mvcc_write_info &winfo = write_info[i];
                memcpy(winfo.dstdata, winfo.srcdata, winfo.size);
                winfo.verp->uncommited = 0;
            }
            __threadfence();
            self_metrics.manager_duration += clock64() - manager_st_time;
            return true;
        }

        __device__ bool Read(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            unsigned long long manager_st_time = clock64();
            volatile mvcc_full_timestamp_t *entry = ((mvcc_full_timestamp_t *)SLOW_VERSION_TABLE) + obj_idx;
            int *latch_entry = ((int *)MVCC_LATCH_TABLE) + obj_idx;

            while (true)
            {
                latch_lock(latch_entry);
                unsigned long long wait_st_time = clock64();
                if (entry->uncommited)
                {
                    self_metrics.wait_duration += clock64() - wait_st_time;
                    latch_unlock(latch_entry);
                    continue;
                }
                mvcc_full_timestamp_t *version = (mvcc_full_timestamp_t *)entry;
                while (true)
                {
                    if (version->wts <= self_ts)
                        break;
                    else
                        version = ((mvcc_full_timestamp_t *)SLOW_VERSION_NODES) + version->prev;
                }
                version->rts = max(version->rts, (unsigned long long)self_ts);
                memcpy(dstdata, srcdata, size);
                latch_unlock(latch_entry);
                break;
            }

            self_metrics.manager_duration += clock64() - manager_st_time;
            return true;
        }

        __device__ bool ReadForUpdate(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            return Read(obj_idx, tx_idx, srcdata, dstdata, size);
        }

        __device__ bool Write(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            unsigned long long manager_st_time = clock64();
            volatile mvcc_full_timestamp_t *entry = ((volatile mvcc_full_timestamp_t *)SLOW_VERSION_TABLE) + obj_idx;
            int *latch_entry = ((int *)MVCC_LATCH_TABLE) + obj_idx;

            latch_lock(latch_entry);
            if (entry->wts > self_ts || entry->rts > self_ts || entry->uncommited)
            {
                latch_unlock(latch_entry);
                rollback();
                return false;
            }
            self_nodes[tx_idx] = *(mvcc_full_timestamp_t *)entry;
            entry->prev = (self_nodes - ((mvcc_full_timestamp_t *)SLOW_VERSION_NODES)) + tx_idx;
            entry->uncommited = 1;
            entry->rts = self_ts;
            entry->wts = self_ts;
            latch_unlock(latch_entry);

            slow_mvcc_write_info *winfo = write_info + tx_idx;
            winfo->verp = (mvcc_full_timestamp_t *)entry;
            winfo->srcdata = srcdata;
            winfo->dstdata = dstdata;
            winfo->size = size;
            has_wts[tx_idx] = true;
            self_metrics.manager_duration += clock64() - manager_st_time;
            return true;
        }

        __device__ void Finalize()
        {
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->abort), self_metrics.abort);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->ts_duration), self_metrics.ts_duration);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->wait_duration), self_metrics.wait_duration);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->abort_duration), self_metrics.abort_duration);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->manager_duration), self_metrics.manager_duration);
        }

    private:
        void __device__ latch_lock(int *latch)
        {
            unsigned long long wait_st_time = clock64();
            while (atomicCAS(latch, 0, 1))
                ;
            self_metrics.wait_duration += clock64() - wait_st_time;
        }

        void __device__ latch_unlock(int *latch)
        {
            *latch = 0;
            __threadfence();
        }

        void __device__ rollback()
        {
            self_metrics.abort++;
#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                if (has_wts[i])
                {
                    slow_mvcc_write_info &winfo = write_info[i];
                    mvcc_full_timestamp_t *node = winfo.verp;
                    int *latch_entry = (int *)MVCC_LATCH_TABLE +
                                       (node - (mvcc_full_timestamp_t *)SLOW_VERSION_TABLE);
                    latch_lock(latch_entry);
                    *node = ((mvcc_full_timestamp_t *)SLOW_VERSION_NODES)[node->prev];
                    latch_unlock(latch_entry);
                }
            }
            __threadfence();
            self_metrics.abort_duration += clock64() - st_time;
        }
    };

#ifndef NVRTC_COMPILE

    class Slow_MVCC_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        mvcc_full_timestamp_t *version_table;
        mvcc_full_timestamp_t *version_nodes;
        int *latch_table;
        slow_mvcc_write_info *write_info;
        char *has_wts;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        common::TSAllocator_CPU *ts_allocator;
        void *mvcc_gpu_info;
        bool dynamic;

        Slow_MVCC_CPU(common::DB_CPU *db,
                      common::TransactionSet_CPU *txinfo,
                      size_t bsize,
                      common::TSAllocator_CPU *ts_allocator)
            : info(txinfo),
              db_cpu(db),
              ts_allocator(ts_allocator),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&version_table, sizeof(mvcc_full_timestamp_t) * db->table_st[db->table_cnt]);
            cudaMalloc(&latch_table, sizeof(int) * db->table_st[db->table_cnt]);
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                size_t totw = dtx->GetTotW();
                cudaMalloc(&version_nodes, sizeof(mvcc_full_timestamp_t) * totw);
                cudaMemset(version_nodes, 0, sizeof(mvcc_full_timestamp_t) * totw);
                cudaMalloc(&has_wts, sizeof(char) * totw);
                cudaMalloc(&write_info, sizeof(slow_mvcc_write_info) * totw);
                DynamicSlowMVCCInfo *tmp = new DynamicSlowMVCCInfo(write_info, has_wts);
                cudaMalloc(&mvcc_gpu_info, sizeof(DynamicSlowMVCCInfo));
                cudaMemcpy(mvcc_gpu_info, tmp, sizeof(DynamicSlowMVCCInfo), cudaMemcpyHostToDevice);
                delete tmp;
            }
            else
            {
                common::StaticTransactionSet_CPU *stx = (common::StaticTransactionSet_CPU *)info;
                cudaMalloc(&version_nodes, sizeof(mvcc_full_timestamp_t) * stx->wcnt * tx_cnt);
                mvcc_gpu_info = nullptr;
            }
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            cudaStream_t stream = streams[batch_id];
            ts_allocator->Init(batch_id, batch_st);
            cudaMemset(version_table, 0, sizeof(mvcc_full_timestamp_t) * db_cpu->table_st[db_cpu->table_cnt]);
            cudaMemset(latch_table, 0, sizeof(int) * db_cpu->table_st[db_cpu->table_cnt]);
            if (dynamic)
            {
                // common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                // size_t totw = dtx->GetTotW();
                // cudaMemsetAsync(
                //     version_nodes + sizeof(mvcc_full_timestamp_t) * dtx->tx_wcnt_st[batch_st],
                //     0,
                //     sizeof(mvcc_full_timestamp_t) * (dtx->tx_wcnt_st[batch_st + batches[batch_id]] - dtx->tx_wcnt_st[batch_st]),
                //     stream);
            }
            else
            {
                common::StaticTransactionSet_CPU *stx = (common::StaticTransactionSet_CPU *)info;
                cudaMemsetAsync(
                    version_nodes + sizeof(mvcc_full_timestamp_t) * stx->wcnt * batch_st,
                    0,
                    sizeof(mvcc_full_timestamp_t) * stx->wcnt * batches[batch_id],
                    stream);
            }
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            ts_allocator->GetCompileOptions(opts);
            opts.push_back(std::string("-D SLOW_MVCC_RUN"));
            opts.push_back(std::string("-D SLOW_VERSION_TABLE=") + std::to_string((unsigned long long)version_table));
            opts.push_back(std::string("-D SLOW_VERSION_NODES=") + std::to_string((unsigned long long)version_nodes));
            opts.push_back("-D MVCC_LATCH_TABLE=" + std::to_string((unsigned long long)latch_table));
            opts.push_back(std::string("-D CC_TYPE=cc::Slow_MVCC_GPU"));
        }

        void *ToGPU() override
        {
            return mvcc_gpu_info;
        }

        size_t GetMemSize() override
        {
            return 0;
        }
    };

#endif
}

#endif