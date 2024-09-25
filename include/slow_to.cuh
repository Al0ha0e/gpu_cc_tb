#ifndef SLOW_TO_H
#define SLOW_TO_H

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>
#include <timestamp.cuh>

namespace cc
{
    struct __align__(8) slow_to_timestamp_struct
    {
        unsigned long long uncommited : 1;
        unsigned long long rts : 31;
        unsigned long long wts : 31;
    };

    union __align__(8) slow_to_timestamp_t
    {
        unsigned long long int ll;
        slow_to_timestamp_struct s;
    };

    struct __align__(8) SlowWriteInfo
    {
        slow_to_timestamp_t *tsp;
        slow_to_timestamp_t prev_ts;
        void *srcdata;
        void *dstdata;
        size_t size;
    };

    // struct TOInfo
    // {
    //     slow_to_timestamp_t *ts_table;
    //     __host__ __device__ TOInfo() {}
    //     __host__ __device__ TOInfo(slow_to_timestamp_t *ts_table) : ts_table(ts_table) {}
    // };

    struct __align__(8) SlowDynamicTOInfo
    {
        SlowWriteInfo *write_info;
        char *has_wts;

        __host__ __device__ SlowDynamicTOInfo() {}
        __host__ __device__ SlowDynamicTOInfo(SlowWriteInfo * write_info,
                                              char *has_wts) : write_info(write_info),
                                                               has_wts(has_wts)
        {
        }
    };

#ifndef SLOW_TO_RUN
#define SLOW_TO_LATCH_TABLE 0
#define SLOW_TO_TS_TABLE 0
#endif

    class Slow_TO_GPU
    {
    public:
        common::Metrics self_metrics;
        size_t self_tid;
        size_t self_ts;
        unsigned long long st_time;

#ifdef TX_DEBUG
        common::Event *self_events;
#endif

#ifdef DYNAMIC_RW_COUNT
        common::DynamicTransactionSet_GPU *txset_info;
        SlowWriteInfo *write_info;
        char *has_wts;
        int rcnt;
        int wcnt;
#else
        SlowWriteInfo write_info[WCNT];
        char has_wts[WCNT];
#endif

        __device__ Slow_TO_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[tid];
            wcnt = txset_info->tx_wcnt[tid];
            size_t wst = txset_info->tx_wcnt_st[tid];
            SlowDynamicTOInfo *tinfo = (SlowDynamicTOInfo *)info;
            write_info = tinfo->write_info + wst;
            has_wts = tinfo->has_wts + wst;
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif

#else
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + (RCNT + WCNT + 1) * self_tid;
#endif
#endif
        }

        __device__ bool TxStart(void *info)
        {
            unsigned long long st_time0 = clock64();
            self_ts = ((common::TS_ALLOCATOR_TYPE *)TS_ALLOCATOR)->Alloc();
            assert(self_ts <= ((1ULL << 31) - 1ULL));
            st_time = clock64();
            self_metrics.ts_duration += st_time - st_time0;
            self_metrics.wait_duration = 0;
            memset(has_wts, 0, sizeof(char) * WCNT);
            self_metrics.manager_duration = clock64() - st_time;
            return true;
        }

        __device__ bool TxEnd(void *info)
        {
            unsigned long long manager_st_time = clock64();
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + WCNT, 0, 0, self_ts, self_tid, 2);
#endif
#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                SlowWriteInfo &winfo = write_info[i];
                volatile slow_to_timestamp_t *entry = winfo.tsp;
                memcpy(winfo.dstdata, winfo.srcdata, winfo.size);
                entry->s.uncommited = 0;
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
            slow_to_timestamp_t *entry = ((slow_to_timestamp_t *)SLOW_TO_TS_TABLE) + obj_idx;
            int *latch_entry = ((int *)SLOW_TO_LATCH_TABLE) + obj_idx;

            unsigned long long wait_st_time = clock64();

            while (true)
            {
                common::latch_lock(self_metrics, latch_entry);
                slow_to_timestamp_t ts1;
                ts1.ll = ((volatile slow_to_timestamp_t *)entry)->ll;
                unsigned long long wts = ts1.s.wts;

                if (wts > self_ts)
                {
                    common::latch_unlock(latch_entry);
                    rollback();
                    return false;
                }

                if (!ts1.s.uncommited)
                {
                    // ts2.s.rts = self_ts;
                    ts1.s.rts = max(ts1.s.rts, (unsigned long long)self_ts);
                    memcpy(dstdata, srcdata, size);
#ifdef TX_DEBUG
                    common::AddEvent(self_events + tx_idx, obj_idx, ts1.ll, self_ts, self_tid, 0);
#endif
                    ((volatile slow_to_timestamp_t *)entry)->ll = ts1.ll;
                    common::latch_unlock(latch_entry);
                    break;
                }

                self_metrics.wait_duration += clock64() - wait_st_time;
                common::latch_unlock(latch_entry);
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
            slow_to_timestamp_t *entry = ((slow_to_timestamp_t *)SLOW_TO_TS_TABLE) + obj_idx;
            SlowWriteInfo *winfo = write_info + tx_idx;
            winfo->tsp = entry;
            winfo->srcdata = srcdata;
            winfo->dstdata = dstdata;
            winfo->size = size;

            int *latch_entry = (int *)SLOW_TO_LATCH_TABLE + obj_idx;

            common::latch_lock(self_metrics, latch_entry);
            slow_to_timestamp_t ts1;
            ts1.ll = ((volatile slow_to_timestamp_t *)entry)->ll;

            if (ts1.s.rts > self_ts || ts1.s.wts > self_ts || ts1.s.uncommited)
            {
                common::latch_unlock(latch_entry);
                rollback();
                return false;
            }

            winfo->prev_ts.ll = ts1.ll;

            ts1.s.uncommited = 1;
            ts1.s.wts = self_ts;
            ts1.s.rts = self_ts;
            ((volatile slow_to_timestamp_t *)entry)->ll = ts1.ll;
            common::latch_unlock(latch_entry);

#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + tx_idx, obj_idx, ts1.ll, self_ts, self_tid, 1);
#endif

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
        void __device__ rollback()
        {
            self_metrics.abort++;
#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                if (has_wts[i])
                {
                    SlowWriteInfo &winfo = write_info[i];
                    volatile slow_to_timestamp_t &ts = *winfo.tsp;
                    ts.ll = winfo.prev_ts.ll;
                }
            }
            __threadfence();
            self_metrics.abort_duration += clock64() - st_time;
        }
    };

#ifndef NVRTC_COMPILE

    class Slow_TO_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        int *latch_table;
        slow_to_timestamp_t *ts_table;
        SlowWriteInfo *write_info;
        char *has_wts;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        common::TSAllocator_CPU *ts_allocator;
        void *to_gpu_info;
        bool dynamic;

        Slow_TO_CPU(common::DB_CPU *db,
                    common::TransactionSet_CPU *txinfo,
                    size_t bsize,
                    common::TSAllocator_CPU *ts_allocator)
            : info(txinfo),
              db_cpu(db),
              ts_allocator(ts_allocator),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&ts_table, sizeof(slow_to_timestamp_t) * db->table_st[db->table_cnt]);
            cudaMalloc(&latch_table, sizeof(int) * db->table_st[db->table_cnt]);
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                size_t totw = dtx->GetTotW();
                cudaMalloc(&has_wts, sizeof(char) * totw);
                cudaMalloc(&write_info, sizeof(SlowWriteInfo) * totw);
                SlowDynamicTOInfo *tmp = new SlowDynamicTOInfo(write_info, has_wts);
                cudaMalloc(&to_gpu_info, sizeof(SlowDynamicTOInfo));
                cudaMemcpy(to_gpu_info, tmp, sizeof(SlowDynamicTOInfo), cudaMemcpyHostToDevice);
                delete tmp;
            }
            else
            {
                to_gpu_info = nullptr;
            }
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            ts_allocator->Init(batch_id, batch_st);
            cudaMemset(ts_table, 0, sizeof(slow_to_timestamp_t *) * db_cpu->table_st[db_cpu->table_cnt]);
            cudaMemset(latch_table, 0, sizeof(int) * db_cpu->table_st[db_cpu->table_cnt]);
        }
        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            ts_allocator->GetCompileOptions(opts);
            opts.push_back("-D SLOW_TO_RUN");
            opts.push_back("-D SLOW_TO_TS_TABLE=" + std::to_string((unsigned long long)ts_table));
            opts.push_back("-D SLOW_TO_LATCH_TABLE=" + std::to_string((unsigned long long)latch_table));
            opts.push_back("-D CC_TYPE=cc::Slow_TO_GPU");
        }

        void *ToGPU() override
        {
            return to_gpu_info;
        }

        size_t GetMemSize() override
        {
            size_t common_sz = sizeof(slow_to_timestamp_t) * db_cpu->table_st[db_cpu->table_cnt];
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                return common_sz + sizeof(int) * tx_cnt * 2 + sizeof(size_t) * (tx_cnt + 1) + sizeof(char) * dtx->GetTotW() + sizeof(SlowWriteInfo) * dtx->GetTotW() + sizeof(SlowDynamicTOInfo);
            }
            return common_sz;
        }

        void Explain(unsigned long long self_info, unsigned long long target_info) override
        {
            slow_to_timestamp_t tt;
            tt.ll = target_info;
            std::cout << "ts" << self_info << " o " << (self_info == (self_info & ((1UL << 31) - 1))) << " | " << tt.s.uncommited << " r " << tt.s.rts << " w " << tt.s.wts << "\n";
        }
    };

#endif
}

#endif