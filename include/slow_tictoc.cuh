#ifndef SLOW_TICTOC_H
#define SLOW_TICTOC_H

#ifndef NVRTC_COMPILE

// #include <nvrtc.h>
// #include <cuda.h>

#endif

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>

namespace cc
{
    struct __align__(8) slow_timestamp_struct
    {
        unsigned long long lock_bit : 1;
        unsigned long long delta : 15;
        unsigned long long wts : 48;
    };

    union __align__(8) slow_timestamp_t
    {
        unsigned long long int ll;
        slow_timestamp_struct s;
    };

    struct __align__(8) slow_replica_entry
    {
        unsigned long long rts;
        unsigned long long wts;
        slow_timestamp_t *entry;
        void *srcdata;
        void *dstdata;
        size_t size;
    };

    struct __align__(8) SlowDynamicTictocInfo
    {
        char *for_update;

        __host__ __device__ SlowDynamicTictocInfo() {}
        __host__ __device__ SlowDynamicTictocInfo(char *for_update) : for_update(for_update) {}
    };

#ifndef SLOW_TICTOC_RUN

#define SLOW_TICTOC_LATCH_TABLE 0
#define SLOW_TICTOC_TS_TABLE 0
#define SLOW_TICTOC_REPLICA_ENTRIES 0

#endif

    class Slow_Tictoc_GPU
    {
    public:
        common::Metrics self_metrics;
        slow_replica_entry *self_r_replica_entries;
        slow_replica_entry *self_w_replica_entries;
        size_t self_tid;
        unsigned long long st_time;

#ifdef DYNAMIC_RW_COUNT
        common::DynamicTransactionSet_GPU *txset_info;
        char *for_update;
        int rcnt;
        int wcnt;
#else
        char for_update[RCNT];
#endif

#ifdef TX_DEBUG
        common::Event *self_events;
#endif

        __device__ Slow_Tictoc_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[tid];
            wcnt = txset_info->tx_wcnt[tid];
            SlowDynamicTictocInfo *tinfo = (SlowDynamicTictocInfo *)info;
            for_update = tinfo->for_update + txset_info->tx_rcnt_st[tid];
            self_r_replica_entries = ((slow_replica_entry *)SLOW_TICTOC_REPLICA_ENTRIES) + txset_info->tx_opcnt_st[tid];
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif

#else
            self_r_replica_entries = ((slow_replica_entry *)SLOW_TICTOC_REPLICA_ENTRIES) + tid * (RCNT + WCNT);
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + (RCNT + WCNT + 1) * self_tid;
#endif

#endif
            self_w_replica_entries = self_r_replica_entries + RCNT;
        }

        __device__ bool TxStart(void *info)
        {
            st_time = clock64();
            self_metrics.wait_duration = 0;
            memset(for_update, 0, sizeof(char) * RCNT);
            self_metrics.manager_duration = clock64() - st_time;
            return true;
        }

        __device__ bool TxEnd(void *info)
        {
            unsigned long long manager_st_time = clock64();
            unsigned long long commit_ts = validation_phase();
            if (commit_ts == ~0ULL)
                return false;
            write_phase(commit_ts);
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
            get_entry(
                ((slow_timestamp_t *)SLOW_TICTOC_TS_TABLE) + obj_idx,
                self_r_replica_entries + tx_idx,
                srcdata,
                dstdata,
                size);
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
            unsigned long long manager_st_time = clock64();
            get_entry(
                ((slow_timestamp_t *)SLOW_TICTOC_TS_TABLE) + obj_idx,
                self_r_replica_entries + tx_idx,
                srcdata,
                dstdata,
                size);
            for_update[tx_idx] = 1;
            self_metrics.manager_duration += clock64() - manager_st_time;
            return true;
        }

        __device__ bool Write(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            unsigned long long manager_st_time = clock64();
            slow_replica_entry &entry = self_w_replica_entries[tx_idx];
            entry.entry = ((slow_timestamp_t *)SLOW_TICTOC_TS_TABLE) + obj_idx;
            entry.srcdata = srcdata;
            entry.dstdata = dstdata;
            entry.size = size;
            self_metrics.manager_duration += clock64() - manager_st_time;
            return true;
        }

        __device__ void Finalize()
        {
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->abort), self_metrics.abort);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->wait_duration), self_metrics.wait_duration);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->abort_duration), self_metrics.abort_duration);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->manager_duration), self_metrics.manager_duration);
        }

    private:
        bool __device__ lock(slow_timestamp_t *entry)
        {
            int *latch_entry = (int *)SLOW_TICTOC_LATCH_TABLE + (entry - (slow_timestamp_t *)SLOW_TICTOC_TS_TABLE);
            common::latch_lock(self_metrics, latch_entry);
            slow_timestamp_t ts1;
            ts1.ll = ((volatile slow_timestamp_t *)entry)->ll;
            if (ts1.s.lock_bit)
            {
                common::latch_unlock(latch_entry);
                return false;
            }
            ts1.s.lock_bit = 1;
            ((volatile slow_timestamp_t *)entry)->ll = ts1.ll;
            common::latch_unlock(latch_entry);
            return true;
        }

        void __device__ unlock(slow_timestamp_t *entry)
        {
            ((volatile slow_timestamp_t *)entry)->s.lock_bit = 0;
            __threadfence();
        }

        void __device__ get_entry(
            slow_timestamp_t *entry,
            slow_replica_entry *ret,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            ret->srcdata = srcdata;
            ret->dstdata = dstdata;
            ret->size = size;

            int *latch_entry = (int *)SLOW_TICTOC_LATCH_TABLE + (entry - (slow_timestamp_t *)SLOW_TICTOC_TS_TABLE);

            slow_timestamp_t ts1;
            while (true)
            {
                common::latch_lock(self_metrics, latch_entry);
                ts1.ll = ((volatile slow_timestamp_t *)entry)->ll;
                if (!ts1.s.lock_bit)
                {
                    memcpy(dstdata, srcdata, size);
                    common::latch_unlock(latch_entry);
                    break;
                }
                common::latch_unlock(latch_entry);
            }

#ifdef TX_DEBUG
            common::AddEvent(self_events + (ret - self_r_replica_entries),
                             entry - ((slow_timestamp_t *)SLOW_TICTOC_TS_TABLE),
                             0, 0, self_tid, 0);
#endif
            ret->wts = ts1.s.wts;
            ret->rts = ts1.s.wts + ts1.s.delta;
            ret->entry = entry;
        }

        unsigned long long __device__ validation_phase()
        {
            slow_replica_entry *RS = self_r_replica_entries;
            slow_replica_entry *WS = self_w_replica_entries;

            // TODO sort WS
            long long lock_st_time = clock64();
            {
#pragma unroll
                for (int i = 0; i < WCNT; i++)
                {
                    if (!lock(WS[i].entry))
                    {
                        for (int j = 0; j < i; j++)
                            unlock(WS[j].entry);
                        self_metrics.abort++;
                        self_metrics.abort_duration += clock64() - st_time;
                        return ~0ULL;
                    }
                }
            }

            self_metrics.wait_duration += clock64() - lock_st_time;

            unsigned long long commit_ts = 0;
            unsigned long long rts;
            slow_timestamp_t ts;

#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                ts.ll = WS[i].entry->ll;
                rts = ts.s.wts + ts.s.delta;
                commit_ts = max(commit_ts, rts + 1);
            }

#pragma unroll
            for (int i = 0; i < RCNT; i++)
                commit_ts = max(commit_ts, RS[i].wts);

            for (int i = 0; i < RCNT; i++)
            {
                if (RS[i].rts < commit_ts)
                {
                    int *latch_entry = (int *)SLOW_TICTOC_LATCH_TABLE + (RS[i].entry - (slow_timestamp_t *)SLOW_TICTOC_TS_TABLE);
                    common::latch_lock(self_metrics, latch_entry);

                    slow_timestamp_t ts1;
                    ts1.ll = ((volatile slow_timestamp_t *)RS[i].entry)->ll;
                    unsigned long long wts, rts;
                    wts = ts1.s.wts;
                    rts = ts1.s.wts + ts1.s.delta;
                    if (
                        RS[i].wts != wts ||
                        (rts <= commit_ts &&
                         ts1.s.lock_bit &&
                         !for_update[i]))
                    {
                        for (int j = 0; j < WCNT; j++)
                            unlock(WS[j].entry);
                        self_metrics.abort++;
                        self_metrics.abort_duration += clock64() - st_time;

                        int *latch_entry = (int *)SLOW_TICTOC_LATCH_TABLE + (RS[i].entry - (slow_timestamp_t *)SLOW_TICTOC_TS_TABLE);
                        common::latch_unlock(latch_entry);

                        return ~0ULL;
                    }

                    if (rts <= commit_ts)
                    {
                        unsigned long long delta = commit_ts - wts;
                        unsigned long long shift = delta - (delta & 0x7fffULL);
                        ts1.s.wts += shift;
                        ts1.s.delta = delta - shift;
                        ((volatile slow_timestamp_t *)RS[i].entry)->ll = ts1.ll;
                    }
                    latch_entry = (int *)SLOW_TICTOC_LATCH_TABLE + (RS[i].entry - (slow_timestamp_t *)SLOW_TICTOC_TS_TABLE);
                    common::latch_unlock(latch_entry);
                }
            }

#ifdef TX_DEBUG
            for (int i = 0; i < WCNT; i++)
                common::AddEvent(self_events + RCNT + i,
                                 WS[i].entry - ((slow_timestamp_t *)SLOW_TICTOC_TS_TABLE),
                                 0, 0, self_tid, 1);
#endif
            return commit_ts;
        }

        void __device__ write_phase(unsigned long long commit_ts)
        {
            slow_replica_entry *WS = self_w_replica_entries;
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + WCNT, 0, 0, 0, self_tid, 2);
#endif
#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                slow_replica_entry &entry = WS[i];
                volatile slow_timestamp_t &ts = *((volatile slow_timestamp_t *)entry.entry);
                memcpy(entry.dstdata, entry.srcdata, entry.size);
                ts.s.wts = commit_ts;
                ts.s.delta = 0;
                unlock(entry.entry);
            }
        }
    };

#ifndef NVRTC_COMPILE

    class Slow_Tictoc_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        int *latch_table;
        slow_timestamp_t *ts_table;
        slow_replica_entry *replica_entries;
        char *for_update;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        void *tictoc_gpu_info;
        bool dynamic;

        Slow_Tictoc_CPU(common::DB_CPU *db, common::TransactionSet_CPU *txinfo, size_t bsize)
            : info(txinfo),
              db_cpu(db),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&ts_table, sizeof(slow_timestamp_t *) * db->table_st[db->table_cnt]);
            cudaMalloc(&latch_table, sizeof(int) * db->table_st[db->table_cnt]);
            cudaMalloc(&replica_entries, sizeof(slow_replica_entry) * txinfo->GetTotOpCnt());

            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                cudaMalloc(&for_update, sizeof(char) * dtx->GetTotR());
                SlowDynamicTictocInfo *tmp = new SlowDynamicTictocInfo(for_update);
                cudaMalloc(&tictoc_gpu_info, sizeof(SlowDynamicTictocInfo));
                cudaMemcpy(tictoc_gpu_info, tmp, sizeof(SlowDynamicTictocInfo), cudaMemcpyHostToDevice);
                delete tmp;
            }
            else
            {
                tictoc_gpu_info = nullptr;
            }
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            cudaMemset(ts_table, 0, sizeof(slow_timestamp_t *) * db_cpu->table_st[db_cpu->table_cnt]);
            cudaMemset(latch_table, 0, sizeof(int) * db_cpu->table_st[db_cpu->table_cnt]);
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D SLOW_TICTOC_RUN");
            opts.push_back(std::string("-D SLOW_TICTOC_TS_TABLE=") + std::to_string((unsigned long long)ts_table));
            opts.push_back(std::string("-D SLOW_TICTOC_REPLICA_ENTRIES=") + std::to_string((unsigned long long)replica_entries));
            opts.push_back("-D SLOW_TICTOC_LATCH_TABLE=" + std::to_string((unsigned long long)latch_table));
            opts.push_back(std::string("-D CC_TYPE=cc::Slow_Tictoc_GPU"));
        }

        void *ToGPU() override
        {
            return tictoc_gpu_info;
        }

        size_t GetMemSize() override
        {
            size_t common_sz = sizeof(slow_timestamp_t *) * db_cpu->table_st[db_cpu->table_cnt] + sizeof(slow_replica_entry) * info->GetTotOpCnt();
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                return common_sz + sizeof(int) * tx_cnt * 2 + sizeof(size_t) * (tx_cnt + 1) * 2 + sizeof(char) * dtx->GetTotR() + sizeof(SlowDynamicTictocInfo);
            }
            return common_sz;
        }
    };

#endif
}

#endif