#ifndef SLOW_SILO_H
#define SLOW_SILO_H

#ifndef NVRTC_COMPILE

// #include <nvrtc.h>
// #include <cuda.h>

#endif

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>

namespace cc
{
    struct __align__(8) slow_silo_timestamp_struct
    {
        unsigned long long lock_bit : 1;
        unsigned long long ts : 63;

        // __host__ __device__ bool operator!=(const timestamp_t &ano) const
        // {
        //     return lock_bit != ano.
        // }
    };

    union __align__(8) slow_silo_timestamp_t
    {
        unsigned long long int ll;
        slow_silo_timestamp_struct s;
    };

    struct __align__(8) slow_silo_replica_entry
    {
        unsigned long long ts;
        slow_silo_timestamp_t *entry;
        void *srcdata;
        void *dstdata;
        size_t size;
    };

    struct __align__(8) SlowDynamicSiloInfo
    {
        char *for_update;

        __host__ __device__ SlowDynamicSiloInfo() {}
        __host__ __device__ SlowDynamicSiloInfo(char *for_update) : for_update(for_update) {}
    };

#ifndef SLOW_SILO_RUN

#define SLOW_SILO_LATCH_TABLE 0
#define SLOW_SILO_TS_TABLE 0
#define SLOW_SILO_REPLICA_ENTRIES 0

#endif

    class Slow_Silo_GPU
    {
    public:
        common::Metrics self_metrics;
        slow_silo_replica_entry *self_r_replica_entries;
        slow_silo_replica_entry *self_w_replica_entries;
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

        __device__ Slow_Silo_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[tid];
            wcnt = txset_info->tx_wcnt[tid];
            SlowDynamicSiloInfo *tinfo = (SlowDynamicSiloInfo *)info;
            for_update = tinfo->for_update + txset_info->tx_rcnt_st[tid];
            self_r_replica_entries = ((slow_silo_replica_entry *)SLOW_SILO_REPLICA_ENTRIES) + txset_info->tx_opcnt_st[tid];
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif
#else
            self_r_replica_entries = ((slow_silo_replica_entry *)SLOW_SILO_REPLICA_ENTRIES) + tid * (RCNT + WCNT);
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
                ((slow_silo_timestamp_t *)SLOW_SILO_TS_TABLE) + obj_idx,
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
                ((slow_silo_timestamp_t *)SLOW_SILO_TS_TABLE) + obj_idx,
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
            slow_silo_replica_entry &entry = self_w_replica_entries[tx_idx];
            entry.entry = ((slow_silo_timestamp_t *)SLOW_SILO_TS_TABLE) + obj_idx;
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
        bool __device__ lock(slow_silo_timestamp_t *entry)
        {
            int *latch_entry = (int *)SLOW_SILO_LATCH_TABLE + (entry - (slow_silo_timestamp_t *)SLOW_SILO_TS_TABLE);
            common::latch_lock(self_metrics, latch_entry);
            slow_silo_timestamp_t ts1;
            ts1.ll = ((volatile slow_silo_timestamp_t *)entry)->ll;
            if (ts1.s.lock_bit)
            {
                common::latch_unlock(latch_entry);
                return false;
            }
            ts1.s.lock_bit = 1;
            ((volatile slow_silo_timestamp_t *)entry)->ll = ts1.ll;
            common::latch_unlock(latch_entry);
            return true;
        }

        void __device__ unlock(slow_silo_timestamp_t *entry)
        {
            ((volatile slow_silo_timestamp_t *)entry)->s.lock_bit = 0;
            __threadfence();
        }

        void __device__ get_entry(
            slow_silo_timestamp_t *entry,
            slow_silo_replica_entry *ret,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            ret->entry = entry;
            ret->srcdata = srcdata;
            ret->dstdata = dstdata;
            ret->size = size;

            int *latch_entry = (int *)SLOW_SILO_LATCH_TABLE + (entry - (slow_silo_timestamp_t *)SLOW_SILO_TS_TABLE);

            slow_silo_timestamp_t ts1;
            while (true)
            {
                common::latch_lock(self_metrics, latch_entry);
                ts1.ll = ((volatile slow_silo_timestamp_t *)entry)->ll;
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
                             entry - ((slow_silo_timestamp_t *)SLOW_SILO_TS_TABLE),
                             ts1.ll, 0, self_tid, 0);
#endif
            ret->ts = ts1.s.ts;
        }

        unsigned long long __device__ validation_phase()
        {
            slow_silo_replica_entry *RS = self_r_replica_entries;
            slow_silo_replica_entry *WS = self_w_replica_entries;

            long long lock_st_time = clock64();

            { // TODO sort WS
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

#pragma unroll
            for (int i = 0; i < RCNT; i++)
                commit_ts = max(commit_ts, RS[i].ts);

            for (int i = 0; i < RCNT; i++)
            {
                slow_silo_timestamp_t ts;
                ts.ll = ((volatile slow_silo_timestamp_t *)RS[i].entry)->ll;

                if (ts.s.ts != RS[i].ts || (ts.s.lock_bit && !for_update[i]))
                {
                    for (int j = 0; j < WCNT; j++)
                        unlock(WS[j].entry);
                    self_metrics.abort++;
                    self_metrics.abort_duration += clock64() - st_time;
                    return ~0ULL;
                }
            }

            {
#pragma unroll
                for (int i = 0; i < WCNT; i++)
                    commit_ts = max(commit_ts, WS[i].ts);
            }

#ifdef TX_DEBUG
            for (int i = 0; i < WCNT; i++)
                common::AddEvent(self_events + RCNT + i,
                                 WS[i].entry - ((slow_silo_timestamp_t *)SLOW_SILO_TS_TABLE),
                                 ((volatile slow_silo_timestamp_t *)WS[i].entry)->ll,
                                 commit_ts + 1, self_tid, 1);
#endif

            return commit_ts + 1;
        }

        void __device__ write_phase(unsigned long long commit_ts)
        {
            slow_silo_replica_entry *WS = self_w_replica_entries;
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + WCNT, 0, 0, 0, self_tid, 2);
#endif
#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                slow_silo_replica_entry &entry = WS[i];
                volatile slow_silo_timestamp_t &ts = *((volatile slow_silo_timestamp_t *)entry.entry);
                memcpy(entry.dstdata, entry.srcdata, entry.size);
                ts.s.ts = commit_ts;
                unlock(entry.entry);
            }
        }
    };

#ifndef NVRTC_COMPILE

    class Slow_Silo_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        int *latch_table;
        slow_silo_timestamp_t *ts_table;
        slow_silo_replica_entry *replica_entries;
        char *for_update;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        void *silo_gpu_info;
        bool dynamic;

        Slow_Silo_CPU(common::DB_CPU *db, common::TransactionSet_CPU *txinfo, size_t bsize)
            : info(txinfo),
              db_cpu(db),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&ts_table, sizeof(slow_silo_timestamp_t *) * db->table_st[db->table_cnt]);
            cudaMalloc(&latch_table, sizeof(int) * db->table_st[db->table_cnt]);
            cudaMalloc(&replica_entries, sizeof(slow_silo_replica_entry) * txinfo->GetTotOpCnt());

            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                cudaMalloc(&for_update, sizeof(char) * dtx->GetTotR());
                SlowDynamicSiloInfo *tmp = new SlowDynamicSiloInfo(for_update);
                cudaMalloc(&silo_gpu_info, sizeof(SlowDynamicSiloInfo));
                cudaMemcpy(silo_gpu_info, tmp, sizeof(SlowDynamicSiloInfo), cudaMemcpyHostToDevice);
                delete tmp;
            }
            else
            {
                silo_gpu_info = nullptr;
            }
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            cudaMemset(ts_table, 0, sizeof(slow_silo_timestamp_t *) * db_cpu->table_st[db_cpu->table_cnt]);
            cudaMemset(latch_table, 0, sizeof(int) * db_cpu->table_st[db_cpu->table_cnt]);
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D SLOW_SILO_RUN");
            opts.push_back(std::string("-D SLOW_SILO_TS_TABLE=") + std::to_string((unsigned long long)ts_table));
            opts.push_back(std::string("-D SLOW_SILO_REPLICA_ENTRIES=") + std::to_string((unsigned long long)replica_entries));
            opts.push_back("-D SLOW_SILO_LATCH_TABLE=" + std::to_string((unsigned long long)latch_table));
            opts.push_back(std::string("-D CC_TYPE=cc::Slow_Silo_GPU"));
        }

        void *ToGPU() override
        {
            return silo_gpu_info;
        }

        size_t GetMemSize() override
        {
            size_t common_sz = sizeof(slow_silo_timestamp_t *) * db_cpu->table_st[db_cpu->table_cnt] + sizeof(slow_silo_replica_entry) * info->GetTotOpCnt();
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                return common_sz + sizeof(int) * tx_cnt * 2 + sizeof(size_t) * (tx_cnt + 1) * 2 + sizeof(char) * dtx->GetTotR() + sizeof(SlowDynamicSiloInfo);
            }
            return common_sz;
        }

        void Explain(unsigned long long self_info, unsigned long long target_info) override
        {
            slow_silo_timestamp_t tt;
            tt.ll = target_info;
            std::cout << "ts" << self_info << " o " << (self_info == (self_info & ((1UL << 63) - 1))) << " | lb " << tt.s.lock_bit << " ts " << tt.s.ts << "\n";
        }
    };

#endif
}

#endif