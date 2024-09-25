#ifndef SILO_H
#define SILO_H

#ifndef NVRTC_COMPILE

// #include <nvrtc.h>
// #include <cuda.h>

#endif

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>

namespace cc
{
    struct __align__(8) silo_timestamp_struct
    {
        unsigned long long lock_bit : 1;
        unsigned long long ts : 63;

        // __host__ __device__ bool operator!=(const timestamp_t &ano) const
        // {
        //     return lock_bit != ano.
        // }
    };

    union __align__(8) silo_timestamp_t
    {
        unsigned long long int ll;
        silo_timestamp_struct s;
    };

    struct __align__(8) silo_replica_entry
    {
        unsigned long long ts;
        silo_timestamp_t *entry;
        void *srcdata;
        void *dstdata;
        size_t size;
    };

    // struct SiloInfo
    // {
    //     silo_timestamp_t *ts_table;
    //     silo_replica_entry *replica_entries;

    //     __host__ __device__ SiloInfo() {}

    //     __host__ __device__ SiloInfo(silo_timestamp_t *ts_table,
    //                                  silo_replica_entry *replica_entries) : ts_table(ts_table),
    //                                                                         replica_entries(replica_entries) {}
    // };

    struct __align__(8) DynamicSiloInfo
    {
        char *for_update;

        __host__ __device__ DynamicSiloInfo() {}
        __host__ __device__ DynamicSiloInfo(char *for_update) : for_update(for_update) {}
    };

#ifndef SILO_RUN

#define SILO_TS_TABLE 0
#define SILO_REPLICA_ENTRIES 0

#endif

    class Silo_GPU
    {
    public:
        common::Metrics self_metrics;
        silo_replica_entry *self_r_replica_entries;
        silo_replica_entry *self_w_replica_entries;
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

        __device__ Silo_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[tid];
            wcnt = txset_info->tx_wcnt[tid];
            DynamicSiloInfo *tinfo = (DynamicSiloInfo *)info;
            for_update = tinfo->for_update + txset_info->tx_rcnt_st[tid];
            self_r_replica_entries = ((silo_replica_entry *)SILO_REPLICA_ENTRIES) + txset_info->tx_opcnt_st[tid];
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif
#else
            self_r_replica_entries = ((silo_replica_entry *)SILO_REPLICA_ENTRIES) + tid * (RCNT + WCNT);
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + (RCNT + WCNT + 1) * self_tid;
#endif
#endif
            self_w_replica_entries = self_r_replica_entries + RCNT;
        }

        __device__ bool TxStart(void *info)
        {
            st_time = clock64();
            memset(for_update, 0, sizeof(char) * RCNT);
            self_metrics.manager_duration = clock64() - st_time;
            self_metrics.wait_duration = 0;
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
                ((silo_timestamp_t *)SILO_TS_TABLE) + obj_idx,
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
                ((silo_timestamp_t *)SILO_TS_TABLE) + obj_idx,
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
            silo_replica_entry &entry = self_w_replica_entries[tx_idx];
            entry.entry = ((silo_timestamp_t *)SILO_TS_TABLE) + obj_idx;
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
        bool __device__ lock(silo_timestamp_t *entry)
        {
            while (true)
            {
                silo_timestamp_t ts1, ts2;
                ts2.ll = ts1.ll = ((volatile silo_timestamp_t *)entry)->ll;
                if (ts1.s.lock_bit)
                    return false;
                // if (!ts1.s.lock_bit)
                // {
                ts2.s.lock_bit = 1;
                if (atomicCAS(&(entry->ll), ts1.ll, ts2.ll) == ts1.ll)
                    break;
                // }
            }
            return true;
        }

        void __device__ unlock(silo_timestamp_t *entry)
        {
            entry->s.lock_bit = 0;
            __threadfence();
        }

        void __device__ get_entry(
            silo_timestamp_t *entry,
            silo_replica_entry *ret,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            ret->entry = entry;
            ret->srcdata = srcdata;
            ret->dstdata = dstdata;
            ret->size = size;
            silo_timestamp_t ts1, ts2;
            // do
            // {
            //     ts1.ll = ((volatile silo_timestamp_t *)entry)->ll;
            //     memcpy(dstdata, srcdata, size);
            //     ts2.ll = ((volatile silo_timestamp_t *)entry)->ll;
            // } while (ts1.ll != ts2.ll || ts1.s.lock_bit);
            while(true)
            {
                unsigned long long wait_st_time = clock64();
                ts1.ll = ((volatile silo_timestamp_t *)entry)->ll;
                memcpy(dstdata, srcdata, size);
                ts2.ll = ((volatile silo_timestamp_t *)entry)->ll;
                if(ts1.ll != ts2.ll || ts1.s.lock_bit) {
                    self_metrics.wait_duration += clock64() - wait_st_time;
                    continue;
                }
                break;
            }
#ifdef TX_DEBUG
                common::AddEvent(self_events + (ret - self_r_replica_entries),
                                 entry - ((silo_timestamp_t *)SILO_TS_TABLE),
                                 ts1.ll, 0, self_tid, 0);
#endif
            ret->ts = ts1.s.ts;
        }

        unsigned long long __device__ validation_phase()
        {
            silo_replica_entry *RS = self_r_replica_entries;
            silo_replica_entry *WS = self_w_replica_entries;

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
                silo_timestamp_t ts;
                ts.ll = ((volatile silo_timestamp_t *)RS[i].entry)->ll;

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
                                 WS[i].entry - ((silo_timestamp_t *)SILO_TS_TABLE),
                                 ((volatile silo_timestamp_t *)WS[i].entry)->ll,
                                 commit_ts + 1, self_tid, 1);
#endif

            return commit_ts + 1;
        }

        void __device__ write_phase(unsigned long long commit_ts)
        {
            silo_replica_entry *WS = self_w_replica_entries;
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + WCNT, 0, 0, 0, self_tid, 2);
#endif
#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                silo_replica_entry &entry = WS[i];
                volatile silo_timestamp_t &ts = *((volatile silo_timestamp_t *)entry.entry);
                memcpy(entry.dstdata, entry.srcdata, entry.size);
                ts.s.ts = commit_ts;
                unlock(entry.entry);
            }
        }
    };

#ifndef NVRTC_COMPILE

    class Silo_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        silo_timestamp_t *ts_table;
        silo_replica_entry *replica_entries;
        char *for_update;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        void *silo_gpu_info;
        bool dynamic;

        Silo_CPU(common::DB_CPU *db, common::TransactionSet_CPU *txinfo, size_t bsize)
            : info(txinfo),
              db_cpu(db),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&ts_table, sizeof(silo_timestamp_t *) * db->table_st[db->table_cnt]);
            cudaMalloc(&replica_entries, sizeof(silo_replica_entry) * txinfo->GetTotOpCnt());

            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                cudaMalloc(&for_update, sizeof(char) * dtx->GetTotR());
                DynamicSiloInfo *tmp = new DynamicSiloInfo(for_update);
                cudaMalloc(&silo_gpu_info, sizeof(DynamicSiloInfo));
                cudaMemcpy(silo_gpu_info, tmp, sizeof(DynamicSiloInfo), cudaMemcpyHostToDevice);
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
            cudaMemset(ts_table, 0, sizeof(silo_timestamp_t *) * db_cpu->table_st[db_cpu->table_cnt]);
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D SILO_RUN");
            opts.push_back(std::string("-D SILO_TS_TABLE=") + std::to_string((unsigned long long)ts_table));
            opts.push_back(std::string("-D SILO_REPLICA_ENTRIES=") + std::to_string((unsigned long long)replica_entries));
            opts.push_back(std::string("-D CC_TYPE=cc::Silo_GPU"));
        }

        void *ToGPU() override
        {
            return silo_gpu_info;
        }

        size_t GetMemSize() override
        {
            size_t common_sz = sizeof(silo_timestamp_t *) * db_cpu->table_st[db_cpu->table_cnt] + sizeof(silo_replica_entry) * info->GetTotOpCnt();
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                return common_sz + sizeof(int) * tx_cnt * 2 + sizeof(size_t) * (tx_cnt + 1) * 2 + sizeof(char) * dtx->GetTotR() + sizeof(DynamicSiloInfo);
            }
            return common_sz;
        }

        void Explain(unsigned long long self_info, unsigned long long target_info) override
        {
            silo_timestamp_t tt;
            tt.ll = target_info;
            std::cout << "ts" << self_info << " o " << (self_info == (self_info & ((1UL << 63) - 1))) << " | lb " << tt.s.lock_bit << " ts " << tt.s.ts << "\n";
        }
    };

#endif
}

#endif