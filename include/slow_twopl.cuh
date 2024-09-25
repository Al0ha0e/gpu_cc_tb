#ifndef SLOW_TWOPL_H
#define SLOW_TWOPL_H

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>

namespace cc
{
    struct __align__(8) slow_lock_info_struct
    {
        unsigned long long shared : 1;
        unsigned long long holder_cnt : 31;
        unsigned long long holder : 31;
    };

    union __align__(8) slow_lock_info_t
    {
        unsigned long long int ll;
        slow_lock_info_struct s;
    };

    struct __align__(8) SlowTPLWriteInfo
    {
        void *srcdata;
        void *dstdata;
        size_t size;
        slow_lock_info_t *lock;
    };

    struct __align__(8) SlowDynamicTwoPLInfo
    {
        char *for_update;

        __host__ __device__ SlowDynamicTwoPLInfo() {}
        __host__ __device__ SlowDynamicTwoPLInfo(char *for_update) : for_update(for_update) {}
    };

#ifndef SLOW_TPL_RUN
#define SLOW_TPL_LATCH_TABLE 0
#define SLOW_LOCK_TABLE 0
#define SLOW_LOCK_HOLD_INFO_R 0
#define SLOW_LOCK_HOLD_INFO_W 0
#endif

    class Slow_TwoPL_GPU
    {
    public:
        common::Metrics self_metrics;
        slow_lock_info_t **lock_hold_info_r;
        SlowTPLWriteInfo *lock_hold_info_w;
        size_t self_tid;
        int max_r;
        int max_w;
        unsigned long long st_time;

#ifdef TX_DEBUG
        common::Event *self_events;
#endif

#ifdef DYNAMIC_RW_COUNT
        common::DynamicTransactionSet_GPU *txset_info;
        char *for_update;
        int rcnt;
        int wcnt;
#else
        char for_update[RCNT];
#endif

        __device__ Slow_TwoPL_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[self_tid];
            wcnt = txset_info->tx_wcnt[self_tid];
            SlowDynamicTwoPLInfo *tinfo = (SlowDynamicTwoPLInfo *)info;
            lock_hold_info_r = ((slow_lock_info_t **)SLOW_LOCK_HOLD_INFO_R) + txset_info->tx_rcnt_st[self_tid];
            lock_hold_info_w = ((SlowTPLWriteInfo *)SLOW_LOCK_HOLD_INFO_W) + txset_info->tx_wcnt_st[self_tid];
            for_update = tinfo->for_update + txset_info->tx_rcnt_st[self_tid];

#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif

#else
            lock_hold_info_r = ((slow_lock_info_t **)SLOW_LOCK_HOLD_INFO_R) + RCNT * self_tid;
            lock_hold_info_w = ((SlowTPLWriteInfo *)SLOW_LOCK_HOLD_INFO_W) + WCNT * self_tid;

#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + (RCNT + WCNT + 1) * self_tid;
#endif

#endif
        }

        __device__ bool TxStart(void *info)
        {
            st_time = clock64();
            self_metrics.wait_duration = 0;
            memset(for_update, 0, sizeof(bool) * RCNT);
            max_r = max_w = -1;
            self_metrics.manager_duration = clock64() - st_time;
            return true;
        }

        __device__ bool TxEnd(void *info)
        {
            // printf("%lld COMMIT\n", self_tid);
            unsigned long long manager_st_time = clock64();
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + WCNT, 0, 0, 0, self_tid, 2);
#endif
#pragma unroll
            for (int i = 0; i < RCNT; i++)
            {
                if (for_update[i])
                    unlock_exclusive(lock_hold_info_r[i]);
                else
                    unlock_shared(lock_hold_info_r[i]);
            }

#pragma unroll
            for (int i = 0; i < WCNT; i++)
            {
                SlowTPLWriteInfo &winfo = lock_hold_info_w[i];
                memcpy(winfo.dstdata, winfo.srcdata, winfo.size);
                unlock_exclusive(winfo.lock);
            }
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
            slow_lock_info_t *info = ((slow_lock_info_t *)SLOW_LOCK_TABLE) + obj_idx;
            if (!lock_shared(info))
            {
                rollback();
                return false;
            }
            max_r = max(max_r, tx_idx);
            lock_hold_info_r[tx_idx] = info;
            memcpy(dstdata, srcdata, size);
            self_metrics.manager_duration += clock64() - manager_st_time;
#ifdef TX_DEBUG
            common::AddEvent(self_events + tx_idx, obj_idx, info->ll, 0, self_tid, 0);
#endif
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
            slow_lock_info_t *info = ((slow_lock_info_t *)SLOW_LOCK_TABLE) + obj_idx;
            if (!lock_exclusive(info))
            {
                rollback();
                return false;
            }
            max_r = max(max_r, tx_idx);
            for_update[tx_idx] = 1;
            lock_hold_info_r[tx_idx] = info;
            memcpy(dstdata, srcdata, size);
            self_metrics.manager_duration += clock64() - manager_st_time;
#ifdef TX_DEBUG
            common::AddEvent(self_events + tx_idx, obj_idx, info->ll, 0, self_tid, 0);
#endif
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
            slow_lock_info_t *info = ((slow_lock_info_t *)SLOW_LOCK_TABLE) + obj_idx;
            if (!lock_exclusive(info))
            {
                rollback();
                return false;
            }
            max_w = max(max_w, tx_idx);
            SlowTPLWriteInfo &winfo = lock_hold_info_w[tx_idx];
            winfo.srcdata = srcdata;
            winfo.dstdata = dstdata;
            winfo.size = size;
            winfo.lock = info;
            self_metrics.manager_duration += clock64() - manager_st_time;
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + tx_idx, obj_idx, info->ll, 0, self_tid, 1);
#endif
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
        // void __device__ common::latch_lock(int *latch)
        // {
        //     unsigned long long wait_st_time = clock64();
        //     while (atomicCAS(latch, 0, 1)){
        //         //printf("LATCH LOCK %lld\n",self_tid);
        //     }
        //     self_metrics.wait_duration += clock64() - wait_st_time;
        //     //__threadfence();
        // }

        // void __device__ common::latch_unlock(int *latch)
        // {
        //     __threadfence();
        //     *latch = 0;
        //     __threadfence();
        // }

        __device__ bool lock_shared(slow_lock_info_t *info)
        {
            int *latch_entry = (int *)SLOW_TPL_LATCH_TABLE + (info - (slow_lock_info_t *)SLOW_LOCK_TABLE);
            while (true)
            {
                common::latch_lock(self_metrics, latch_entry);
                slow_lock_info_t ts1;
                ts1.ll = ((volatile slow_lock_info_t *)info)->ll;
#ifdef SLOW_WAIT_DIE
                unsigned long long wait_st_time = clock64();
#endif
                if (ts1.s.shared)
                {
                    ts1.s.holder_cnt++;
                    ((volatile slow_lock_info_t *)info)->ll = ts1.ll;
                    common::latch_unlock(latch_entry);
                    break;
                }
#ifdef SLOW_WAIT_DIE
                if (ts1.s.holder_cnt > 0)
                {
                    common::latch_unlock(latch_entry);
                    if (ts1.s.holder < self_tid)
                        return false;
                    if (ts1.s.holder == self_tid)
                        return true;
                    self_metrics.wait_duration += clock64() - wait_st_time;
                    continue;
                }
#else
                if (ts1.s.holder_cnt > 0)
                {
                    common::latch_unlock(latch_entry);
                    return ts1.s.holder == self_tid;
                }

#endif
                ts1.s.shared = true;
                ts1.s.holder_cnt = 1;
                ((volatile slow_lock_info_t *)info)->ll = ts1.ll;
                common::latch_unlock(latch_entry);
                break;
            }
            return true;
        }

        __device__ bool lock_exclusive(slow_lock_info_t *info)
        {
            int *latch_entry = (int *)SLOW_TPL_LATCH_TABLE + (info - (slow_lock_info_t *)SLOW_LOCK_TABLE);
            while (true)
            {
                common::latch_lock(self_metrics, latch_entry);
#ifdef SLOW_WAIT_DIE
                unsigned long long wait_st_time = clock64();
#endif
                slow_lock_info_t ts1;
                ts1.ll = ((volatile slow_lock_info_t *)info)->ll;
                if (ts1.s.holder_cnt == 0)
                {
                    ts1.s.shared = false;
                    ts1.s.holder_cnt = 1;
                    ts1.s.holder = self_tid;
                    ((volatile slow_lock_info_t *)info)->ll = ts1.ll;
                    common::latch_unlock(latch_entry);
                    break;
                }

                common::latch_unlock(latch_entry);
                if (ts1.s.shared)
                    return false;
#ifdef SLOW_WAIT_DIE
                if (ts1.s.holder < self_tid)
                    return false;
                if (ts1.s.holder == self_tid)
                    return true;
                self_metrics.wait_duration += clock64() - wait_st_time;
                continue;
#else
                return ts1.s.holder == self_tid;
#endif
            }
            return true;
        }

        __device__ void unlock_shared(slow_lock_info_t *lock)
        {
            int *latch_entry = (int *)SLOW_TPL_LATCH_TABLE + (lock - (slow_lock_info_t *)SLOW_LOCK_TABLE);
            common::latch_lock(self_metrics, latch_entry);
            slow_lock_info_t ts1;
            ts1.ll = ((volatile slow_lock_info_t *)lock)->ll;
            ts1.s.holder_cnt--;
            ((volatile slow_lock_info_t *)lock)->ll = ts1.ll;
            common::latch_unlock(latch_entry);
        }

        __device__ void unlock_exclusive(slow_lock_info_t *lock)
        {
            assert(((volatile slow_lock_info_t *)lock)->s.holder == self_tid);
            ((volatile slow_lock_info_t *)lock)->s.holder_cnt = 0;
            __threadfence();
        }

        __device__ void rollback()
        {
            self_metrics.abort++;
            // printf("%lld RB %lld\n", self_tid, self_metrics.abort);
#pragma unroll
            for (int i = 0; i <= max_r; i++)
            {
                if (for_update[i])
                    unlock_exclusive(lock_hold_info_r[i]);
                else
                    unlock_shared(lock_hold_info_r[i]);
            }

#pragma unroll
            for (int i = 0; i <= max_w; i++)
                unlock_exclusive(lock_hold_info_w[i].lock);

            self_metrics.abort_duration += clock64() - st_time;
        }
    };

#ifndef NVRTC_COMPILE

    class Slow_TwoPL_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        int *latch_table;
        slow_lock_info_t *lock_table;
        slow_lock_info_t **lock_hold_info_r;
        SlowTPLWriteInfo *lock_hold_info_w;
        char *for_update;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        void *twopl_gpu_info;
        bool dynamic;
        bool wait_die;

        Slow_TwoPL_CPU(common::DB_CPU *db, common::TransactionSet_CPU *txinfo, bool wait_die, size_t bsize)
            : info(txinfo),
              db_cpu(db),
              wait_die(wait_die),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&lock_table, sizeof(slow_lock_info_t) * db->table_st[db->table_cnt]);
            cudaMalloc(&latch_table, sizeof(int) * db->table_st[db->table_cnt]);

            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dyinfo = dynamic_cast<common::DynamicTransactionSet_CPU *>(info);
                cudaMalloc(&lock_hold_info_r, sizeof(slow_lock_info_t *) * dyinfo->GetTotR());
                cudaMalloc(&lock_hold_info_w, sizeof(SlowTPLWriteInfo) * dyinfo->GetTotW());
                cudaMalloc(&for_update, sizeof(char) * dyinfo->GetTotR());

                SlowDynamicTwoPLInfo *tmp = new SlowDynamicTwoPLInfo(for_update);
                cudaMalloc(&twopl_gpu_info, sizeof(SlowDynamicTwoPLInfo));
                cudaMemcpy(twopl_gpu_info, tmp, sizeof(SlowDynamicTwoPLInfo), cudaMemcpyHostToDevice);
                delete tmp;
            }
            else
            {
                common::StaticTransactionSet_CPU *stinfo = dynamic_cast<common::StaticTransactionSet_CPU *>(info);
                cudaMalloc(&lock_hold_info_r, sizeof(slow_lock_info_t *) * stinfo->rcnt * stinfo->tx_cnt);
                cudaMalloc(&lock_hold_info_w, sizeof(SlowTPLWriteInfo) * stinfo->wcnt * stinfo->tx_cnt);

                twopl_gpu_info = nullptr;
            }
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            cudaMemset(lock_table, 0, sizeof(slow_lock_info_t) * db_cpu->table_st[db_cpu->table_cnt]);
            cudaMemset(latch_table, 0, sizeof(int) * db_cpu->table_st[db_cpu->table_cnt]);
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D SLOW_TPL_RUN");
            opts.push_back(std::string("-D SLOW_LOCK_TABLE=") + std::to_string((unsigned long long)lock_table));
            opts.push_back(std::string("-D SLOW_LOCK_HOLD_INFO_R=") + std::to_string((unsigned long long)lock_hold_info_r));
            opts.push_back(std::string("-D SLOW_LOCK_HOLD_INFO_W=") + std::to_string((unsigned long long)lock_hold_info_w));
            opts.push_back("-D SLOW_TPL_LATCH_TABLE=" + std::to_string((unsigned long long)latch_table));
            opts.push_back(std::string("-D CC_TYPE=cc::Slow_TwoPL_GPU"));
            if (wait_die)
                opts.push_back(std::string("-D SLOW_WAIT_DIE"));
        }

        void *ToGPU() override
        {
            return twopl_gpu_info;
        }

        size_t GetMemSize() override
        {
            size_t common_sz = sizeof(slow_lock_info_t) * db_cpu->table_st[db_cpu->table_cnt];
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dyinfo = (common::DynamicTransactionSet_CPU *)info;
                return common_sz +
                       sizeof(SlowDynamicTwoPLInfo) +
                       sizeof(slow_lock_info_t *) * dyinfo->GetTotR() + sizeof(SlowTPLWriteInfo) * dyinfo->GetTotW() +
                       sizeof(bool) * dyinfo->GetTotR() +
                       sizeof(int) * tx_cnt * 2 +
                       sizeof(size_t) * (tx_cnt + 1) * 2;
            }
            common::StaticTransactionSet_CPU *stinfo = dynamic_cast<common::StaticTransactionSet_CPU *>(info);

            return common_sz +
                   sizeof(slow_lock_info_t *) * stinfo->rcnt * stinfo->tx_cnt +
                   sizeof(SlowTPLWriteInfo) * stinfo->wcnt * stinfo->tx_cnt;
        }

        void Explain(unsigned long long self_info, unsigned long long target_info) override {}
    };

#endif
}

#endif