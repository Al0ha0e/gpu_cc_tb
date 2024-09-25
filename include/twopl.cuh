#ifndef TWOPL_H
#define TWOPL_H

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>

namespace cc
{
    struct lock_info_struct
    {
        unsigned long long shared : 1;
        unsigned long long holder_cnt : 31;
        unsigned long long holder : 31;
    };

    union lock_info_t
    {
        unsigned long long int ll;
        lock_info_struct s;
    };

    struct TPLWriteInfo
    {
        void *srcdata;
        void *dstdata;
        size_t size;
        lock_info_t *lock;
    };

    // struct TwoPLInfo
    // {
    //     lock_info_t *lock_table;
    //     lock_info_t **lock_hold_info_r;
    //     TPLWriteInfo *lock_hold_info_w;

    //     __host__ __device__ TwoPLInfo() {}
    //     __host__ __device__ TwoPLInfo(lock_info_t *lock_table,
    //                                   lock_info_t **lock_hold_info_r,
    //                                   TPLWriteInfo *lock_hold_info_w) : lock_table(lock_table),
    //                                                                     lock_hold_info_r(lock_hold_info_r),
    //                                                                     lock_hold_info_w(lock_hold_info_w) {}
    // };

    struct DynamicTwoPLInfo
    {
        char *for_update;

        __host__ __device__ DynamicTwoPLInfo() {}
        __host__ __device__ DynamicTwoPLInfo(char *for_update) : for_update(for_update) {}
    };

#ifndef TPL_RUN
#define LOCK_TABLE 0
#define LOCK_HOLD_INFO_R 0
#define LOCK_HOLD_INFO_W 0
#endif

    class TwoPL_GPU
    {
    public:
        common::Metrics self_metrics;
        lock_info_t **lock_hold_info_r;
        TPLWriteInfo *lock_hold_info_w;
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

        __device__ TwoPL_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[self_tid];
            wcnt = txset_info->tx_wcnt[self_tid];
            DynamicTwoPLInfo *tinfo = (DynamicTwoPLInfo *)info;
            lock_hold_info_r = ((lock_info_t **)LOCK_HOLD_INFO_R) + txset_info->tx_rcnt_st[self_tid];
            lock_hold_info_w = ((TPLWriteInfo *)LOCK_HOLD_INFO_W) + txset_info->tx_wcnt_st[self_tid];
            for_update = tinfo->for_update + txset_info->tx_rcnt_st[self_tid];

#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif

#else
            lock_hold_info_r = ((lock_info_t **)LOCK_HOLD_INFO_R) + RCNT * self_tid;
            lock_hold_info_w = ((TPLWriteInfo *)LOCK_HOLD_INFO_W) + WCNT * self_tid;

#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + (RCNT + WCNT + 1) * self_tid;
#endif

#endif
        }

        __device__ bool TxStart(void *info)
        {
            st_time = clock64();
            memset(for_update, 0, sizeof(bool) * RCNT);
            max_r = max_w = -1;
            self_metrics.manager_duration = clock64() - st_time;
            self_metrics.wait_duration = 0;
            return true;
        }

        __device__ bool TxEnd(void *info)
        {
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
                TPLWriteInfo &winfo = lock_hold_info_w[i];
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
            lock_info_t *info = ((lock_info_t *)LOCK_TABLE) + obj_idx;
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
            lock_info_t *info = ((lock_info_t *)LOCK_TABLE) + obj_idx;
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
            lock_info_t *info = ((lock_info_t *)LOCK_TABLE) + obj_idx;
            if (!lock_exclusive(info))
            {
                rollback();
                return false;
            }
            max_w = max(max_w, tx_idx);
            TPLWriteInfo &winfo = lock_hold_info_w[tx_idx];
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
        __device__ bool lock_shared(lock_info_t *info)
        {
            bool fail = true;
            do
            {
                unsigned long long wait_st_time = clock64();
                lock_info_t ts1, ts2;
                ts2.ll = ts1.ll = ((volatile lock_info_t *)info)->ll;
                if (ts1.s.shared)
                    ts2.s.holder_cnt++;
#ifdef WAIT_DIE
                else if (ts1.s.holder_cnt > 0)
                {
                    if (ts1.s.holder < self_tid)
                        return false;
                    if (ts1.s.holder == self_tid)
                        return true;
                    self_metrics.wait_duration += clock64() - wait_st_time;
                    continue;
                }
#else
                else if (ts1.s.holder_cnt > 0)
                    return ts2.s.holder == self_tid;
#endif
                else
                {
                    ts2.s.shared = true;
                    ts2.s.holder_cnt = 1;
                }

                fail = atomicCAS(&(info->ll), ts1.ll, ts2.ll) != ts1.ll;
                if(fail) self_metrics.wait_duration += clock64() - wait_st_time;
            } while (fail);
            return true;
        }

        __device__ bool lock_exclusive(lock_info_t *info)
        {
            bool fail = true;
            do
            {
                unsigned long long wait_st_time = clock64();
                lock_info_t ts1, ts2;
                ts2.ll = ts1.ll = ((volatile lock_info_t *)info)->ll;
                if (ts1.s.holder_cnt == 0)
                {
                    ts2.s.shared = false;
                    ts2.s.holder_cnt = 1;
                    ts2.s.holder = self_tid;
                    fail = atomicCAS(&(info->ll), ts1.ll, ts2.ll) != ts1.ll;
                    if(fail) self_metrics.wait_duration += clock64() - wait_st_time;
                }
                else if (ts1.s.shared)
                    return false;
                else
                {
#ifdef WAIT_DIE
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

            } while (fail);
            return true;
        }

        __device__ void unlock_shared(lock_info_t *lock)
        {
            bool fail = true;
            do
            {
                lock_info_t ts1, ts2;
                ts2.ll = ts1.ll = ((volatile lock_info_t *)lock)->ll;
                ts2.s.holder_cnt--;
                fail = atomicCAS(&(lock->ll), ts1.ll, ts2.ll) != ts1.ll;
            } while (fail);
        }

        __device__ void unlock_exclusive(lock_info_t *lock)
        {
            assert(lock->s.holder == self_tid);
            ((volatile lock_info_t *)lock)->s.holder_cnt = 0;
            __threadfence();
        }

        __device__ void rollback()
        {
            self_metrics.abort++;

#ifdef DYNAMIC_RW_COUNT
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
#else

            if (max_r + 1 == RCNT)
            {
#pragma unroll
                for (int i = 0; i < RCNT; i++)
                {
                    if (for_update[i])
                        unlock_exclusive(lock_hold_info_r[i]);
                    else
                        unlock_shared(lock_hold_info_r[i]);
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i <= max_r; i++)
                {
                    if (for_update[i])
                        unlock_exclusive(lock_hold_info_r[i]);
                    else
                        unlock_shared(lock_hold_info_r[i]);
                }
            }

            if (max_w + 1 == WCNT)
            {
#pragma unroll
                for (int i = 0; i < WCNT; i++)
                    unlock_exclusive(lock_hold_info_w[i].lock);
            }
            else
            {
#pragma unroll
                for (int i = 0; i <= max_w; i++)
                    unlock_exclusive(lock_hold_info_w[i].lock);
            }

#endif
            self_metrics.abort_duration += clock64() - st_time;
        }
    };

#ifndef NVRTC_COMPILE

    class TwoPL_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        lock_info_t *lock_table;
        lock_info_t **lock_hold_info_r;
        TPLWriteInfo *lock_hold_info_w;
        char *for_update;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        void *twopl_gpu_info;
        bool dynamic;
        bool wait_die;

        TwoPL_CPU(common::DB_CPU *db, common::TransactionSet_CPU *txinfo, bool wait_die, size_t bsize)
            : info(txinfo),
              db_cpu(db),
              wait_die(wait_die),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&lock_table, sizeof(lock_info_t) * db->table_st[db->table_cnt]);

            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dyinfo = dynamic_cast<common::DynamicTransactionSet_CPU *>(info);
                cudaMalloc(&lock_hold_info_r, sizeof(lock_info_t *) * dyinfo->GetTotR());
                cudaMalloc(&lock_hold_info_w, sizeof(TPLWriteInfo) * dyinfo->GetTotW());
                cudaMalloc(&for_update, sizeof(char) * dyinfo->GetTotR());

                DynamicTwoPLInfo *tmp = new DynamicTwoPLInfo(for_update);
                cudaMalloc(&twopl_gpu_info, sizeof(DynamicTwoPLInfo));
                cudaMemcpy(twopl_gpu_info, tmp, sizeof(DynamicTwoPLInfo), cudaMemcpyHostToDevice);
                delete tmp;
            }
            else
            {
                common::StaticTransactionSet_CPU *stinfo = dynamic_cast<common::StaticTransactionSet_CPU *>(info);
                cudaMalloc(&lock_hold_info_r, sizeof(lock_info_t *) * stinfo->rcnt * stinfo->tx_cnt);
                cudaMalloc(&lock_hold_info_w, sizeof(TPLWriteInfo) * stinfo->wcnt * stinfo->tx_cnt);

                twopl_gpu_info = nullptr;
            }
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            cudaMemset(lock_table, 0, sizeof(lock_info_t) * db_cpu->table_st[db_cpu->table_cnt]);
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D TPL_RUN");
            opts.push_back(std::string("-D LOCK_TABLE=") + std::to_string((unsigned long long)lock_table));
            opts.push_back(std::string("-D LOCK_HOLD_INFO_R=") + std::to_string((unsigned long long)lock_hold_info_r));
            opts.push_back(std::string("-D LOCK_HOLD_INFO_W=") + std::to_string((unsigned long long)lock_hold_info_w));
            opts.push_back(std::string("-D CC_TYPE=cc::TwoPL_GPU"));
            if (wait_die)
                opts.push_back(std::string("-D WAIT_DIE"));
        }

        void *ToGPU() override
        {
            return twopl_gpu_info;
        }

        size_t GetMemSize() override
        {
            size_t common_sz = sizeof(lock_info_t) * db_cpu->table_st[db_cpu->table_cnt];
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dyinfo = (common::DynamicTransactionSet_CPU *)info;
                return common_sz +
                       sizeof(DynamicTwoPLInfo) +
                       sizeof(lock_info_t *) * dyinfo->GetTotR() + sizeof(TPLWriteInfo) * dyinfo->GetTotW() +
                       sizeof(bool) * dyinfo->GetTotR() +
                       sizeof(int) * tx_cnt * 2 +
                       sizeof(size_t) * (tx_cnt + 1) * 2;
            }
            common::StaticTransactionSet_CPU *stinfo = dynamic_cast<common::StaticTransactionSet_CPU *>(info);

            return common_sz +
                   sizeof(lock_info_t *) * stinfo->rcnt * stinfo->tx_cnt +
                   sizeof(TPLWriteInfo) * stinfo->wcnt * stinfo->tx_cnt;
        }

        void Explain(unsigned long long self_info, unsigned long long target_info) override {}
    };

#endif
}

#endif