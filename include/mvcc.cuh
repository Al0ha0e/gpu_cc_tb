#ifndef MVCC_H
#define MVCC_H

#include <cc.cuh>
#include <transaction.cuh>
#include <runtime.cuh>
#include <timestamp.cuh>

namespace cc
{

    struct __align__(8) mvcc_timestamp_struct
    {
        unsigned long long uncommited : 1;
        unsigned long long rts : 31;
        unsigned long long wts : 31;
    };

    union __align__(8) mvcc_timestamp_t
    {
        unsigned long long int ll;
        mvcc_timestamp_struct s;
    };

    struct __align__(8) version_node
    {
        mvcc_timestamp_t ts;
        version_node *prev;
    };

    struct __align__(8) mvcc_write_info
    {
        version_node *verp;
        void *srcdata;
        void *dstdata;
        size_t size;
    };

    struct __align__(8) DynamicMVCCInfo
    {
        mvcc_write_info *write_info;
        char *has_wts;

        __host__ __device__ DynamicMVCCInfo() {}
        __host__ __device__ DynamicMVCCInfo(
            mvcc_write_info * write_info,
            char *has_wts) : write_info(write_info),
                             has_wts(has_wts)
        {
        }
    };

#ifndef MVCC_RUN

#define VERSION_TABLE 0
#define VERSION_NODES 0
#define VERSION_DATA 0

#endif

    class MVCC_GPU
    {
    public:
        common::Metrics self_metrics;
        version_node *self_nodes;
        size_t self_ts;
        size_t self_tid;
        unsigned long long st_time;

#ifdef TX_DEBUG
        common::Event *self_events;
#endif

#ifdef DYNAMIC_RW_COUNT
        common::DynamicTransactionSet_GPU *txset_info;
        mvcc_write_info *write_info;
        char *has_wts;
        int rcnt;
        int wcnt;
#else
        mvcc_write_info write_info[WCNT];
        char has_wts[WCNT];
#endif

        __device__ MVCC_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
            memset(&self_metrics, 0, sizeof(common::Metrics));

#ifdef DYNAMIC_RW_COUNT
            txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[tid];
            wcnt = txset_info->tx_wcnt[tid];
            size_t wst = txset_info->tx_wcnt_st[tid];
            DynamicMVCCInfo *tinfo = (DynamicMVCCInfo *)info;
            write_info = tinfo->write_info + wst;
            has_wts = tinfo->has_wts + wst;
            self_nodes = ((version_node *)VERSION_NODES) + wst;

#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif

#else
            self_nodes = ((version_node *)VERSION_NODES) + tid * WCNT;

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
            memset(has_wts, 0, sizeof(char) * WCNT);
            self_metrics.manager_duration = clock64() - st_time;
            self_metrics.ts_duration += st_time - st_time0;
            self_metrics.wait_duration = 0;
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
                mvcc_write_info &winfo = write_info[i];
                volatile version_node *current_version = winfo.verp;

                self_nodes[i].prev = current_version->prev;
                current_version->prev = self_nodes + i;

                auto offset = (self_nodes - (version_node *)VERSION_NODES) + i;
                memcpy((version_node *)VERSION_DATA + offset, winfo.dstdata, winfo.size);
                // memcpy(winfo.dstdata, winfo.srcdata, winfo.size);

                current_version->ts.s.uncommited = 0;
                // __threadfence();
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
            volatile version_node *entry = ((volatile version_node *)VERSION_TABLE) + obj_idx;
            bool fail = true;
            do
            {
                unsigned long long wait_st_time = clock64();
                mvcc_timestamp_t ts1;
                ts1.ll = entry->ts.ll;
                if (!ts1.s.uncommited)
                {
                    if (ts1.s.wts <= self_ts)
                    {
#ifdef TX_DEBUG
                        common::AddEvent(self_events + tx_idx, obj_idx, ts1.ll, self_ts, self_tid, 0);
#endif
                        memcpy(dstdata, srcdata, size);
                        mvcc_timestamp_t ts2;
                        ts2.ll = ts1.ll;
                        ts2.s.rts = max(ts2.s.rts, (unsigned long long)self_ts);
                        fail = atomicCAS((unsigned long long *)(&(entry->ts.ll)), ts1.ll, ts2.ll) != ts1.ll;
                    }
                    else
                    {
                        volatile version_node *version = entry->prev;
                        while (version->ts.s.wts > self_ts)
                        {
                            assert(version != NULL);
                            version = (volatile version_node *)(version->prev);
                        }
#ifdef TX_DEBUG
                        common::AddEvent(self_events + tx_idx, obj_idx, ts1.ll, self_ts, self_tid, 0);
#endif
                        auto offset = version - (version_node *)VERSION_NODES;
                        // memcpy(dstdata, srcdata, size);
                        memcpy(dstdata, (version_node *)VERSION_DATA + offset, size);
                        fail = entry->ts.ll != ts1.ll;
                    }
                    if(fail) self_metrics.wait_duration += clock64() - wait_st_time;
                }
                else
                {
                    self_metrics.wait_duration += clock64() - wait_st_time;
                }

            } while (fail);
            __threadfence();

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
            version_node *entry = ((version_node *)VERSION_TABLE) + obj_idx;
            mvcc_write_info *winfo = write_info + tx_idx;
            winfo->verp = entry;

            mvcc_timestamp_t ts1;
            bool fail = true;
            do
            {
                unsigned long long wait_st_time = clock64();
                ts1.ll = ((volatile mvcc_timestamp_t *)(&(entry->ts)))->ll;
                if (ts1.s.wts > self_ts || ts1.s.rts > self_ts || ts1.s.uncommited) // TODO Thomas
                    break;
                winfo->srcdata = srcdata;
                winfo->dstdata = dstdata; //(void *)(self_nodes + tx_idx);
                winfo->size = size;

                mvcc_timestamp_t ts2;

                ts2.s.uncommited = 1;
                ts2.s.rts = self_ts;
                ts2.s.wts = self_ts;
                fail = atomicCAS(&(entry->ts.ll), ts1.ll, ts2.ll) != ts1.ll;
                if(fail) self_metrics.wait_duration += clock64() - wait_st_time;
            } while (fail);
            __threadfence();

            if (fail)
            {
                rollback();
                return false;
            }

#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + tx_idx, obj_idx, ts1.ll, self_ts, self_tid, 1);
#endif
            self_nodes[tx_idx].ts.ll = ts1.ll;
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
                    mvcc_write_info &winfo = write_info[i];
                    volatile version_node &current_version = *winfo.verp;
                    current_version.ts.ll = self_nodes[i].ts.ll;
                }
            }
            __threadfence();
            self_metrics.abort_duration += clock64() - st_time;
        }
    };

#ifndef NVRTC_COMPILE

    class MVCC_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        version_node *version_table;
        version_node *version_nodes;
        char *version_data;
        mvcc_write_info *write_info;
        char *has_wts;

        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        common::TSAllocator_CPU *ts_allocator;
        void *mvcc_gpu_info;
        bool dynamic;

        MVCC_CPU(common::DB_CPU *db,
                 common::TransactionSet_CPU *txinfo,
                 size_t bsize,
                 common::TSAllocator_CPU *ts_allocator)
            : info(txinfo),
              db_cpu(db),
              ts_allocator(ts_allocator),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), db->table_st[db->table_cnt])
        {
            cudaMalloc(&version_table, sizeof(version_node) * db->table_st[db->table_cnt]);
            // TODO DEBUG
            size_t entry_size = db->tables[0]->entry_size;
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                size_t totw = dtx->GetTotW();
                cudaMalloc(&version_nodes, sizeof(version_node) * totw);
                cudaMalloc(&version_data, entry_size * totw);
                cudaMalloc(&has_wts, sizeof(char) * totw);
                cudaMalloc(&write_info, sizeof(mvcc_write_info) * totw);
                DynamicMVCCInfo *tmp = new DynamicMVCCInfo(write_info, has_wts);
                cudaMalloc(&mvcc_gpu_info, sizeof(DynamicMVCCInfo));
                cudaMemcpy(mvcc_gpu_info, tmp, sizeof(DynamicMVCCInfo), cudaMemcpyHostToDevice);
                delete tmp;
            }
            else
            {
                common::StaticTransactionSet_CPU *stx = (common::StaticTransactionSet_CPU *)info;
                cudaMalloc(&version_nodes, sizeof(version_node) * stx->wcnt * tx_cnt);
                cudaMalloc(&version_data, entry_size * stx->wcnt * tx_cnt);
                mvcc_gpu_info = nullptr;
            }
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            cudaStream_t stream = streams[batch_id];
            ts_allocator->Init(batch_id, batch_st);
            cudaMemset(version_table, 0, sizeof(version_node) * db_cpu->table_st[db_cpu->table_cnt]);
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dtx = (common::DynamicTransactionSet_CPU *)info;
                size_t totw = dtx->GetTotW();
                cudaMemsetAsync(
                    version_nodes + sizeof(version_node) * dtx->tx_wcnt_st[batch_st],
                    0,
                    sizeof(version_node) * (dtx->tx_wcnt_st[batch_st + batches[batch_id]] - dtx->tx_wcnt_st[batch_st]),
                    stream);
            }
            else
            {
                common::StaticTransactionSet_CPU *stx = (common::StaticTransactionSet_CPU *)info;
                cudaMemsetAsync(
                    version_nodes + sizeof(version_node) * stx->wcnt * batch_st,
                    0,
                    sizeof(version_node) * stx->wcnt * batches[batch_id],
                    stream);
            }
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            ts_allocator->GetCompileOptions(opts);
            opts.push_back(std::string("-D MVCC_RUN"));
            opts.push_back(std::string("-D VERSION_TABLE=") + std::to_string((unsigned long long)version_table));
            opts.push_back(std::string("-D VERSION_NODES=") + std::to_string((unsigned long long)version_nodes));
            opts.push_back(std::string("-D VERSION_DATA=") + std::to_string((unsigned long long)version_data));
            opts.push_back(std::string("-D CC_TYPE=cc::MVCC_GPU"));
        }

        void *ToGPU() override
        {
            return mvcc_gpu_info;
        }

        size_t GetMemSize() override
        {
            return 0;
        }

        void Explain(unsigned long long self_info, unsigned long long target_info) override {}
    };

#endif
}

#endif