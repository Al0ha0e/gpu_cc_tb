#include <db.cuh>
#include <transaction.cuh>
#include <runtime.cuh>
#include <cc.cuh>
#include <tictoc.cuh>
#include <gacco.cuh>
#include <gputx.cuh>
#include <silo.cuh>
#include <twopl.cuh>
#include <to.cuh>
#include <mvcc.cuh>
#include <slow_twopl.cuh>
#include <slow_to.cuh>
#include <slow_mvcc.cuh>
#include <slow_tictoc.cuh>
#include <slow_silo.cuh>
#include <ycsb.cuh>

using namespace bench_ycsb;

__launch_bounds__(1024)
    __global__ void ycsb(
        common::DB_GPU *db,
        char *txs,
        void *txset_info,
        int batch_id,
        int batch_st,
        int batch_en,
        void *info,
        char *output)
{
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id & ((1 << THREAD_TX_SHIFT) - 1))
        return;
    size_t tid = batch_st + (thread_id >> THREAD_TX_SHIFT);

    if (tid >= batch_en)
        return;

    unsigned long long tx_start_time = clock64();
    unsigned long long index_duration;

    CC_TYPE cc(txset_info, info, tid);
    tid = cc.self_tid;

    YCSBTx tx = ((YCSBTx *)txs)[tid];

    common::Table_GPU table = db->tables[0];
    INDEX_0_TYPE<unsigned int> *index = (INDEX_0_TYPE<unsigned int> *)table.indices[0];

    bool tx_success = false;
    while (!tx_success)
    {
        if (!cc.TxStart(info))
            continue;

        Item item;
        int r_idx = 0, w_idx = 0;
        index_duration = 0;
        bool success1 = true;

        for (int i = 0; i < tx.request_cnt; i++)
        {
            YCSBReq &req = tx.requests[i];

            unsigned long long index_start_time = clock64();
            size_t idx = index->Find(req.key);
            index_duration += clock64() - index_start_time;

            if (req.read)
            {
                if (!cc.Read(
                        idx,
                        r_idx,
                        table.data + idx * sizeof(Item),
                        &item,
                        sizeof(Item)))
                {
                    success1 = false;
                    break;
                }
                r_idx++;
            }
            else
            {
                if (!cc.Write(
                        idx,
                        w_idx,
                        &item,
                        table.data + idx * sizeof(Item),
                        sizeof(Item)))
                {
                    success1 = false;
                    break;
                }
                w_idx++;
            }
        }
        if (!success1 || !cc.TxEnd(NULL))
        {
            // common::sleep(1.0f * exp(-(float)tid) / (r_idx + w_idx + 1));
            // common::sleep(1);
            continue;
        }
        tx_success = true;
    }

    atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->tot_duration), clock64() - tx_start_time);
    atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->index_duration), index_duration);
    cc.Finalize();
}