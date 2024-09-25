#include <gputx.cuh>

namespace cc
{
    __global__ __launch_bounds__(1024) void gputx_pre2(
        int dense_obj_cnt,
        int *offset,
        GTX_AccessItem *access_table,
        RankItem *rank_item,
        int *tot_rank_cnt)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid >= dense_obj_cnt)
            return;

        int st = offset[tid];
        int en = offset[tid + 1];
        int cur_rank = 0;
        for (int i = st; i < en; i++)
        {
            GTX_AccessItem &item = access_table[i];
            int sb = atomicMax(&(rank_item[item.tx_idx].rank), cur_rank);
            if (i + 1 < en && (!item.read || !access_table[i + 1].read))
                cur_rank++;
        }
        atomicMax(tot_rank_cnt, cur_rank + 1);
    }

    __global__ __launch_bounds__(1024) void gputx_pre3(int tx_cnt, RankItem *rank_item, int *waiting_cnt_ptr)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= tx_cnt)
            return;
        RankItem &item = rank_item[tid];
        item.tx_idx = tid;
        atomicAdd(waiting_cnt_ptr + item.rank, 1);
    }
}