#include <gacco.cuh>

namespace cc
{
    __global__ void gacco_pre2(int dense_obj_cnt, cc::AuxiliaryItem *aux_table, int *ori_idx, int *offset, cc::AccessItem *access_table)
    {
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= dense_obj_cnt)
            return;

        cc::AuxiliaryItem &aux = aux_table[ori_idx[tid]];
        int off = offset[tid];
        aux.offset = off;
        aux.lock = access_table[off].tx_idx;
    }
}