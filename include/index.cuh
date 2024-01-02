#ifndef INDEX_H
#define INDEX_H

#include <type.cuh>

#ifndef NVRTC_COMPILE

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <string>

#endif

namespace common
{
    template <typename KT, typename VT>
    struct IndexItem
    {
        KT key;
        VT value;

        __host__ __device__ IndexItem() {}

        __host__ __device__ IndexItem(KT k, VT v) : key(k), value(v) {}

        __host__ __device__ bool operator<(const IndexItem &ano) const
        {
            return key < ano.key;
        }

        __host__ __device__ bool operator==(const IndexItem &ano) const
        {
            return key == ano.key;
        }
    };

    // template <typename KT, typename VT>
    // struct IndexIterator
    // {
    //     using item_t = IndexItem<KT, VT>;

    //     item_t *index;

    //     __device__ IndexIterator() {}
    //     __device__ IndexIterator(item_t *idx) : index(idx) {}

    //     __device__ item_t operator*() const
    //     {
    //         return *index;
    //     }

    //     __device__ item_t *operator->() const
    //     {
    //         return index;
    //     }
    // };

    enum IndexType
    {
        NAIVE,
        SORTED_ARRAY
    };

    template <typename KT>
    class SortedArray_GPU
    {
    public:
        IndexItem<KT, size_t> *arr;
        unsigned int length;

        __host__ __device__ SortedArray_GPU() {}

        __device__ size_t Find(KT key)
        {
            unsigned int st = 0, en = length;
            while (st < en)
            {
                unsigned int mid = (st + en) >> 1;
                KT k = arr[mid].key;
                if (k == key)
                    return arr[mid].value;
                if (k > key)
                    en = mid;
                else
                    st = mid + 1;
            }
            if (st < length && arr[st].key == key)
                return arr[st].value;
            return length;
        }
    };

#ifndef NVRTC_COMPILE

    template <typename KT>
    class SortedArray_CPU
    {
    public:
        using gpu_t = SortedArray_GPU<KT>;
        using kv_t = IndexItem<KT, size_t>;

        kv_t *arr;
        unsigned int length;

        SortedArray_CPU(kv_t *a, int l) : length(l)
        {
            std::sort(a, a + l);
            cudaMalloc(&arr, sizeof(kv_t) * length);
            cudaMemcpy(arr, a, sizeof(kv_t) * length, cudaMemcpyHostToDevice);
        }

        gpu_t *ToGPU()
        {
            gpu_t *ret = nullptr;
            gpu_t *tmp = new gpu_t;

            tmp->arr = arr;
            tmp->length = length;

            cudaMalloc(&ret, sizeof(gpu_t));
            cudaMemcpy(ret, tmp, sizeof(gpu_t), cudaMemcpyHostToDevice);

            delete tmp;
            return ret;
        }
    };

#define GET_INDEX_BY_TYPE(index)                                 \
    template <typename KT>                                       \
    index##_GPU<KT> *Get##index(IndexItem<KT, size_t> *a, int l) \
    {                                                            \
        index##_CPU<KT> idx(a, l);                               \
        return idx.ToGPU();                                      \
    }

    GET_INDEX_BY_TYPE(SortedArray)
    // template <typename KT>
    // SortedArray_GPU<KT> *GetSortedArray(kv_t *a, int l)
    // {
    //     SortedArray_CPU<KT> sr(a, l);
    //     return sr.ToGPU();
    // }

    std::string GetIndexName(IndexType index_type, TypeCode key_type);

    template <typename KT>
    void *GetGPUIndex(IndexItem<KT, size_t> *a, int l, IndexType type)
    {
        switch (type)
        {
        case NAIVE:
            return nullptr;
        case SORTED_ARRAY:
            return GetSortedArray<KT>(a, l);
        default:
            return nullptr;
        }
    }

    template <typename T>
    class IndexPlaceholder
    {
    public:
        __device__ size_t Find(T key)
        {
            return 0;
        }
    };

#define INDEX_0_TYPE common::IndexPlaceholder
#define INDEX_1_TYPE common::IndexPlaceholder
#define INDEX_2_TYPE common::IndexPlaceholder
#define INDEX_3_TYPE common::IndexPlaceholder
#define INDEX_4_TYPE common::IndexPlaceholder

#endif

};

#endif