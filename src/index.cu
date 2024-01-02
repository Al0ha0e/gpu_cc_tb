#include <index.cuh>

#define TO_STRING_LITERAL(x) #x

#define GET_INDEX_NAME_BY_TYPE(index, type) TO_STRING_LITERAL(index##<##type##>)

namespace common
{
    static std::string IndexNames[] = {"common::Naive_GPU", "common::SortedArray_GPU"};

    std::string GetIndexName(IndexType index_type, TypeCode key_type)
    {
        std::string ret = IndexNames[index_type];
        switch (key_type)
        {
        case INT32:
            return ret + "<int>";
        case INT64:
            return ret + "<long long>";
        default:
            return ret;
        }
    }
}