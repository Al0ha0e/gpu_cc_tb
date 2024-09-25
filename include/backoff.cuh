#ifndef BACKOFF_H
#define BACKOFF_H

#include <runtime.cuh>

namespace cc
{
    __device__ __inline__ void constant_backoff(float us)
    {
        common::sleep(us);
    }

    __device__ __inline__ void down_backoff(float us, float)
    {
        common::sleep(us);
    }
} // namespace cc

#endif