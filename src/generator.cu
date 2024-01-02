#include <generator.cuh>

namespace common
{
    GeneratorBase *GetGenerator(GeneratorType type,
                                long long minval,
                                long long maxval,
                                unsigned int vsize,
                                int seed)
    {
        switch (type)
        {
        case RANDOM_INT_GENERATOR:
            return new RandomIntegerGenerator(minval, maxval, vsize, seed);
        case UNIQUE_INT_GENERATOR:
            return new UniqueIntegerGenerator(minval, maxval, vsize, seed);
        default:
            return nullptr;
        }
    }
}