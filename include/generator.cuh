#ifndef GENERATOR_H
#define GENERATOR_H

#ifndef NVRTC_COMPILE

#include <random>
#include <stdio.h>
#include <type.cuh>

#endif

namespace common
{
#ifndef NVRTC_COMPILE

    enum GeneratorType
    {
        CONST_INT_GENERATOR,
        SEQUENCE_INT_GENERATOR,
        RANDOM_INT_GENERATOR,
        UNIQUE_INT_GENERATOR,
        CONST_FLOAT_GENERATOR,
        RANDOM_FLOAT_GENERATOR,
    };

    class GeneratorBase
    {
    public:
        virtual char *Generate(size_t cnt) = 0;
        virtual void GenerateTo(TypeWithSize &type, size_t cnt, size_t size, char *dst)
        {
            char *data = Generate(cnt);
            char *srcnow = data;
            char *dstnow = dst + type.offset;
            for (int i = 0; i < cnt; i++, srcnow += type.size, dstnow += size)
                memcpy(dstnow, srcnow, type.size);
            delete[] data;
        }
    };

    GeneratorBase *GetGenerator(GeneratorType type,
                                long long minval,
                                long long maxval,
                                unsigned int vsize,
                                int seed);

    class DoNothingGenerator : public GeneratorBase
    {
    public:
        char *Generate(size_t cnt) override { return nullptr; }
        void GenerateTo(TypeWithSize &type, size_t cnt, size_t size, char *dst) override {}
    };

    class IntegerGeneratorBase : public GeneratorBase
    {
    public:
        long long min_val;
        long long max_val;
        unsigned int val_size;

        // virtual char *Generate(size_t cnt) = 0;
        IntegerGeneratorBase() {}
        IntegerGeneratorBase(unsigned int vsize, int s) : min_val(0),
                                                          max_val(0),
                                                          val_size(vsize),
                                                          seed(s) {}
        IntegerGeneratorBase(
            long long minval,
            long long maxval,
            unsigned int vsize,
            int s) : min_val(minval),
                     max_val(maxval),
                     val_size(vsize),
                     seed(s)
        {
            // TODO Check range
            engine = std::ranlux48(s);
            distrib8 = std::uniform_int_distribution<long long>(minval, maxval);
            distrib4 = std::uniform_int_distribution<int>(minval, maxval);
            distrib2 = std::uniform_int_distribution<short>(minval, maxval);
            distrib1 = std::uniform_int_distribution<short>(minval, maxval);
        }

    protected:
        int seed;
        std::ranlux48 engine;
        std::uniform_int_distribution<long long> distrib8;
        std::uniform_int_distribution<int> distrib4;
        std::uniform_int_distribution<short> distrib2;
        std::uniform_int_distribution<short> distrib1;
    };

    class ConstIntegerGenerator : public IntegerGeneratorBase
    {
    public:
        ConstIntegerGenerator() {}
        ConstIntegerGenerator(long long val, unsigned int vsize)
            : IntegerGeneratorBase(val, val, vsize, 0) {}

        char *Generate(size_t cnt) override
        {
            char *ret = new char[cnt * val_size];
            switch (val_size)
            {
            case 8:
                gen((long long *)ret, cnt);
                break;
            case 4:
                gen((int *)ret, cnt);
                break;
            case 2:
                gen((short *)ret, cnt);
                break;
            default:
                gen(ret, cnt);
                break;
            }

            return ret;
        }

    private:
        template <typename T>
        void gen(T *now, size_t cnt)
        {
            for (int i = 0; i < cnt; i++)
                now[i] = min_val;
        }
    };

    class RandomIntegerGenerator : public IntegerGeneratorBase
    {
    public:
        RandomIntegerGenerator() {}

        RandomIntegerGenerator(
            long long minval,
            long long maxval,
            unsigned int vsize,
            int seed) : IntegerGeneratorBase(minval, maxval, vsize, seed)
        {
        }

        char *Generate(size_t cnt) override
        {
            char *ret = new char[cnt * val_size];
            switch (val_size)
            {
            case 8:
            {
                long long *now = (long long *)ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = distrib8(engine);
            }
            break;
            case 4:
            {
                int *now = (int *)ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = distrib4(engine);
            }
            break;
            case 2:
            {
                short *now = (short *)ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = distrib2(engine);
            }
            break;
            default:
            {
                char *now = ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = distrib1(engine);
            }
            break;
            }

            return ret;
        }
    };

    class SequenceIntegerGenerator : public IntegerGeneratorBase
    {
    public:
        struct SequenceInfo
        {
            long long st;
            long long step;
            unsigned int period;
        };

        SequenceIntegerGenerator() {}
        SequenceIntegerGenerator(
            long long st,
            long long step,
            unsigned int period,
            unsigned int vsize,
            int seed)
            : SequenceIntegerGenerator(vsize, seed, std::vector<SequenceInfo>{SequenceInfo{st, step, period}}) {}

        SequenceIntegerGenerator(unsigned int vsize, int seed, std::vector<SequenceInfo> &&info)
            : seq_info(info), IntegerGeneratorBase(vsize, seed) {}

        char *Generate(size_t cnt) override
        {
            char *ret = new char[cnt * val_size];
            switch (val_size)
            {
            case 8:
                gen((long long *)ret, cnt);
                break;
            case 4:
                gen((int *)ret, cnt);
                break;
            case 2:
                gen((short *)ret, cnt);
                break;
            default:
                gen((char *)ret, cnt);
                break;
            }

            return ret;
        }

    private:
        std::vector<SequenceInfo> seq_info;

        template <typename T>
        void gen(T *now, size_t cnt)
        {
            for (int i = 0; i < cnt; i++)
            {
                T val = 0;
                size_t curcnt = i;
                for (int j = 0; j < seq_info.size(); j++)
                {
                    SequenceInfo &info = seq_info[j];
                    int idx = curcnt % info.period;
                    curcnt /= info.period;
                    val += idx * info.step + info.st;
                }
                now[i] = val;
            }
        }
    };

    class UniqueIntegerGenerator : public IntegerGeneratorBase
    {
    public:
        UniqueIntegerGenerator() {}

        UniqueIntegerGenerator(
            long long minval,
            long long maxval,
            unsigned int vsize,
            int seed) : IntegerGeneratorBase(minval, maxval, vsize, seed) {}

        char *Generate(size_t cnt) override
        {
            char *ret = new char[cnt * val_size];

            switch (val_size)
            {
            case 8:
            {
                long long *tmp = new long long[max_val - min_val + 1];
                for (int i = 0; i <= max_val - min_val; i++)
                    tmp[i] = min_val + i;

                long long *now = (long long *)ret;
                for (int i = 0; i < cnt; i++)
                {
                    std::uniform_int_distribution<size_t> u(0, max_val - min_val - i);
                    size_t pos = u(engine);
                    long long t = tmp[pos];
                    now[i] = t;
                    tmp[pos] = tmp[max_val - min_val - i];
                    tmp[max_val - min_val - i] = t;
                }
                delete[] tmp;
            }
            break;
            case 4:
            {
                int *tmp = new int[max_val - min_val + 1];
                for (int i = 0; i <= max_val - min_val; i++)
                    tmp[i] = min_val + i;

                int *now = (int *)ret;
                for (int i = 0; i < cnt; i++)
                {
                    std::uniform_int_distribution<size_t> u(0, max_val - min_val - i);
                    size_t pos = u(engine);
                    int t = tmp[pos];
                    now[i] = t;
                    tmp[pos] = tmp[max_val - min_val - i];
                    tmp[max_val - min_val - i] = t;
                }
                delete[] tmp;
            }
            break;
            case 2:
            {
                short *tmp = new short[max_val - min_val + 1];
                for (int i = 0; i <= max_val - min_val; i++)
                    tmp[i] = min_val + i;

                short *now = (short *)ret;
                for (int i = 0; i < cnt; i++)
                {
                    std::uniform_int_distribution<size_t> u(0, max_val - min_val - i);
                    size_t pos = u(engine);
                    short t = tmp[pos];
                    now[i] = t;
                    tmp[pos] = tmp[max_val - min_val - i];
                    tmp[max_val - min_val - i] = t;
                }
                delete[] tmp;
            }
            break;
            default:
            {
                char *tmp = new char[max_val - min_val + 1];
                for (int i = 0; i <= max_val - min_val; i++)
                    tmp[i] = min_val + i;

                char *now = ret;
                for (int i = 0; i < cnt; i++)
                {
                    std::uniform_int_distribution<size_t> u(0, max_val - min_val - i);
                    size_t pos = u(engine);
                    char t = tmp[pos];
                    now[i] = t;
                    tmp[pos] = tmp[max_val - min_val - i];
                    tmp[max_val - min_val - i] = t;
                }
                delete[] tmp;
            }
            break;
            }

            return ret;
        }
    };

    class FloatGeneratorBase : public GeneratorBase
    {
    public:
        double min_val;
        double max_val;
        unsigned int val_size;

        FloatGeneratorBase() {}

        FloatGeneratorBase(
            double minval,
            double maxval,
            unsigned int vsize,
            int s) : min_val(minval),
                     max_val(maxval),
                     val_size(vsize),
                     seed(s)
        {
            // TODO Check range
            engine = std::ranlux48(s);
            distrib8 = std::uniform_real_distribution<double>(minval, maxval);
            distrib4 = std::uniform_real_distribution<float>(minval, maxval);
        }

    protected:
        int seed;
        std::ranlux48 engine;
        std::uniform_real_distribution<double> distrib8;
        std::uniform_real_distribution<float> distrib4;
    };

    class ConstFloatGenerator : public FloatGeneratorBase
    {
    public:
        ConstFloatGenerator() {}

        ConstFloatGenerator(
            double val,
            unsigned int vsize) : FloatGeneratorBase(val, val, vsize, 0)
        {
        }

        char *Generate(size_t cnt) override
        {
            char *ret = new char[cnt * val_size];
            switch (val_size)
            {
            case 8:
            {
                double *now = (double *)ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = min_val;
            }
            break;
            default:
            {
                float *now = (float *)ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = max_val;
            }
            break;
            }

            return ret;
        }
    };

    class RandomFloatGenerator : public FloatGeneratorBase
    {
    public:
        RandomFloatGenerator() {}

        RandomFloatGenerator(
            double minval,
            double maxval,
            unsigned int vsize,
            int seed) : FloatGeneratorBase(minval, maxval, vsize, seed)
        {
        }

        char *Generate(size_t cnt) override
        {
            char *ret = new char[cnt * val_size];
            switch (val_size)
            {
            case 8:
            {
                double *now = (double *)ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = distrib8(engine);
            }
            break;
            default:
            {
                float *now = (float *)ret;
                for (int i = 0; i < cnt; i++)
                    now[i] = distrib4(engine);
            }
            break;
            }

            return ret;
        }
    };
#endif
}

#endif