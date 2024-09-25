#ifndef RUNTIME_H
#define RUNTIME_H

#ifndef NVRTC_COMPILE
#include <string>
#include <fstream>
#include <cc.cuh>
#define CLOCK_RATE 0

#endif

#include <transaction.cuh>

// #define NVRTC_COMPILE

namespace common
{

#ifndef TX_DEBUG

#define EVENTS_ST 0
#define GLOBAL_EVENT_ID 0

#endif

#define MAX_EVENT_CNT 64

    struct __align__(8) Metrics
    {
        unsigned long long abort;
        unsigned long long ts_duration;      //
        unsigned long long wait_duration;    //
        unsigned long long abort_duration;   //
        unsigned long long manager_duration; //
        unsigned long long tot_duration;     //
        unsigned long long index_duration;   //
    };

    struct __align__(8) Event
    {
        unsigned long long id;
        unsigned long long oid;
        unsigned long long target_info;
        unsigned long long self_info;
        int tid;
        int type;

        __host__ __device__ Event() {}

        __host__ __device__ Event(unsigned long long id,
                                  unsigned long long oid,
                                  unsigned long long tinfo,
                                  unsigned long long sinfo,
                                  int tid,
                                  int type)
            : id(id), oid(oid), target_info(tinfo), self_info(sinfo), tid(tid), type(type) {}

        __host__ __device__ bool operator<(const Event &ano) const
        {
            return id < ano.id;
        }
    };

    __device__ __inline__ void AddEvent(
        Event *event,
        unsigned long long oid,
        unsigned long long tinfo,
        unsigned long long sinfo,
        int tid,
        int type)
    {
        unsigned long long id = atomicAdd((unsigned long long *)GLOBAL_EVENT_ID, 1);
        volatile Event *vevent = (volatile Event *)event;
        vevent->id = id;
        vevent->oid = oid;
        vevent->target_info = tinfo;
        vevent->self_info = sinfo;
        vevent->tid = tid;
        vevent->type = type;
        // *((volatile Event *)event) = Event(id, oid, tinfo, sinfo, tid, type);
    }

    __inline__ __device__ void sleep(float us)
    {
        clock_t t0 = clock64();
        clock_t t1 = t0;
        while (t1 - t0 < (unsigned long long)(us * 0.001 * CLOCK_RATE))
            t1 = clock64();
    }

    __inline__ void __device__ latch_lock(Metrics &metrics, int *latch)
    {
        unsigned long long wait_st_time = clock64();
        while (atomicCAS(latch, 0, 1))
            ;
        metrics.wait_duration += clock64() - wait_st_time;
    }

    __inline__ bool __device__ latch_lock_nw(int *latch)
    {
        return !atomicCAS(latch, 0, 1);
    }

    __inline__ void __device__ latch_unlock(int *latch)
    {
        __threadfence();
        *latch = 0;
        __threadfence();
    }

#ifndef NVRTC_COMPILE
    // ExecInfo *InitRuntime();

    struct ExecInfo
    {
        common::ConcurrencyControlCPUBase *cc;
        clock_t clock_rate;
        // ThreadEvents *events;
        Metrics *gpu_metrics;
        float precompute_time;
        float processing_time;
        float tot_time;
        size_t txcnt;
        int valid_txn_bitoffset;
        int warp_cnt;
        int batch_size;
        std::string bench_name;
        std::string cc_type;
        bool debug;
        size_t event_cnt;

        ExecInfo() = default;

        ExecInfo(std::string bench_name, ConcurrencyControlCPUBase *cc, TransactionSet_CPU *txinfo, size_t txcnt, char *cc_type,
                 int valid_txn_bitoffset, int warp_cnt, int batch_size, bool debug)
            : bench_name(bench_name), cc(cc), txcnt(txcnt), cc_type(cc_type),
              valid_txn_bitoffset(valid_txn_bitoffset), warp_cnt(warp_cnt), batch_size(batch_size),
              precompute_time(0), processing_time(0), tot_time(0), debug(debug)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            clock_rate = prop.clockRate;

            cudaMalloc(&gpu_metrics, sizeof(Metrics));
            cudaMemset(gpu_metrics, 0, sizeof(Metrics));

            if (debug)
            {
                cudaMalloc(&global_event_id, sizeof(unsigned long long));
                cudaMemset(global_event_id, 0, sizeof(unsigned long long));
                event_cnt = txinfo->GetTotOpCnt() + txcnt;
                cudaMalloc(&gpu_events, event_cnt * sizeof(Event));
                cudaMemset(gpu_events, 0, event_cnt * sizeof(Event));
            }
        }

        void GetCompileOptions(std::vector<std::string> &opts)
        {
            opts.push_back("-D GLOBAL_METRICS=" + std::to_string((unsigned long long)gpu_metrics));
            opts.push_back("-D CLOCK_RATE=" + std::to_string((unsigned long long)clock_rate));
            if (debug)
            {
                opts.push_back("-D TX_DEBUG");
                opts.push_back("-D EVENTS_ST=" + std::to_string((unsigned long long)gpu_events));
                opts.push_back("-D GLOBAL_EVENT_ID=" + std::to_string((unsigned long long)global_event_id));
            }
        }

        void Analyse()
        {
            cudaMemcpy(&host_metrics, gpu_metrics, sizeof(Metrics), cudaMemcpyDeviceToHost);
            float inv = 1.0f / (txcnt * clock_rate * 1.0f);
            float ts_duration = host_metrics.ts_duration * inv;           //
            float wait_duration = host_metrics.wait_duration * inv;       //
            float abort_duration = host_metrics.abort_duration * inv;     //
            float manager_duration = host_metrics.manager_duration * inv; //
            float tot_duration = host_metrics.tot_duration * inv;         //
            float index_duration = host_metrics.index_duration * inv;     //

            std::fstream of;
            of.open(bench_name + ".txt", std::ios::out | std::ios::app);
            of << cc_type << ","
               << valid_txn_bitoffset << ","
               << warp_cnt << ","
               << batch_size << ","
               << 1000.0f / tot_time * txcnt << ","
               << host_metrics.abort << ","
               << ts_duration << ","
               << wait_duration << ","
               << abort_duration << ","
               << manager_duration << ","
               << tot_duration << ","
               << index_duration << ","
               << precompute_time / 1000.f << ","
               << processing_time / 1000.f << ","
               << tot_time / 1000.f << "\n";
            of.close();
            std::cout << cc_type << ","
                      << "valid_txn_bitoffset " << valid_txn_bitoffset << ",\n"
                      << "warp_cnt " << warp_cnt << ",\n"
                      << "batch_size " << batch_size << ",\n"
                      << "through " << 1000.0f / tot_time * txcnt << ",\n"
                      << "abort " << host_metrics.abort << " diff " << ((1LL << 31) - 1) - ((long long)host_metrics.abort + (long long)batch_size)
                      << " avr " << host_metrics.abort * 1.0 / batch_size << ",\n"
                      << "ts_d " << ts_duration << ",\n"
                      << "wait_d " << wait_duration << ",\n"
                      << "abort_d " << abort_duration << ",\n"
                      << "mng_d " << manager_duration << ",\n"
                      << "tot_d " << tot_duration << ",\n"
                      << "idx_d " << index_duration << ",\n"
                      << "pre " << precompute_time / 1000.f << ",\n"
                      << "proc " << processing_time / 1000.f << ",\n"
                      << "tot " << tot_time / 1000.f << "\n";
        }

        void Check();

    private:
        unsigned long long *global_event_id;
        Event *gpu_events;
        Event *host_events;
        Metrics host_metrics;

        void scan_events();
        bool check(int now);
        void check0();
        void backtrack(int now);
    };

#endif
};

#endif