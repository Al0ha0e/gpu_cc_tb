#ifndef GPUTx_H
#define GPUTx_H

#ifndef NVRTC_COMPILE

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <map>
#include <env.cuh>

#endif

#include <runtime.cuh>
#include <cc.cuh>
#include <transaction.cuh>

namespace cc
{
    struct GTX_AuxiliaryItem
    {
        int offset;
        int lock;

        __host__ __device__ GTX_AuxiliaryItem() {}

        __host__ __device__ GTX_AuxiliaryItem(int o, int l) : offset(o), lock(l) {}
    };

    struct GTX_AccessItem
    {
        int obj_idx;
        int tx_idx;
        char read;
        __host__ __device__ bool operator<(const GTX_AccessItem &ano) const
        {
            if (obj_idx == ano.obj_idx)
                return tx_idx < ano.tx_idx;
            return obj_idx < ano.obj_idx;
        }
    };

    struct RankItem
    {
        int rank;
        int tx_idx;

        __host__ __device__ bool operator<(const RankItem &ano) const
        {
            return rank < ano.rank;
        }
    };

#ifndef GPUTX_RUN
#define RANK_INFO 0
#endif

    class GPUTx_GPU
    {
    public:
        common::Metrics self_metrics;
        size_t self_tid;
        int self_rank;

#ifdef DYNAMIC_RW_COUNT
        int rcnt;
        int wcnt;
#endif

#ifdef TX_DEBUG
        common::Event *self_events;
#endif

        __device__ GPUTx_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = ((RankItem *)RANK_INFO)[tid].tx_idx;
            self_rank = ((RankItem *)RANK_INFO)[tid].rank;
            memset(&self_metrics, 0, sizeof(common::Metrics));
#ifdef DYNAMIC_RW_COUNT
            common::DynamicTransactionSet_GPU *txset_info = (common::DynamicTransactionSet_GPU *)txs_info;
            rcnt = txset_info->tx_rcnt[self_tid];
            wcnt = txset_info->tx_wcnt[self_tid];
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + txset_info->tx_opcnt_st[self_tid] + self_tid;
#endif
#else
#ifdef TX_DEBUG
            self_events = ((common::Event *)EVENTS_ST) + (RCNT + WCNT + 1) * self_tid;
#endif
#endif
        }

        __device__ bool TxStart(void *info) { return true; }

        __device__ bool TxEnd(void *info)
        {
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + WCNT, 0, 0, 0, self_tid, 2);
#endif
            return true;
        }

        __device__ bool Read(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            memcpy(dstdata, srcdata, size);
#ifdef TX_DEBUG
            common::AddEvent(self_events + tx_idx, obj_idx, 0, self_rank, self_tid, 0);
#endif
            return true;
        }

        __device__ bool ReadForUpdate(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            memcpy(dstdata, srcdata, size);
#ifdef TX_DEBUG
            common::AddEvent(self_events + tx_idx, obj_idx, 0, self_rank, self_tid, 0);
#endif
            return true;
        }

        __device__ bool Write(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            memcpy(dstdata, srcdata, size);
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + tx_idx, obj_idx, 0, self_rank, self_tid, 1);
#endif
            return true;
        }

        __device__ void Finalize() {}
    };

#ifndef NVRTC_COMPILE

    __global__ void gputx_pre2(
        int dense_obj_cnt,
        int *offset,
        GTX_AccessItem *access_table,
        RankItem *rank_item,
        int *tot_rank_cnt);

    __global__ void gputx_pre3(int tx_cnt, RankItem *rank_item, int *waiting_cnt_ptr);

    class GPUTx_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        common::DB_GPU *db_gpu;
        void *txdata;
        bool dynamic;

        GPUTx_CPU() {}

        GPUTx_CPU(common::DB_CPU *dbc, common::DB_GPU *dbg, common::TransactionSet_CPU *txinfo, void *txdata, size_t bsize)
            : db_cpu(dbc), db_gpu(dbg), info(txinfo), txdata(txdata),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), dbc->table_st[dbc->table_cnt])
        {
            cudaMalloc(&aux_table, sizeof(GTX_AuxiliaryItem) * (obj_cnt + 1));
            cudaMalloc(&access_table, sizeof(GTX_AccessItem) * info->tot_operation_cnt);
            cudaMalloc(&aux_ori_idx, sizeof(int) * info->tot_operation_cnt);
            cudaMalloc(&aux_dense_cnt, sizeof(int) * info->tot_operation_cnt);
            cudaMalloc(&acc_cnt, sizeof(int));
            cudaMalloc(&rank, sizeof(RankItem) * tx_cnt);
            cudaMalloc(&tot_rank_cnt, sizeof(int));
            cudaMalloc(&waiting_cnt_ptr, sizeof(int) * tx_cnt);

            cudaMemset(aux_table, 0, sizeof(GTX_AuxiliaryItem) * (obj_cnt + 1));
            cudaMemset(rank, 0, sizeof(RankItem) * tx_cnt);
            cudaMemset(waiting_cnt_ptr, 0, sizeof(int) * tx_cnt);
            cudaMemset(tot_rank_cnt, 0, sizeof(int));
            std::string include_path("./include");
            if (dynamic)
                pre1_code = genPre1DynamicCode(dynamic_cast<common::DynamicTransactionSet_CPU *>(info));
            else
                pre1_code = genPre1StaticCode(dynamic_cast<common::StaticTransactionSet_CPU *>(info));

            // std::cout << pre1_code << "\n";

            program = new common::RTCProgram(pre1_code, include_path, "gputx_pre1", false);
            program->AddNameExpression("pre1");
            std::vector<std::string> str_opts = {
                std::string("-I ") + include_path,
                std::string("-I /usr/include"),
                "-D NVRTC_COMPILE",
                "-D CLOCK_RATE=0",
                "-arch=compute_75"};
            db_cpu->GetCompileOptions(str_opts);
            std::vector<const char *> opts;
            for (auto &s : str_opts)
                opts.push_back(s.c_str());

            if (!program->Compile(opts))
                exit(1);
        }

        void Init0() override
        {
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            dim3 num_blocks(tx_cnt / 1024 + (tx_cnt % 1024 > 0), 1, 1);
            dim3 num_threads(tx_cnt > 1024 ? 1024 : tx_cnt, 1, 1);

            if (dynamic)
            {
                common::DynamicTransactionSet_GPU *txset_info = dynamic_cast<common::DynamicTransactionSet_CPU *>(info)->ToGPU();
                void **args = new void *[9];
                args[0] = &txdata;
                args[1] = &txset_info;
                args[2] = &db_gpu;
                args[3] = &aux_table;
                args[4] = &aux_ori_idx;
                args[5] = &aux_dense_cnt;
                args[6] = &access_table;
                args[7] = &acc_cnt;
                program->Call("pre1", num_blocks, num_threads, args, stream);
            }
            else
            {
                void **args = new void *[8];
                args[0] = &txdata;
                args[1] = &db_gpu;
                args[2] = &aux_table;
                args[3] = &aux_ori_idx;
                args[4] = &aux_dense_cnt;
                args[5] = &access_table;
                args[6] = &acc_cnt;
                program->Call("pre1", num_blocks, num_threads, args, stream);
            }
            cudaStreamSynchronize(stream);

            int acc_cnt_cpu;
            cudaMemcpy(&acc_cnt_cpu, acc_cnt, sizeof(int), cudaMemcpyDeviceToHost);

            thrust::sort_by_key(thrust::cuda::par.on(stream), aux_ori_idx, aux_ori_idx + acc_cnt_cpu, aux_dense_cnt);
            thrust::exclusive_scan(thrust::cuda::par.on(stream), aux_dense_cnt, aux_dense_cnt + acc_cnt_cpu, aux_dense_cnt);

            size_t opcnt;
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dyinfo = dynamic_cast<common::DynamicTransactionSet_CPU *>(info);
                opcnt = dyinfo->tx_opcnt_st[tx_cnt] - dyinfo->tx_opcnt_st[0];
            }
            else
            {
                common::StaticTransactionSet_CPU *stinfo = dynamic_cast<common::StaticTransactionSet_CPU *>(info);
                opcnt = stinfo->opcnt * tx_cnt;
            }
            thrust::sort(thrust::cuda::par.on(stream), access_table, access_table + opcnt);

            num_blocks = dim3(acc_cnt_cpu / 1024 + (acc_cnt_cpu % 1024 > 0), 1, 1);
            num_threads = dim3(acc_cnt_cpu > 1024 ? 1024 : acc_cnt_cpu, 1, 1);
            gputx_pre2<<<num_blocks, num_threads, 0, stream>>>(
                acc_cnt_cpu,
                aux_dense_cnt,
                access_table,
                rank,
                tot_rank_cnt);

            num_blocks = dim3(tx_cnt / 1024 + (tx_cnt % 1024 > 0), 1, 1);
            num_threads = dim3(tx_cnt > 1024 ? 1024 : tx_cnt, 1, 1);
            gputx_pre3<<<num_blocks, num_threads, 0, stream>>>(tx_cnt, rank, waiting_cnt_ptr);

            thrust::sort(thrust::cuda::par.on(stream), rank, rank + tx_cnt);

            int cpu_rank_cnt;
            int *cpu_waiting_cnt = new int[tx_cnt];
            cudaMemcpyAsync(&cpu_rank_cnt, tot_rank_cnt, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(cpu_waiting_cnt, waiting_cnt_ptr, sizeof(int) * tx_cnt, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            printf("-------------\n");
            for (int i = 0; i < cpu_rank_cnt; i++)
            {
                // printf("%d ", cpu_waiting_cnt[i]);
                batches.push_back(cpu_waiting_cnt[i]);
            }
            streams.resize(batches.size());
            printf("-------------\n");

            printf("OK1\n");
            cudaStreamDestroy(stream);
        }

        void Init(int batch_id, int batch_st) override
        {
            if (batch_id)
                cudaStreamSynchronize(streams[batch_id - 1]);
            cudaStreamCreate(streams.data() + batch_id);
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D CC_TYPE=cc::GPUTx_GPU");
            opts.push_back(std::string("-D RANK_INFO=") + std::to_string((unsigned long long)rank));
            opts.push_back("-D GPUTX_RUN");
        }

        void *ToGPU() override
        {
            return rank;
        }

        size_t GetMemSize() override
        {
            return 0;
        }

        void Explain(unsigned long long self_info, unsigned long long target_info) override
        {
            std::cout << "rank " << self_info << "\n";
        }

    private:
        GTX_AuxiliaryItem *aux_table;
        GTX_AccessItem *access_table;
        int *aux_ori_idx;
        int *aux_dense_cnt;
        int *acc_cnt;
        RankItem *rank;
        int *tot_rank_cnt;
        int *waiting_cnt_ptr;
        std::string pre1_code;
        common::RTCProgram *program;

        std::string genPre1StaticCode(common::StaticTransactionSet_CPU *txinfo)
        {
            std::stringstream ret;
            std::string tx_struct_name = ((common::StructType *)(txinfo->type))->name;
            ret << "#include<db.cuh>\n";
            ret << "#include<gputx.cuh>\n";
            ret << "#include<index.cuh>\n";
            ret << txinfo->type->GetName();
            // ret << "__device__ unsigned long long global_st = 0;\n";
            ret << "__global__ void pre1(" << tx_struct_name << " *txs, "
                << "common::DB_GPU *db, cc::GTX_AuxiliaryItem *aux_table, int *ori_idx, int *dense_cnt, cc::GTX_AccessItem *access_table, int *acc_cnt) {\n";
            ret << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
            ret << "if(tid >= " << txinfo->tx_cnt << ") return;\n";
            ret << "if(tid == 0) { *acc_cnt = 0;}\n";
            ret << "__threadfence();";
            ret << "int prev;\n";
            ret << tx_struct_name << " tx = txs[tid];";
            ret << txinfo->GetAccessorGPUCode();

            int opcnt = txinfo->opcnt;

            ret << "size_t st = tid * " << opcnt << ";\n"; // atomicAdd(&global_st, " << opcnt << "UL);\n"; // size_t st = atomicAdd(&global_st, opcnt);

            for (int i = 0; i < txinfo->rcnt; i++)
            {
                common::TxAccessor &acc = txinfo->rset[i];
                ret << "size_t auxoff_r" << i << " = " << acc.name << "_idx + "
                    << "TABLE_" << acc.table_idx << "_ST;\n";
                ret << "prev = atomicAdd(&(aux_table[auxoff_r" << i << "].offset), 1);\n"; // atomicAdd(&(aux_table + XXX_idx + TABLE_idx_ST)->offset, 1);
                ret << "if(prev == 0) aux_table[auxoff_r" << i << "].lock = atomicAdd(acc_cnt, 1);\n";
                ret << "access_table[st].obj_idx = auxoff_r" << i << ";\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
                ret << "access_table[st].read = true;\n";
                ret << "access_table[st++].tx_idx = tid;\n"; // access_table[st++].tx_idx =  tid;
            }

            for (int i = 0; i < txinfo->wcnt; i++)
            {
                common::TxAccessor &acc = txinfo->wset[i];
                ret << "size_t auxoff_w" << i << " = " << acc.name << "_idx + "
                    << "TABLE_" << acc.table_idx << "_ST;\n";
                ret << "prev = atomicAdd(&(aux_table[auxoff_w" << i << "].offset), 1);\n"; // atomicAdd(&(aux_table + XXX_idx + TABLE_idx_ST)->offset, 1);
                ret << "if(prev == 0) aux_table[auxoff_w" << i << "].lock = atomicAdd(acc_cnt, 1);\n";
                ret << "access_table[st].obj_idx = auxoff_w" << i << ";\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
                ret << "access_table[st].read = false;\n";
                ret << "access_table[st++].tx_idx = tid;\n"; // access_table[st++].tx_idx =  tid;
            }

            ret << "__threadfence();\n";

            ret << "cc::GTX_AuxiliaryItem *auxp;\n";
            for (int i = 0; i < txinfo->rcnt; i++)
            {
                ret << "auxp = aux_table + auxoff_r" << i << ";\n";
                ret << "ori_idx[auxp->lock] = auxoff_r" << i << ";\n";
                ret << "dense_cnt[auxp->lock] = auxp->offset;\n";
            }

            for (int i = 0; i < txinfo->wcnt; i++)
            {
                ret << "auxp = aux_table + auxoff_w" << i << ";\n";
                ret << "ori_idx[auxp->lock] = auxoff_w" << i << ";\n";
                ret << "dense_cnt[auxp->lock] = auxp->offset;\n";
            }

            ret << "}\n";

            return ret.str();
        }

        std::string genPre1DynamicCode(common::DynamicTransactionSet_CPU *txinfo)
        {
            std::stringstream ret;
            std::string tx_struct_name = ((common::StructType *)(txinfo->type))->name;
            int tx_type_cnt = txinfo->tx_type_cnt;

            ret << "#include<db.cuh>\n";
            ret << "#include<gputx.cuh>\n";
            ret << "#include<index.cuh>\n";
            ret << "#include<transaction.cuh>\n";
            ret << txinfo->type->GetName();
            // ret << "__device__ unsigned long long global_st = 0;\n";
            ret << "__global__ void pre1(" << tx_struct_name << " *txs, "
                << "common::DynamicTransactionSet_GPU *txset_info, common::DB_GPU *db, cc::GTX_AuxiliaryItem *aux_table, int *ori_idx, int *dense_cnt, cc::GTX_AccessItem*access_table, int *acc_cnt) {\n";
            ret << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
            ret << "if(tid >= " << txinfo->tx_cnt << ") return;\n";
            ret << "if(tid == 0) {*acc_cnt = 0;}\n";
            ret << "__threadfence();\n";
            ret << txinfo->GetGlobalAccessorGPUCode();
            ret << "int prev,st,en;\n";
            ret << "en = txset_info->tx_detailed_opcnt_st["
                << "tid * " << tx_type_cnt << "];\n";
            for (int i = 0; i < tx_type_cnt; i++)
            {
                common::TxAccessor &acc = txinfo->accessors[i];
                ret << "st = en;\n";
                ret << "en = txset_info->tx_detailed_opcnt_st["
                    << "tid * " << tx_type_cnt << " + 1 + " << i << "];\n";
                ret << "for(int i = st; i < en; i++) {\n";
                ret << tx_struct_name << " &tx = txs[i];\n";
                ret << txinfo->GetLocalAccessorGPUCode(i);
                ret << "size_t auxoff = " << acc.name << "_idx + "
                    << "TABLE_" << acc.table_idx << "_ST;\n";
                ret << "prev = atomicAdd(&(aux_table[auxoff].offset), 1);\n"; // atomicAdd(&(aux_table + XXX_idx + TABLE_idx_ST)->offset, 1);
                ret << "if(prev == 0) aux_table[auxoff].lock = atomicAdd(acc_cnt, 1);\n";
                ret << "access_table[i].obj_idx = auxoff;\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
                ret << "access_table[i].read = tx.read;\n";
                ret << "access_table[i].tx_idx = tid;\n"; // access_table[st++].tx_idx =  tid;
                ret << "}\n";
            }

            ret << "__threadfence();\n";
            ret << "cc::GTX_AuxiliaryItem *auxp;\n";

            ret << "st = txset_info->tx_opcnt_st[tid];\n";
            ret << "en = txset_info->tx_opcnt_st[tid + 1];\n";
            ret << "for(int i = st; i < en; i++) {\n";
            ret << "size_t auxoff = access_table[i].obj_idx;\n";
            ret << "auxp = aux_table + auxoff;\n";
            ret << "ori_idx[auxp->lock] = auxoff;\n";
            ret << "dense_cnt[auxp->lock] = auxp->offset;\n";
            ret << "}\n";

            ret << "}\n";

            return ret.str();
        }
    };

#endif
}

#endif