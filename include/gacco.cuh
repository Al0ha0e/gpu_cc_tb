#ifndef GACCO_H
#define GACCO_H

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
    struct AuxiliaryItem
    {
        int offset;
        int lock;

        __host__ __device__ AuxiliaryItem() {}

        __host__ __device__ AuxiliaryItem(int o, int l) : offset(o), lock(l)
        {
        }
    };

    struct AccessItem
    {
        int obj_idx;
        int tx_idx;
        __host__ __device__ bool operator<(const AccessItem &ano) const
        {
            if (obj_idx == ano.obj_idx)
                return tx_idx < ano.tx_idx;
            return obj_idx < ano.obj_idx;
        }
    };

#ifndef GACCO_RUN
#define AUX_TABLE 0
#define ACCESS_TABLE 0
#endif

    class Gacco_GPU
    {
    public:
        common::Metrics self_metrics;
        size_t self_tid;

#ifdef DYNAMIC_RW_COUNT
        int rcnt;
        int wcnt;
#endif

#ifdef TX_DEBUG
        common::Event *self_events;
#endif

        __device__ Gacco_GPU(void *txs_info, void *info, size_t tid)
        {
            self_tid = tid;
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
            long long lock_st_time = clock64();
            lock(obj_idx);
            self_metrics.wait_duration += clock64() - lock_st_time;
            memcpy(dstdata, srcdata, size);
#ifdef TX_DEBUG
            common::AddEvent(self_events + tx_idx, obj_idx, 0, 0, self_tid, 0);
#endif
            unlock(obj_idx);
            self_metrics.manager_duration += clock64() - lock_st_time;
            return true;
        }

        __device__ bool ReadForUpdate(
            size_t obj_idx,
            int tx_idx,
            void *srcdata,
            void *dstdata,
            size_t size)
        {
            long long lock_st_time = clock64();
            lock(obj_idx);
            self_metrics.wait_duration += clock64() - lock_st_time;
            memcpy(dstdata, srcdata, size);
            unlock(obj_idx);
            self_metrics.manager_duration += clock64() - lock_st_time;
#ifdef TX_DEBUG
            common::AddEvent(self_events + tx_idx, obj_idx, 0, 0, self_tid, 0);
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
            long long lock_st_time = clock64();
            lock(obj_idx);
            self_metrics.wait_duration += clock64() - lock_st_time;
            memcpy(dstdata, srcdata, size);
#ifdef TX_DEBUG
            common::AddEvent(self_events + RCNT + tx_idx, obj_idx, 0, 0, self_tid, 1);
#endif
            unlock(obj_idx);
            self_metrics.manager_duration += clock64() - lock_st_time;
            return true;
        }

        __device__ void Finalize()
        {
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->wait_duration), self_metrics.wait_duration);
            atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->manager_duration), self_metrics.manager_duration);
        }

    private:
        __device__ void lock(size_t obj_idx)
        {
            AuxiliaryItem *cur_aux = ((AuxiliaryItem *)AUX_TABLE) + obj_idx;
            bool ok = false;
            while (!ok)
            {
                int curlock = *((volatile int *)(&cur_aux->lock));
                if (curlock == self_tid)
                    ok = true;
            }
        }

        void __device__ unlock(size_t obj_idx)
        {
            AuxiliaryItem &cur_aux = ((AuxiliaryItem *)AUX_TABLE)[obj_idx];
            cur_aux.offset += 1;

            cur_aux.lock = ((AccessItem *)ACCESS_TABLE)[cur_aux.offset].tx_idx;
            __threadfence();
        }
    };

#ifndef NVRTC_COMPILE

    __global__ void gacco_pre2(int dense_obj_cnt, cc::AuxiliaryItem *aux_table, int *ori_idx, int *offset, cc::AccessItem *access_table);

    class Gacco_CPU : public common::ConcurrencyControlCPUBase
    {
    public:
        AuxiliaryItem *aux_table;
        AccessItem *access_table;
        common::TransactionSet_CPU *info;
        common::DB_CPU *db_cpu;
        common::DB_GPU *db_gpu;
        void *txdata;
        bool dynamic;

        Gacco_CPU() {}

        Gacco_CPU(common::DB_CPU *dbc, common::DB_GPU *dbg, common::TransactionSet_CPU *txinfo, void *txdata, size_t bsize)
            : db_cpu(dbc), db_gpu(dbg), info(txinfo), txdata(txdata),
              dynamic(typeid(*info) == typeid(common::DynamicTransactionSet_CPU)),
              ConcurrencyControlCPUBase(bsize, txinfo->GetTxCnt(), dbc->table_st[dbc->table_cnt])
        {
            cudaMalloc(&aux_table, sizeof(AuxiliaryItem) * (obj_cnt + 1));
            cudaMalloc(&access_table, sizeof(AccessItem) * (info->tot_operation_cnt + 1));
            cudaMalloc(&aux_ori_idx, sizeof(int) * info->tot_operation_cnt);
            cudaMalloc(&aux_dense_cnt, sizeof(int) * info->tot_operation_cnt);
            cudaMalloc(&acc_cnt, sizeof(int));
            std::string include_path("./include");
            if (dynamic)
            {
                pre1_code = genPre1DynamicCode(dynamic_cast<common::DynamicTransactionSet_CPU *>(info));
                pre1_5_code = genPre1_5DynamicCode(dynamic_cast<common::DynamicTransactionSet_CPU *>(info));
            }
            else
            {
                pre1_code = genPre1StaticCode(dynamic_cast<common::StaticTransactionSet_CPU *>(info));
                pre1_5_code = genPre1_5StaticCode(dynamic_cast<common::StaticTransactionSet_CPU *>(info));
            }

            // std::cout << pre1_code << "\n";

            program1 = new common::RTCProgram(pre1_code, include_path, "gacco_pre1", false);
            program1->AddNameExpression("pre1");

            program1_5 = new common::RTCProgram(pre1_5_code, include_path, "gacco_pre1_5", false);
            program1_5->AddNameExpression("pre1_5");

            std::vector<std::string> str_opts = {
                std::string("-I ") + include_path,
                "-I /usr/local/cuda-12.0/include",
                "-D NVRTC_COMPILE",
                "-D GLOBAL_METRICS=0",
                "-D CLOCK_RATE=0",
                "-arch=compute_75"};

            db_cpu->GetCompileOptions(str_opts);
            std::vector<const char *> opts;
            for (auto &s : str_opts)
                opts.push_back(s.c_str());

            if (!program1->Compile(opts))
                exit(1);
            if (!program1_5->Compile(opts))
                exit(1);
        }

        void Init(int batch_id, int batch_st) override
        {
            cudaStreamCreate(streams.data() + batch_id);
            cudaStream_t stream = streams[batch_id];

            cudaMemset(aux_table, 0, sizeof(AuxiliaryItem) * (obj_cnt + 1));
            size_t batched_txnum = min(batch_size, tx_cnt - batch_id * batch_size);
            dim3 num_blocks(batched_txnum / 1024 + (batched_txnum % 1024 > 0), 1, 1);
            dim3 num_threads(batched_txnum > 1024 ? 1024 : batched_txnum, 1, 1);
            cudaMemset(acc_cnt, 0, sizeof(int));
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dyinfo = dynamic_cast<common::DynamicTransactionSet_CPU *>(info);

                common::DynamicTransactionSet_GPU *txset_info = dyinfo->ToGPU();
                void **args1 = new void *[9];
                args1[0] = &txdata;
                args1[1] = &batch_id;
                args1[2] = &txset_info;
                args1[3] = &db_gpu;
                args1[4] = &aux_table;
                args1[5] = &aux_ori_idx;
                args1[6] = &aux_dense_cnt;
                args1[7] = &access_table;
                args1[8] = &acc_cnt;
                program1->Call("pre1", num_blocks, num_threads, args1, stream);

                void **args1_5 = new void *[6];
                args1_5[0] = &batch_id;
                args1_5[1] = &txset_info;
                args1_5[2] = &aux_table;
                args1_5[3] = &aux_ori_idx;
                args1_5[4] = &aux_dense_cnt;
                args1_5[5] = &access_table;
                program1_5->Call("pre1_5", num_blocks, num_threads, args1_5, stream);
            }
            else
            {
                void **args1 = new void *[8];
                args1[0] = &txdata;
                args1[1] = &batch_id;
                args1[2] = &db_gpu;
                args1[3] = &aux_table;
                args1[4] = &aux_ori_idx;
                args1[5] = &aux_dense_cnt;
                args1[6] = &access_table;
                args1[7] = &acc_cnt;
                program1->Call("pre1", num_blocks, num_threads, args1, stream);

                void **args1_5 = new void *[5];
                args1_5[0] = &batch_id;
                args1_5[1] = &aux_table;
                args1_5[2] = &aux_ori_idx;
                args1_5[3] = &aux_dense_cnt;
                args1_5[4] = &access_table;
                program1_5->Call("pre1_5", num_blocks, num_threads, args1_5, stream);
            }
            // cudaStreamSynchronize(stream);
            // printf("1\n");

            int acc_cnt_cpu;
            cudaMemcpy(&acc_cnt_cpu, acc_cnt, sizeof(int), cudaMemcpyDeviceToHost);

            thrust::sort_by_key(thrust::cuda::par.on(stream), aux_ori_idx, aux_ori_idx + acc_cnt_cpu, aux_dense_cnt);

            // int *cpu_ori_idx = new int[info->tot_operation_cnt];
            // int *cpu_dense_cnt = new int[info->tot_operation_cnt];
            // cudaMemcpy(cpu_ori_idx, aux_ori_idx, acc_cnt_cpu * sizeof(int), cudaMemcpyDeviceToHost);
            // cudaMemcpy(cpu_dense_cnt, aux_dense_cnt, acc_cnt_cpu * sizeof(int), cudaMemcpyDeviceToHost);
            // printf("------------ %d ----------\n", acc_cnt_cpu);
            // for (int i = 0; i < acc_cnt_cpu; i++)
            //     printf("%d %d\n", cpu_ori_idx[i], cpu_dense_cnt[i]);
            // printf("---------------------------\n");

            thrust::exclusive_scan(thrust::cuda::par.on(stream), aux_dense_cnt, aux_dense_cnt + acc_cnt_cpu, aux_dense_cnt);
            // printf("2\n");
            size_t opcnt;
            if (dynamic)
            {
                common::DynamicTransactionSet_CPU *dyinfo = dynamic_cast<common::DynamicTransactionSet_CPU *>(info);
                opcnt = dyinfo->tx_opcnt_st[batch_id * batch_size + batched_txnum] - dyinfo->tx_opcnt_st[batch_id * batch_size];
            }
            else
            {
                common::StaticTransactionSet_CPU *stinfo = dynamic_cast<common::StaticTransactionSet_CPU *>(info);
                opcnt = stinfo->opcnt * batched_txnum;
            }
            thrust::sort(thrust::cuda::par.on(stream), access_table, access_table + opcnt);
            // printf("3\n");

            num_blocks = dim3(acc_cnt_cpu / 1024 + (acc_cnt_cpu % 1024 > 0), 1, 1);
            num_threads = dim3(acc_cnt_cpu > 1024 ? 1024 : acc_cnt_cpu, 1, 1);
            gacco_pre2<<<num_blocks, num_threads, 0, stream>>>(acc_cnt_cpu, aux_table, aux_ori_idx, aux_dense_cnt, access_table);
            // cudaStreamSynchronize(stream);
            // printf("4\n");
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D GACCO_RUN");
            opts.push_back(std::string("-D AUX_TABLE=") + std::to_string((unsigned long long)aux_table));
            opts.push_back(std::string("-D ACCESS_TABLE=") + std::to_string((unsigned long long)access_table));
            opts.push_back(std::string("-D CC_TYPE=cc::Gacco_GPU"));
        }

        void *ToGPU() override
        {
            return nullptr;
        }

        size_t GetMemSize() override
        {
            return sizeof(AuxiliaryItem) * (obj_cnt + 1) +
                   sizeof(AccessItem) * info->tot_operation_cnt +
                   sizeof(int) * info->tot_operation_cnt * 2;
        }

    private:
        std::string pre1_code;
        std::string pre1_5_code;
        common::RTCProgram *program1;
        common::RTCProgram *program1_5;
        int *aux_ori_idx;
        int *aux_dense_cnt;
        int *acc_cnt;

        // std::string genPre1StaticCode(common::StaticTransactionSet_CPU *txinfo)
        // {
        //     std::stringstream ret;
        //     std::string tx_struct_name = ((common::StructType *)(txinfo->type))->name;
        //     ret << "#include<db.cuh>\n";
        //     ret << "#include<gacco.cuh>\n";
        //     ret << "#include<index.cuh>\n";
        //     ret << "#include <cooperative_groups.h>\n";
        //     ret << txinfo->type->GetName();
        //     // ret << "__device__ unsigned long long global_st = 0;\n";
        //     ret << "__global__ void pre1(" << tx_struct_name << " *txs, "
        //         << "int batch_id, common::DB_GPU *db, cc::AuxiliaryItem *aux_table, int *ori_idx, int *dense_cnt, cc::AccessItem *access_table, int *acc_cnt) {\n";
        //     ret << "size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;\n";
        //     ret << "size_t tid = batch_id * " << batch_size << " + thread_id;\n";
        //     ret << "if(tid >= " << txinfo->tx_cnt << ") return;\n";
        //     ret << "if(thread_id == 0) { *acc_cnt = 0;}\n";
        //     ret << "__threadfence();";
        //     ret << "int prev;\n";
        //     ret << tx_struct_name << " tx = txs[tid];";
        //     ret << txinfo->GetAccessorGPUCode();

        //     int opcnt = txinfo->opcnt;

        //     ret << "size_t st = thread_id * " << opcnt << ";\n";

        //     for (int i = 0; i < txinfo->rcnt; i++)
        //     {
        //         common::TxAccessor &acc = txinfo->rset[i];
        //         ret << "size_t auxoff_r" << i << " = " << acc.name << "_idx + "
        //             << "TABLE_" << acc.table_idx << "_ST;\n";
        //         ret << "prev = atomicAdd(&(aux_table[auxoff_r" << i << "].offset), 1);\n"; // atomicAdd(&(aux_table + XXX_idx + TABLE_idx_ST)->offset, 1);
        //         ret << "if(prev == 0) aux_table[auxoff_r" << i << "].lock = atomicAdd(acc_cnt, 1);\n";
        //         ret << "access_table[st].obj_idx = auxoff_r" << i << ";\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
        //         ret << "access_table[st++].tx_idx = tid;\n";                // access_table[st++].tx_idx =  tid;
        //     }

        //     for (int i = 0; i < txinfo->wcnt; i++)
        //     {
        //         common::TxAccessor &acc = txinfo->wset[i];
        //         ret << "size_t auxoff_w" << i << " = " << acc.name << "_idx + "
        //             << "TABLE_" << acc.table_idx << "_ST;\n";
        //         ret << "prev = atomicAdd(&(aux_table[auxoff_w" << i << "].offset), 1);\n"; // atomicAdd(&(aux_table + XXX_idx + TABLE_idx_ST)->offset, 1);
        //         ret << "if(prev == 0) aux_table[auxoff_w" << i << "].lock = atomicAdd(acc_cnt, 1);\n";
        //         ret << "access_table[st].obj_idx = auxoff_w" << i << ";\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
        //         ret << "access_table[st++].tx_idx = tid;\n";                // access_table[st++].tx_idx =  tid;
        //     }

        //     ret << "cooperative_groups::grid_group grid = cooperative_groups::this_grid();\n";
        //     ret << "grid.sync();\n";

        //     ret << "cc::AuxiliaryItem *auxp;\n";
        //     for (int i = 0; i < txinfo->rcnt; i++)
        //     {
        //         ret << "auxp = aux_table + auxoff_r" << i << ";\n";
        //         ret << "ori_idx[auxp->lock] = auxoff_r" << i << ";\n";
        //         ret << "dense_cnt[auxp->lock] = auxp->offset;\n";
        //     }

        //     for (int i = 0; i < txinfo->wcnt; i++)
        //     {
        //         ret << "auxp = aux_table + auxoff_w" << i << ";\n";
        //         ret << "ori_idx[auxp->lock] = auxoff_w" << i << ";\n";
        //         ret << "dense_cnt[auxp->lock] = auxp->offset;\n";
        //     }

        //     ret << "}\n";

        //     return ret.str();
        // }
        std::string genPre1StaticCode(common::StaticTransactionSet_CPU *txinfo)
        {
            std::stringstream ret;
            std::string tx_struct_name = ((common::StructType *)(txinfo->type))->name;
            ret << "#include<db.cuh>\n";
            ret << "#include<gacco.cuh>\n";
            ret << "#include<index.cuh>\n";
            // ret << "#include <cooperative_groups.h>\n";
            ret << txinfo->type->GetName();
            // ret << "__device__ unsigned long long global_st = 0;\n";
            ret << "__global__ void pre1(" << tx_struct_name << " *txs, "
                << "int batch_id, common::DB_GPU *db, cc::AuxiliaryItem *aux_table, int *ori_idx, int *dense_cnt, cc::AccessItem *access_table, int *acc_cnt) {\n";
            ret << "size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;\n";
            ret << "size_t tid = batch_id * " << batch_size << " + thread_id;\n";
            ret << "if(tid >= " << txinfo->tx_cnt << ") return;\n";
            ret << "if(thread_id == 0) { *acc_cnt = 0;}\n";
            ret << "__threadfence();";
            ret << "int prev;\n";
            ret << tx_struct_name << " tx = txs[tid];";
            ret << txinfo->GetAccessorGPUCode();

            int opcnt = txinfo->opcnt;

            ret << "size_t st = thread_id * " << opcnt << ";\n";

            for (int i = 0; i < txinfo->rcnt; i++)
            {
                common::TxAccessor &acc = txinfo->rset[i];
                ret << "size_t auxoff_r" << i << " = " << acc.name << "_idx + "
                    << "TABLE_" << acc.table_idx << "_ST;\n";
                ret << "prev = atomicAdd(&(aux_table[auxoff_r" << i << "].offset), 1);\n"; // atomicAdd(&(aux_table + XXX_idx + TABLE_idx_ST)->offset, 1);
                ret << "if(prev == 0) aux_table[auxoff_r" << i << "].lock = atomicAdd(acc_cnt, 1);\n";
                ret << "access_table[st].obj_idx = auxoff_r" << i << ";\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
                ret << "access_table[st++].tx_idx = tid;\n";                // access_table[st++].tx_idx =  tid;
            }

            for (int i = 0; i < txinfo->wcnt; i++)
            {
                common::TxAccessor &acc = txinfo->wset[i];
                ret << "size_t auxoff_w" << i << " = " << acc.name << "_idx + "
                    << "TABLE_" << acc.table_idx << "_ST;\n";
                ret << "prev = atomicAdd(&(aux_table[auxoff_w" << i << "].offset), 1);\n"; // atomicAdd(&(aux_table + XXX_idx + TABLE_idx_ST)->offset, 1);
                ret << "if(prev == 0) aux_table[auxoff_w" << i << "].lock = atomicAdd(acc_cnt, 1);\n";
                ret << "access_table[st].obj_idx = auxoff_w" << i << ";\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
                ret << "access_table[st++].tx_idx = tid;\n";                // access_table[st++].tx_idx =  tid;
            }

            ret << "}\n";

            return ret.str();
        }

        std::string genPre1_5StaticCode(common::StaticTransactionSet_CPU *txinfo)
        {
            std::stringstream ret;
            ret << "#include<db.cuh>\n";
            ret << "#include<gacco.cuh>\n";
            ret << "#include<index.cuh>\n";
            // ret << "#include <cooperative_groups.h>\n";
            //  ret << "__device__ unsigned long long global_st = 0;\n";
            ret << "__global__ void pre1_5(int batch_id, cc::AuxiliaryItem *aux_table, int *ori_idx, int *dense_cnt, cc::AccessItem *access_table) {\n";
            ret << "size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;\n";
            ret << "size_t tid = batch_id * " << batch_size << " + thread_id;\n";
            ret << "if(tid >= " << txinfo->tx_cnt << ") return;\n";
            int opcnt = txinfo->opcnt;
            ret << "size_t st = thread_id * " << opcnt << ";\n";
            ret << "cc::AuxiliaryItem *auxp;\n";
            ret << "size_t auxoff;\n";
            for (int i = 0; i < txinfo->rcnt; i++)
            {
                ret << "auxoff = access_table[st++].obj_idx;\n";
                ret << "auxp = aux_table + auxoff;\n";
                ret << "ori_idx[auxp->lock] = auxoff;\n";
                ret << "dense_cnt[auxp->lock] = auxp->offset;\n";
            }

            for (int i = 0; i < txinfo->wcnt; i++)
            {
                ret << "auxoff = access_table[st++].obj_idx;\n";
                ret << "auxp = aux_table + auxoff;\n";
                ret << "ori_idx[auxp->lock] = auxoff;\n";
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
            ret << "#include<gacco.cuh>\n";
            ret << "#include<index.cuh>\n";
            ret << "#include<transaction.cuh>\n";
            ret << txinfo->type->GetName();
            ret << "__global__ void pre1(" << tx_struct_name << " *txs, "
                << "int batch_id, common::DynamicTransactionSet_GPU *txset_info, common::DB_GPU *db, cc::AuxiliaryItem *aux_table, int *ori_idx, int *dense_cnt, cc::AccessItem *access_table, int *acc_cnt) {\n";
            ret << "size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;\n";
            ret << "size_t batch_st = batch_id * " << batch_size << ";\n";
            ret << "size_t tid = batch_st + thread_id;\n";
            ret << "if(tid >= " << txinfo->tx_cnt << ") return;\n";
            ret << "if(thread_id == 0) {*acc_cnt = 0;}\n";
            ret << "__threadfence();\n";
            ret << "size_t global_st = txset_info->tx_opcnt_st[batch_st];\n";
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
                ret << "access_table[i - global_st].obj_idx = auxoff;\n"; // access_table[st].obj_idx =  XXX_idx + TABLE_idx_ST;
                ret << "access_table[i - global_st].tx_idx = tid;\n";     // access_table[st++].tx_idx =  tid;
                ret << "}\n";
            }
            // ret << "__threadfence();\n";
            // ret << "cc::AuxiliaryItem *auxp;\n";

            // ret << "st = txset_info->tx_opcnt_st[tid];\n";
            // ret << "en = txset_info->tx_opcnt_st[tid + 1];\n";
            // ret << "for(int i = st; i < en; i++) {\n";
            // ret << "size_t auxoff = access_table[i - global_st].obj_idx;\n";
            // ret << "auxp = aux_table + auxoff;\n";
            // ret << "ori_idx[auxp->lock] = auxoff;\n";
            // ret << "dense_cnt[auxp->lock] = auxp->offset;\n";
            // ret << "}\n";

            ret << "}\n";

            return ret.str();
        }

        std::string genPre1_5DynamicCode(common::DynamicTransactionSet_CPU *txinfo)
        {
            std::stringstream ret;
            ret << "#include<db.cuh>\n";
            ret << "#include<gacco.cuh>\n";
            ret << "#include<index.cuh>\n";
            ret << "#include<transaction.cuh>\n";
            // ret << "#include <cooperative_groups.h>\n";
            ret << "__global__ void pre1_5(int batch_id, common::DynamicTransactionSet_GPU *txset_info, cc::AuxiliaryItem *aux_table, int *ori_idx, int *dense_cnt, cc::AccessItem *access_table) {\n";
            ret << "size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;\n";
            ret << "size_t batch_st = batch_id * " << batch_size << ";\n";
            ret << "size_t tid = batch_st + thread_id;\n";
            ret << "if(tid >= " << txinfo->tx_cnt << ") return;\n";
            ret << "size_t global_st = txset_info->tx_opcnt_st[batch_st];\n";
            ret << "size_t st = txset_info->tx_opcnt_st[tid];\n";
            ret << "size_t en = txset_info->tx_opcnt_st[tid + 1];\n";
            ret << "for(int i = st; i < en; i++) {\n";
            ret << "size_t auxoff = access_table[i - global_st].obj_idx;\n";
            ret << "cc::AuxiliaryItem *auxp = aux_table + auxoff;\n";
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