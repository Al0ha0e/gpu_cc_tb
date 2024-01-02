#ifndef TX_H
#define TX_H

#ifndef NVRTC_COMPILE

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <type.cuh>
#include <generator.cuh>
#include <operator.cuh>
#include <db.cuh>

#define RCNT 1
#define WCNT 1
#define TX_CNT 0

#endif

#include <db.cuh>

namespace common
{

    typedef char *Transaction_GPU;

    struct DynamicTransactionSet_GPU
    {
        int *tx_rcnt;
        int *tx_wcnt;
        int *tx_opcnt;
        size_t *tx_rcnt_st;
        size_t *tx_wcnt_st;
        size_t *tx_opcnt_st;
        size_t *tx_detailed_opcnt_st;
    };

#ifndef NVRTC_COMPILE

    struct TxAccessor
    {
        std::string name;
        gputp_operator::Operator *acc_operations;
        int table_idx;
        int index_idx;

        TxAccessor() {}
        TxAccessor(std::string n, int tidx, int iidx, gputp_operator::Operator *ops)
            : name(n),
              table_idx(tidx),
              index_idx(iidx),
              acc_operations(ops) {}
    };

    class TransactionSet_CPU
    {
    public:
        size_t tx_cnt;
        size_t tot_operation_cnt; // = key accessed, = tx_cnt * (rcnt + wcnt) in static
        size_t tot_size;          // = tx_cnt * sizeof(type) in static, tot_operation_cnt * sizeof(type) in dynamic
        StructType *type;
        char *data;
        gputp_operator::Scope *scope;
        DB_CPU *db_cpu;

        TransactionSet_CPU()
            : tx_cnt(0), tot_operation_cnt(0), tot_size(0), type(nullptr), data(nullptr), db_cpu(nullptr) {}

        TransactionSet_CPU(
            size_t txcnt,
            StructType *tp,
            DB_CPU *db)
            : tx_cnt(txcnt),
              tot_operation_cnt(0),
              tot_size(0),
              type(tp),
              db_cpu(db)
        {
            scope = new gputp_operator::Scope;
        }

        size_t GetTxCnt() { return tx_cnt; }
        size_t GetTotOpCnt() { return tot_operation_cnt; }
        size_t GetTotSize() { return tot_size; }

        virtual void GetCompileOptions(std::vector<std::string> &opts) = 0;
    };

    class StaticTransactionSet_CPU : public TransactionSet_CPU
    {
    public:
        unsigned int rcnt;
        unsigned int wcnt;
        int opcnt;
        std::vector<TxAccessor> rset;
        std::vector<TxAccessor> wset;

        StaticTransactionSet_CPU() : rcnt(0), wcnt(0), opcnt(0), TransactionSet_CPU()
        {
        }

        StaticTransactionSet_CPU(size_t txcnt, StructType *tp, DB_CPU *db)
            : rcnt(0),
              wcnt(0),
              opcnt(0),
              TransactionSet_CPU(txcnt, tp, db)
        {
            tot_size = txcnt * tp->GetSize();
            scope->SetName("tx", tp);
        }

        void AddAccessor(std::string name, int table_idx, int index_idx, gputp_operator::Operator *op, bool write)
        {
            op->ApplyScope(scope);
            if (write)
            {
                wcnt++;
                wset.push_back(TxAccessor(name, table_idx, index_idx, op));
            }
            else
            {
                rcnt++;
                rset.push_back(TxAccessor(name, table_idx, index_idx, op));
            }
            opcnt++;
            tot_operation_cnt = opcnt * tx_cnt;
        }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back(std::string("-D RCNT=" + std::to_string(rcnt)));
            opts.push_back(std::string("-D WCNT=" + std::to_string(wcnt)));
            opts.push_back((std::string("-D TX_CNT=") + std::to_string(tx_cnt)));
        }

        std::string GetAccessorGPUCode()
        {
            scope->Clear();

            std::stringstream table_code;
            std::stringstream index_code;
            std::stringstream access_code;
            std::stringstream idx_code;

            for (int i = 0; i < rcnt; i++)
                getAccessorGPUCode(rset[i], table_code, index_code, access_code, idx_code);

            for (int i = 0; i < wcnt; i++)
                getAccessorGPUCode(wset[i], table_code, index_code, access_code, idx_code);

            return table_code.str() + index_code.str() + access_code.str() + idx_code.str();
        }

    private:
        void getAccessorGPUCode(TxAccessor &acc,
                                std::stringstream &table_code,
                                std::stringstream &index_code,
                                std::stringstream &access_code,
                                std::stringstream &idx_code)
        {
            // construct table code
            std::string table_name = std::string("table_") + std::to_string(acc.table_idx);
            if (!scope->HasName(table_name))
            {
                scope->SetName(table_name, new common::PlaceHolderType);
                table_code << "common::Table_GPU "
                           << table_name
                           << " = db->tables["
                           << acc.table_idx
                           << "];\n";
            }

            // construct access code
            std::string index_name = std::string("index_") +
                                     std::to_string(acc.table_idx) +
                                     std::string("_") +
                                     std::to_string(acc.index_idx);

            if (!scope->HasName(index_name))
            {
                scope->SetName(index_name, new common::PlaceHolderType);
                std::string index_type = db_cpu->tables[acc.table_idx]->index_types[acc.index_idx];
                index_code << index_type << " *"
                           << index_name
                           << " = (" << index_type << " *)"
                           << table_name << ".indices[" << acc.index_idx << "];\n";
            }

            std::string accop;
            acc.acc_operations->ToGPUCode(accop);

            access_code << acc.acc_operations->type->GetName() << " " << acc.name << "_id = " << accop << ";\n";

            idx_code << "size_t " << acc.name << "_idx = " << index_name << "->Find(" << acc.name << "_id);\n";
        }
    };

    class DynamicTransactionSet_CPU : public TransactionSet_CPU
    {
    public:
        std::vector<TxAccessor> accessors;
        int tx_type_cnt;
        std::vector<int> tx_rcnt;
        std::vector<int> tx_wcnt;
        std::vector<int> tx_opcnt;
        std::vector<int> tx_detailed_opcnt;
        std::vector<size_t> tx_rcnt_st;
        std::vector<size_t> tx_wcnt_st;
        std::vector<size_t> tx_opcnt_st;
        std::vector<size_t> tx_detailed_opcnt_st;

        DynamicTransactionSet_GPU *dynamic_set_gpu;

        DynamicTransactionSet_CPU() {}

        DynamicTransactionSet_CPU(
            size_t txcnt,
            int tx_type_cnt,
            StructType *tp,
            std::vector<int> &&rcnt,
            std::vector<int> &&wcnt,
            std::vector<int> &&detailed_opcnt,
            DB_CPU *db)
            : TransactionSet_CPU(txcnt, tp, db), tx_type_cnt(tx_type_cnt), tx_rcnt(rcnt), tx_wcnt(wcnt), tx_detailed_opcnt(detailed_opcnt)
        {
            size_t rtot = 0;
            size_t wtot = 0;
            size_t dtot = 0;
            for (int i = 0; i < tx_cnt; i++)
            {
                tx_opcnt.push_back(rcnt[i] + wcnt[i]);
                tx_rcnt_st.push_back(rtot);
                tx_wcnt_st.push_back(wtot);
                tx_opcnt_st.push_back(rtot + wtot);
                rtot += rcnt[i];
                wtot += wcnt[i];

                for (int j = 0; j < tx_type_cnt; j++)
                {
                    tx_detailed_opcnt_st.push_back(dtot);
                    dtot += detailed_opcnt[i * tx_type_cnt + j];
                }
            }
            tx_rcnt_st.push_back(rtot);
            tx_wcnt_st.push_back(wtot);
            tx_opcnt_st.push_back(rtot + wtot);
            tx_detailed_opcnt_st.push_back(dtot);

            tot_operation_cnt = rtot + wtot;
            scope->SetName("tx", tp);
            tot_size = tot_operation_cnt * tp->GetSize();

            DynamicTransactionSet_GPU *tmp = new DynamicTransactionSet_GPU;

            cudaMalloc(&tmp->tx_rcnt, sizeof(int) * txcnt);
            cudaMalloc(&tmp->tx_wcnt, sizeof(int) * txcnt);
            cudaMalloc(&tmp->tx_opcnt, sizeof(int) * txcnt);
            cudaMalloc(&tmp->tx_rcnt_st, sizeof(size_t) * (txcnt + 1));
            cudaMalloc(&tmp->tx_wcnt_st, sizeof(size_t) * (txcnt + 1));
            cudaMalloc(&tmp->tx_opcnt_st, sizeof(size_t) * (txcnt + 1));
            cudaMalloc(&tmp->tx_detailed_opcnt_st, sizeof(size_t) * (txcnt * tx_type_cnt + 1));

            cudaMemcpy(tmp->tx_rcnt, tx_rcnt.data(), sizeof(int) * txcnt, cudaMemcpyHostToDevice);
            cudaMemcpy(tmp->tx_wcnt, tx_wcnt.data(), sizeof(int) * txcnt, cudaMemcpyHostToDevice);
            cudaMemcpy(tmp->tx_opcnt, tx_opcnt.data(), sizeof(int) * txcnt, cudaMemcpyHostToDevice);
            cudaMemcpy(tmp->tx_rcnt_st, tx_rcnt_st.data(), sizeof(size_t) * (txcnt + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(tmp->tx_wcnt_st, tx_wcnt_st.data(), sizeof(size_t) * (txcnt + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(tmp->tx_opcnt_st, tx_opcnt_st.data(), sizeof(size_t) * (txcnt + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(tmp->tx_detailed_opcnt_st, tx_detailed_opcnt_st.data(), sizeof(size_t) * (txcnt * tx_type_cnt + 1), cudaMemcpyHostToDevice);

            cudaMalloc(&dynamic_set_gpu, sizeof(DynamicTransactionSet_GPU));
            cudaMemcpy(dynamic_set_gpu, tmp, sizeof(DynamicTransactionSet_GPU), cudaMemcpyHostToDevice);
            delete tmp;
        }

        void AddAccessor(std::string name, int table_idx, int index_idx, gputp_operator::Operator *op, bool write)
        {
            op->ApplyScope(scope);
            accessors.push_back(TxAccessor(name, table_idx, index_idx, op));
        }

        DynamicTransactionSet_GPU *ToGPU()
        {
            return dynamic_set_gpu;
        }

        int *GetRcnt()
        {
            return tx_rcnt.data();
        }

        int *GetWcnt() { return tx_wcnt.data(); }

        size_t *GetRcntSt() { return tx_rcnt_st.data(); }

        size_t *GetWcntSt() { return tx_wcnt_st.data(); }

        size_t *GetOpcntSt() { return tx_opcnt_st.data(); }

        size_t GetTotR() { return tx_rcnt_st[tx_cnt]; }

        size_t GetTotW() { return tx_wcnt_st[tx_cnt]; }

        void GetCompileOptions(std::vector<std::string> &opts) override
        {
            opts.push_back("-D RCNT=rcnt");
            opts.push_back("-D WCNT=wcnt");
            opts.push_back((std::string("-D TX_CNT=") + std::to_string(tx_cnt)));
            opts.push_back("-D DYNAMIC_RW_COUNT");
        }

        std::string GetGlobalAccessorGPUCode()
        {
            scope->Clear();

            std::stringstream table_code;
            std::stringstream index_code;

            for (auto &accessor : accessors)
            {
                std::string table_name = std::string("table_") + std::to_string(accessor.table_idx);
                if (!scope->HasName(table_name))
                {
                    scope->SetName(table_name, new common::PlaceHolderType);
                    table_code << "common::Table_GPU "
                               << table_name
                               << " = db->tables["
                               << accessor.table_idx
                               << "];\n";
                }

                std::string index_name = std::string("index_") +
                                         std::to_string(accessor.table_idx) +
                                         std::string("_") +
                                         std::to_string(accessor.index_idx);
                if (!scope->HasName(index_name))
                {
                    scope->SetName(index_name, new common::PlaceHolderType);
                    std::string index_type = db_cpu->tables[accessor.table_idx]->index_types[accessor.index_idx];
                    index_code << index_type << " *" << index_name
                               << " = (" << index_type << " *)"
                               << table_name << ".indices[" << accessor.index_idx << "];\n";
                }
            }

            return table_code.str() + index_code.str();
        }

        std::string GetLocalAccessorGPUCode(int acc_idx)
        {
            std::stringstream ret;
            std::string accop;
            TxAccessor &accessor = accessors[acc_idx];

            std::string index_name = std::string("index_") +
                                     std::to_string(accessor.table_idx) +
                                     std::string("_") +
                                     std::to_string(accessor.index_idx);

            accessor.acc_operations->ToGPUCode(accop);

            ret << accessor.acc_operations->type->GetName() << " " << accessor.name << "_id = " << accop << ";\n";
            ret << "size_t " << accessor.name << "_idx = " << index_name << "->Find(" << accessor.name << "_id);\n";

            return ret.str();
        }
    };

#endif
}

#endif