#ifndef DB_H
#define DB_H

#include <table.cuh>

#ifndef NVRTC_COMPILE

#include <vector>

#define TABLE_0_ST 0
#define TABLE_1_ST 0
#define TABLE_2_ST 0
#define TABLE_3_ST 0
#define TABLE_4_ST 0
#define TABLE_5_ST 0
#define TABLE_6_ST 0
#define TABLE_7_ST 0
#define TABLE_8_ST 0
#define TABLE_9_ST 0
#define TABLE_10_ST 0
#define TABLE_11_ST 0

#define TABLE_ST_ARR __device__ const size_t table_st_arr[10] = {0, 0, 0, 114514, 0, 0, 0, 0, 0, 0};
#define TABLE_MEM_ST_ARR __device__ const size_t table_mem_st_arr[10] = {0, 0, 0, 0, 0, 0, 0, 1919810, 0, 0};

#endif

namespace common
{

    class DB_GPU
    {
    public:
        unsigned int table_cnt;
        Table_GPU *tables;
        char *table_data;
        size_t *table_st;
        size_t *table_mem_st;
    };

#ifndef NVRTC_COMPILE
    class DB_CPU
    {
    public:
        unsigned int table_cnt;
        std::vector<Table_CPU *> tables;
        std::vector<size_t> entry_cnt;
        std::vector<size_t> table_st;
        std::vector<size_t> table_mem_st;

        DB_CPU() : table_cnt(0) {}

        void GetCompileOptions(std::vector<std::string> &opts)
        {
            for (int i = 0; i < table_cnt; i++)
                opts.push_back(std::string("-D TABLE_") + std::to_string(i) + std::string("_ST=") + std::to_string(table_st[i]));

            std::string table_st_str = "-D TABLE_ST_ARR=__device__ const size_t table_st_arr[";
            table_st_str += std::to_string(table_cnt + 1) + "]={";
            for (int i = 0; i < table_cnt; i++)
                table_st_str += std::to_string(table_st[i]) + ",";
            table_st_str += std::to_string(table_st[table_cnt]) + "};";
            opts.push_back(table_st_str);

            // opts.push_back("-D TABLE_ST_ARR=__device__ const size_t table_st_arr[10] = {0, 0, 0, 114514, 0, 0, 0, 0, 0, 0};");
            // opts.push_back("-D TABLE_MEM_ST_ARR=__device__ const size_t table_mem_st_arr[10] = {0, 0, 0, 0, 0, 0, 0, 1919810, 0, 0};");
        }

        void AddTable(Table_CPU *table)
        {
            table_cnt++;
            tables.push_back(table);
        }

        void AddTables(std::vector<Table_CPU *> &tbs)
        {
            for (auto tb : tbs)
                AddTable(tb);
        }

        void Init()
        {
            size_t tot = 0;
            size_t mem_tot = 0;
            for (auto &table : tables)
            {
                entry_cnt.push_back(table->entry_cnt);
                table_st.push_back(tot);
                table_mem_st.push_back(mem_tot);
                tot += table->entry_cnt;
                mem_tot += table->data_size;
            }
            table_st.push_back(tot);
            table_mem_st.push_back(mem_tot);
        }

        DB_GPU *ToGPU()
        {
            DB_GPU *ret = nullptr;
            DB_GPU *tmp = new DB_GPU;

            tmp->table_cnt = table_cnt;
            cudaMalloc(&(tmp->tables), table_cnt * sizeof(Table_GPU));
            cudaMalloc(&(tmp->table_data), table_mem_st[table_cnt]);
            Table_GPU *table_gpu = tmp->tables;
            char *data_gpu = tmp->table_data;

            for (auto &table : tables)
            {
                table->ToGPU(table_gpu, data_gpu);
                table_gpu++;
                data_gpu += table->data_size;
            }

            cudaMalloc(&tmp->table_st, sizeof(size_t) * (table_cnt + 1));
            cudaMemcpy(tmp->table_st, table_st.data(), sizeof(size_t) * (table_cnt + 1), cudaMemcpyHostToDevice);

            cudaMalloc(&tmp->table_mem_st, sizeof(size_t) * (table_cnt + 1));
            cudaMemcpy(tmp->table_mem_st, table_mem_st.data(), sizeof(size_t) * (table_cnt + 1), cudaMemcpyHostToDevice);

            cudaMalloc(&ret, sizeof(DB_GPU));
            cudaMemcpy(ret, tmp, sizeof(DB_GPU), cudaMemcpyHostToDevice);

            delete tmp;
            return ret;
        }

        size_t GetMemSize()
        {
            return table_cnt * sizeof(Table_GPU) + table_mem_st[table_cnt] + sizeof(size_t) * (table_cnt + 1) * 2 + sizeof(DB_GPU);
        }
    };
#endif

}

#endif