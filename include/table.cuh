#ifndef TABLE_H
#define TABLE_H

#include <index.cuh>
#include <type.cuh>

#ifndef NVRTC_COMPILE

#include <vector>
#include <generator.cuh>
#include <stdlib.h>

#endif

namespace common
{

    class Table_GPU
    {
    public:
        char *data;
        size_t entry_size;
        size_t entry_cnt;
        int index_cnt;
        void **indices;

        __device__ char *GetEntry(size_t idx)
        {
            return data + idx * entry_size;
        }
    };
#ifndef NVRTC_COMPILE

    class Table_CPU
    {
    public:
        std::vector<TypeWithSize> columns;
        std::vector<GeneratorBase *> generators;
        size_t entry_size;
        size_t entry_cnt;
        size_t data_size;
        char *data_cpu;
        std::vector<void *> gpu_indices;
        std::vector<std::string> index_types;

        Table_CPU() : entry_size(0), entry_cnt(0) {}

        void AddColumn(TypeWithSize info, GeneratorBase *generator)
        {
            info.offset = 0;
            if (columns.size() > 0)
            {
                TypeWithSize &prev = columns[columns.size() - 1];
                info.offset = prev.offset + prev.size + prev.padding;
            }
            columns.push_back(info);
            generators.push_back(generator);
            entry_size += info.size + info.padding;
        }

        void AddColumns(std::vector<TypeWithSize> &infos, std::vector<GeneratorBase *> gs)
        {
            for (int i = 0; i < infos.size(); i++)
                AddColumn(infos[i], gs[i]);
        }

        void InitData(size_t ecnt)
        {
            entry_cnt = ecnt;
            data_size = entry_cnt * entry_size;

            data_cpu = new char[data_size];
            for (int i = 0; i < columns.size(); i++)
                generators[i]->GenerateTo(columns[i], entry_cnt, entry_size, data_cpu);
        }

        void LoadData(size_t ecnt, std::string pth)
        {
            entry_cnt = ecnt;
            data_size = entry_cnt * entry_size;
            data_cpu = new char[data_size];
            FILE *file = fopen(pth.c_str(), "rb");
            fread(data_cpu, sizeof(char), data_size, file);
            fclose(file);
        }

        void SaveData(std::string pth)
        {
            data_size = entry_cnt * entry_size;
            FILE *file = fopen(pth.c_str(), "wb");
            fwrite(data_cpu, sizeof(char), data_size, file);
            fclose(file);
        }

        void AddIndex(IndexType type, int column)
        {
            TypeWithSize info = columns[column];

            char *st = data_cpu + info.offset;

            index_types.push_back(GetIndexName(type, info.type));

            switch (info.type)
            {
            case INT32:
            {
                std::vector<IndexItem<int, size_t>> tmp;
                for (int i = 0; i < entry_cnt; i++)
                {
                    tmp.push_back(IndexItem<int, size_t>(*((int *)st), i));
                    st += entry_size;
                }
                gpu_indices.push_back(GetGPUIndex(tmp.data(), entry_cnt, type));
            }
            break;
            case INT64:
            {
                std::vector<IndexItem<long long, size_t>> tmp;
                for (int i = 0; i < entry_cnt; i++)
                {
                    tmp.push_back(IndexItem<long long, size_t>(*((long long *)st), i));
                    st += entry_size;
                }
                gpu_indices.push_back(GetGPUIndex(tmp.data(), entry_cnt, type));
            }
            break;
            default:
                break;
            }
        }

        void AddIndex(IndexType type, std::vector<int> cols, std::vector<int> coeff)
        {
            std::vector<IndexItem<int, size_t>> tmp;
            char *st = data_cpu;
            for (int i = 0; i < entry_cnt; i++)
            {
                int sum = 0;
                for (int j = 0; j < cols.size(); j++)
                    sum += *(int *)(st + columns[cols[j]].offset) * coeff[j];
                tmp.push_back(IndexItem<int, size_t>(sum, i));
                st += entry_size;
            }
            index_types.push_back(GetIndexName(type, INT32));
            gpu_indices.push_back(GetGPUIndex(tmp.data(), entry_cnt, type));
        }

        void SaveIndex(IndexType type, int column, std::string pth)
        {
            TypeWithSize info = columns[column];
            char *st = data_cpu + info.offset;

            switch (info.type)
            {
            case INT32:
            {
                std::vector<IndexItem<int, size_t>> tmp;
                for (int i = 0; i < entry_cnt; i++)
                {
                    tmp.push_back(IndexItem<int, size_t>(*((int *)st), i));
                    st += entry_size;
                }
                FILE *file = fopen(pth.c_str(), "wb");
                fwrite(tmp.data(), sizeof(IndexItem<int, size_t>), entry_cnt, file);
                fclose(file);
            }
            break;
            case INT64:
            {
                std::vector<IndexItem<long long, size_t>> tmp;
                for (int i = 0; i < entry_cnt; i++)
                {
                    tmp.push_back(IndexItem<long long, size_t>(*((long long *)st), i));
                    st += entry_size;
                }
                FILE *file = fopen(pth.c_str(), "wb");
                fwrite(tmp.data(), sizeof(IndexItem<long long, size_t>), entry_cnt, file);
                fclose(file);
            }
            break;
            default:
                break;
            }
        }

        void SaveIndex(IndexType type, std::vector<int> cols, std::vector<int> coeff, std::string pth)
        {
            std::vector<IndexItem<int, size_t>> tmp;
            char *st = data_cpu;
            for (int i = 0; i < entry_cnt; i++)
            {
                int sum = 0;
                for (int j = 0; j < cols.size(); j++)
                    sum += *(int *)(st + columns[cols[j]].offset) * coeff[j];
                tmp.push_back(IndexItem<int, size_t>(sum, i));
                st += entry_size;
            }

            FILE *file = fopen(pth.c_str(), "wb");
            fwrite(tmp.data(), sizeof(IndexItem<int, size_t>), entry_cnt, file);
            fclose(file);
        }

        void LoadIndex(IndexType type, int column, std::string pth)
        {
            TypeWithSize info = columns[column];
            index_types.push_back(GetIndexName(type, info.type));

            switch (info.type)
            {
            case INT32:
            {
                IndexItem<int, size_t> *index_data = new IndexItem<int, size_t>[entry_cnt];
                FILE *file = fopen(pth.c_str(), "rb");
                fread(index_data, sizeof(IndexItem<int, size_t>), entry_cnt, file);
                fclose(file);
                gpu_indices.push_back(GetGPUIndex(index_data, entry_cnt, type));
            }
            break;
            case INT64:
            {
                IndexItem<long long, size_t> *index_data = new IndexItem<long long, size_t>[entry_cnt];
                FILE *file = fopen(pth.c_str(), "rb");
                fread(index_data, sizeof(IndexItem<long long, size_t>), entry_cnt, file);
                fclose(file);
                gpu_indices.push_back(GetGPUIndex(index_data, entry_cnt, type));
            }
            break;
            default:
                break;
            }
        }

        void LoadIndex(IndexType type, std::string pth)
        {
            IndexItem<int, size_t> *index_data = new IndexItem<int, size_t>[entry_cnt];
            FILE *file = fopen(pth.c_str(), "rb");
            fread(index_data, sizeof(IndexItem<int, size_t>), entry_cnt, file);
            fclose(file);
            index_types.push_back(GetIndexName(type, INT32));
            gpu_indices.push_back(GetGPUIndex(index_data, entry_cnt, type));
        }

        Table_GPU *ToGPU()
        {
            Table_GPU *ret = nullptr;
            Table_GPU *tmp = new Table_GPU;

            cudaMalloc(&tmp->data, data_size);
            cudaMemcpy(tmp->data, data_cpu, data_size, cudaMemcpyHostToDevice);
            cudaMalloc(&tmp->indices, sizeof(void *) * gpu_indices.size());
            cudaMemcpy(tmp->indices, gpu_indices.data(), sizeof(void *) * gpu_indices.size(), cudaMemcpyHostToDevice);
            tmp->index_cnt = gpu_indices.size();
            tmp->entry_size = entry_size;
            tmp->entry_cnt = entry_cnt;

            cudaMalloc(&ret, sizeof(Table_GPU));
            cudaMemcpy(ret, tmp, sizeof(Table_GPU), cudaMemcpyHostToDevice);
            delete tmp;
            return ret;
        }

        void ToGPU(Table_GPU *table_gpu, char *data_gpu)
        {
            Table_GPU *tmp = new Table_GPU;
            tmp->data = data_gpu;
            cudaMemcpy(tmp->data, data_cpu, data_size, cudaMemcpyHostToDevice);
            cudaMalloc(&tmp->indices, sizeof(void *) * gpu_indices.size());
            cudaMemcpy(tmp->indices, gpu_indices.data(), sizeof(void *) * gpu_indices.size(), cudaMemcpyHostToDevice);
            tmp->index_cnt = gpu_indices.size();
            tmp->entry_size = entry_size;
            tmp->entry_cnt = entry_cnt;

            cudaMemcpy(table_gpu, tmp, sizeof(Table_GPU), cudaMemcpyHostToDevice);
            delete tmp;
        }

        char *GetEntry(unsigned int column_id, size_t row_id)
        {
            return data_cpu + row_id * entry_size + columns[column_id].offset;
        }

        char *GetEntry(size_t row_id, size_t offset)
        {
            return data_cpu + row_id * entry_size + offset;
        }

        size_t GetMemSize()
        {
            return data_size + sizeof(void *) * gpu_indices.size() + sizeof(Table_GPU);
        }
    };

#endif
};

#endif