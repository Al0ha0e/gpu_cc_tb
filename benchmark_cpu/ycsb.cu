#include <db.cuh>
#include <table.cuh>
#include <runtime.cuh>
#include <env.cuh>
#include <tictoc.cuh>
#include <gacco.cuh>
#include <gputx.cuh>
#include <silo.cuh>
#include <to.cuh>
#include <mvcc.cuh>
#include <slow_twopl.cuh>
#include <slow_to.cuh>
#include <slow_mvcc.cuh>
#include <slow_tictoc.cuh>
#include <slow_silo.cuh>
#include <twopl.cuh>
#include <timestamp.cuh>
#include <ctime>
#include <set>
#include <thread>
#include <mutex>
#include <stdio.h>
#include <ycsb.cuh>

using namespace bench_ycsb;

const int SEED = 141919810;

// std::vector<common::TypeWithSize> info{
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
//     common::TypeWithSize(common::INT64, 8, 0),
// };
std::vector<common::TypeWithSize> info{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
    common::TypeWithSize(common::STRING, 10, 0),
};

common::GeneratorBase *default_donothing_generator = new common::DoNothingGenerator;

std::vector<common::GeneratorBase *> gens{
    new common::SequenceIntegerGenerator(0, 1, 0xffffffff, 4, SEED),
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator,
    default_donothing_generator};

using StructItem = common::StructType::StructItem;

std::vector<common::StructType::StructItem> struct_items{
    StructItem("read", new common::IntegerType(common::INT32, false), 0),
    StructItem("key", new common::IntegerType(common::INT32, true), 0)};

common::StructType struct_type("TxItem", std::move(struct_items));

int rownum;

int main(int argc, char **argv)
{
    common::Context context;

    srand(SEED);
    std::string bench_name("ycsb");
    std::string path("./benchmark/ycsb.cu");
    std::string include_path("./include");
    int txcnt, valid_txn_bitoffset, warp_cnt, batch_size, debug;
    sscanf(argv[2], "%d", &rownum);
    sscanf(argv[3], "%d", &txcnt);
    std::string cc_type(argv[4]);
    sscanf(argv[5], "%d", &valid_txn_bitoffset);
    sscanf(argv[6], "%d", &warp_cnt);
    sscanf(argv[7], "%d", &batch_size);
    sscanf(argv[8], "%d", &debug);
    if (warp_cnt > 32 || valid_txn_bitoffset > 5)
        return -1;

    std::cout << "TRY " << argv[4] << "," << valid_txn_bitoffset << "," << warp_cnt << "," << batch_size << "\n";

    common::DB_CPU db;

    common::Table_CPU *table = new common::Table_CPU;
    table->AddColumns(info, gens);
    table->InitData(rownum);
    table->AddIndex(common::SORTED_ARRAY, 0);

    db.AddTable(table);
    db.Init();

    common::DB_GPU *db_gpu = db.ToGPU();

    // std::cout << "GPUDB SIZE " << db.GetMemSize() << "\n";

    std::vector<int> rcnts(txcnt);
    std::vector<int> wcnts(txcnt);
    std::vector<int> detailed_opcnts(txcnt);

    YCSBTx *txdata = new YCSBTx[txcnt];

    FILE *file = fopen(argv[1], "rb");
    fread(txdata, sizeof(YCSBTx), txcnt, file);
    fclose(file);

    std::vector<YCSBReq> gacco_txdata;
    for (int i = 0; i < txcnt; i++)
    {
        int rcnt = 0, wcnt = 0, detailed_opcnt;
        for (int j = 0; j < txdata[i].request_cnt; j++)
        {
            gacco_txdata.push_back(txdata[i].requests[j]);
            txdata[i].requests[j].read ? ++rcnt : ++wcnt;
        }
        detailed_opcnt = rcnt + wcnt;
        rcnts[i] = rcnt;
        wcnts[i] = wcnt;
        detailed_opcnts[i] = detailed_opcnt;
    }

    char *gpu_gaccotxs;
    cudaMalloc(&gpu_gaccotxs, sizeof(YCSBReq) * gacco_txdata.size());
    cudaMemcpy(gpu_gaccotxs, gacco_txdata.data(), sizeof(YCSBReq) * gacco_txdata.size(), cudaMemcpyHostToDevice);

    char *gpu_txs;
    size_t tx_size = txcnt * sizeof(YCSBTx);
    cudaMalloc(&gpu_txs, tx_size);
    cudaMemcpy(gpu_txs, txdata, tx_size, cudaMemcpyHostToDevice);

    common::DynamicTransactionSet_CPU *txinfo = new common::DynamicTransactionSet_CPU(
        txcnt,
        1,
        &struct_type,
        std::move(rcnts),
        std::move(wcnts),
        std::move(detailed_opcnts),
        &db);

    gputp_operator::Operator *op1 = new gputp_operator::StructMemberAccessOperator(
        new gputp_operator::VariableOperator("tx"),
        "key");
    op1->type = new common::IntegerType(common::INT32, true);
    txinfo->AddAccessor("", 0, 0, op1, false);

    common::TSAllocator_CPU *ts_allocator = new common::NaiveTSAllocator_CPU();
    common::ConcurrencyControlCPUBase *cc = nullptr;

    if (cc_type == "tictoc")
        cc = new cc::Tictoc_CPU(&db, txinfo, batch_size);
    else if (cc_type == "slow_tictoc")
        cc = new cc::Slow_Tictoc_CPU(&db, txinfo, batch_size);
    else if (cc_type == "silo")
        cc = new cc::Silo_CPU(&db, txinfo, batch_size);
    else if (cc_type == "slow_silo")
        cc = new cc::Slow_Silo_CPU(&db, txinfo, batch_size);
    else if (cc_type == "to")
        cc = new cc::TO_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "slow_to")
        cc = new cc::Slow_TO_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "mvcc")
        cc = new cc::MVCC_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "slow_mvcc")
        cc = new cc::Slow_MVCC_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "tpl_nw")
        cc = new cc::TwoPL_CPU(&db, txinfo, false, batch_size);
    else if (cc_type == "tpl_wd")
        cc = new cc::TwoPL_CPU(&db, txinfo, true, batch_size);
    else if (cc_type == "slow_tpl_nw")
        cc = new cc::Slow_TwoPL_CPU(&db, txinfo, false, batch_size);
    else if (cc_type == "slow_tpl_wd")
        cc = new cc::Slow_TwoPL_CPU(&db, txinfo, true, batch_size);
    else if (cc_type == "gacco")
        cc = new cc::Gacco_CPU(&db, db_gpu, txinfo, gpu_gaccotxs, batch_size);
    else if (cc_type == "gputx")
        cc = new cc::GPUTx_CPU(&db, db_gpu, txinfo, gpu_gaccotxs, batch_size);

    common::ExecInfo exec_info(argv[1], cc, txinfo, txcnt, argv[4],
                               valid_txn_bitoffset, warp_cnt, batch_size, debug);

    CUdeviceptr cc_device_info = (CUdeviceptr)cc->ToGPU();
    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());

    common::RTCProgram program(path, include_path, "bench_ycsb", true);
    program.AddNameExpression(bench_name);

    std::vector<std::string> str_opts = {
        (std::string("-I ") + include_path),
        std::string("-I /usr/include"),
        "-D NVRTC_COMPILE",
        "-lineinfo",
        // "-G",
        "-D INDEX_0_TYPE=common::SortedArray_GPU",
        "-arch=compute_89",
        "-D THREAD_TX_SHIFT=" + std::to_string(valid_txn_bitoffset)};

    exec_info.GetCompileOptions(str_opts);
    txinfo->GetCompileOptions(str_opts);
    cc->GetCompileOptions(str_opts);
    db.GetCompileOptions(str_opts);

    std::vector<const char *> opts;
    for (auto &s : str_opts)
        opts.push_back(s.c_str());

    if (!program.Compile(opts))
        exit(1);

    void *txset_info = txinfo->ToGPU();

    char *output = nullptr;

    void *args[] = {
        &db_gpu,
        &gpu_txs,
        &txset_info,
        nullptr,
        nullptr,
        nullptr,
        &cc_device_info,
        &output};

    cudaEvent_t start, stop, start1, mid1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&mid1);
    cudaEventCreate(&stop1);
    float elapsed_time;
    cudaEventRecord(start);

    cc->Init0();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&elapsed_time, start, stop1);
    exec_info.precompute_time = elapsed_time;

    std::vector<size_t> &batches = cc->batches;
    int batch_st = 0;
    int batch_en = 0;
    size_t block_size = warp_cnt << 5;

    for (int i = 0; i < batches.size(); i++)
    {
        cudaEventRecord(start1);
        size_t batched_txnum = batches[i];
        if (!batched_txnum)
            continue;
        batch_en = batch_st + batched_txnum;
        size_t thread_num = batched_txnum << valid_txn_bitoffset;
        dim3 num_blocks(thread_num / block_size + (thread_num % block_size == 0 ? 0 : 1), 1, 1);
        dim3 num_threads(thread_num > block_size ? block_size : thread_num, 1, 1);
        args[3] = &i;
        args[4] = &batch_st;
        args[5] = &batch_en;
        cc->Init(i, batch_st);
        cudaEventRecord(mid1);
        program.Call(bench_name, num_blocks, num_threads, args, cc->streams[i]);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&elapsed_time, start1, mid1);
        exec_info.precompute_time += elapsed_time;
        cudaEventElapsedTime(&elapsed_time, mid1, stop1);
        exec_info.processing_time += elapsed_time;
        batch_st = batch_en;
    }
    cc->Sync();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    exec_info.tot_time = elapsed_time;
    exec_info.Analyse();
    if (debug)
        exec_info.Check();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(mid1);
    return 0;
}