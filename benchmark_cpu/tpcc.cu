#include <db.cuh>
#include <table.cuh>
#include <runtime.cuh>
#include <env.cuh>
#include <tictoc.cuh>
#include <gacco.cuh>
#include <gputx.cuh>
#include <silo.cuh>
#include <to.cuh>
#include <slow_to.cuh>
#include <slow_mvcc.cuh>
#include <twopl.cuh>
#include <mvcc.cuh>
#include <timestamp.cuh>
#include <tpcc.cuh>
#include <ctime>
#include <set>
#include <algorithm>

const int SEED = 141919810;

using namespace tpcc;

common::GeneratorBase *do_nothing_generator = new common::DoNothingGenerator;

std::vector<common::TypeWithSize> infos_warehouse{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 4),
    // common::TypeWithSize(common::STRING, 81, 3)
};

std::vector<common::GeneratorBase *> gens_warehouse{
    new common::SequenceIntegerGenerator(0, 1, 0xffffffff, 4, SEED),
    new common::RandomFloatGenerator(0.0, 0.2, 4, SEED),
    new common::ConstFloatGenerator(300000, 4),
    // do_nothing_generator
};

std::vector<common::TypeWithSize> infos_district{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 4),
    // common::TypeWithSize(common::STRING, 81, 3)
};

std::vector<common::GeneratorBase *> gens_district{
    new common::SequenceIntegerGenerator(
        4,
        SEED,
        std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
            common::SequenceIntegerGenerator::SequenceInfo{0, 1, DISTRICTS_PER_W},
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
        }),
    new common::SequenceIntegerGenerator(
        4,
        SEED,
        std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, DISTRICTS_PER_W},
            common::SequenceIntegerGenerator::SequenceInfo{0, 1, 0xffffffff},
        }),
    new common::RandomFloatGenerator(0.0, 0.2, 4, SEED),
    new common::ConstFloatGenerator(30000, 4),
    new common::ConstIntegerGenerator(3001, 4),
    // do_nothing_generator
};

std::vector<common::TypeWithSize> infos_customer{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    // common::TypeWithSize(common::STRING, 623, 1),
};

std::vector<common::GeneratorBase *> gens_customer{
    new common::SequenceIntegerGenerator(
        4,
        SEED,
        std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
            common::SequenceIntegerGenerator::SequenceInfo{0, 1, CUSTOMERS_PER_D},
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, DISTRICTS_PER_W},
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
        }),
    new common::SequenceIntegerGenerator(
        4,
        SEED,
        std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, CUSTOMERS_PER_D},
            common::SequenceIntegerGenerator::SequenceInfo{0, 1, DISTRICTS_PER_W},
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
        }),
    new common::SequenceIntegerGenerator(
        4,
        SEED,
        std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, CUSTOMERS_PER_D},
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, DISTRICTS_PER_W},
            common::SequenceIntegerGenerator::SequenceInfo{0, 1, 0xffffffff},
        }),
    new common::ConstIntegerGenerator(0, 4),
    new common::ConstFloatGenerator(50000, 4),
    new common::RandomFloatGenerator(0.0, 0.5, 4, SEED),
    new common::ConstFloatGenerator(-10, 4),
    new common::ConstFloatGenerator(10, 4),
    new common::ConstIntegerGenerator(1, 4),
    new common::ConstIntegerGenerator(0, 4),
    // do_nothing_generator
};

std::vector<common::TypeWithSize> infos_item{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 4),
    // common::TypeWithSize(common::STRING, 74, 2),
};

std::vector<common::GeneratorBase *> gens_item{
    new common::SequenceIntegerGenerator(0, 1, 0xffffffff, 4, SEED),
    new common::ConstIntegerGenerator(0, 4),
    new common::ConstFloatGenerator(0, 4),
    // do_nothing_generator
};

std::vector<common::TypeWithSize> infos_stock{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    // common::TypeWithSize(common::STRING, 290, 6),
};

std::vector<common::GeneratorBase *> gens_stock{
    new common::SequenceIntegerGenerator(
        4,
        SEED,
        std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
            common::SequenceIntegerGenerator::SequenceInfo{0, 1, ITEMS_NUM},
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
        }),
    new common::SequenceIntegerGenerator(
        4,
        SEED,
        std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
            common::SequenceIntegerGenerator::SequenceInfo{0, 0, ITEMS_NUM},
            common::SequenceIntegerGenerator::SequenceInfo{0, 1, 0xffffffff},
        }),
    new common::RandomIntegerGenerator(10, 100, 4, SEED),
    new common::ConstIntegerGenerator(0, 4),
    new common::ConstIntegerGenerator(0, 4),
    new common::ConstIntegerGenerator(0, 4),
    // do_nothing_generator
};

std::vector<common::TypeWithSize> infos_history{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::INT64, 8, 0),
    // common::TypeWithSize(common::STRING, 24, 0),
};

std::vector<common::GeneratorBase *> gens_history{
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    // do_nothing_generator
};

std::vector<common::TypeWithSize> infos_order{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 4),
    common::TypeWithSize(common::INT64, 8, 0),
};

std::vector<common::GeneratorBase *> gens_order{
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
};

std::vector<common::TypeWithSize> infos_neworder{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 4),
};

std::vector<common::GeneratorBase *> gens_neworder{
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
};

std::vector<common::TypeWithSize> infos_orderline{
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::INT64, 8, 0),
    common::TypeWithSize(common::INT32, 4, 0),
    common::TypeWithSize(common::FLOAT32, 4, 0),
    common::TypeWithSize(common::INT32, 4, 4),
};

std::vector<common::GeneratorBase *> gens_orderline{
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
    do_nothing_generator,
};

using StructItem = common::StructType::StructItem;

std::vector<common::StructType::StructItem> payment_struct_items{
    StructItem("w_id", new common::IntegerType(common::INT32, true), 0),
    StructItem("d_id", new common::IntegerType(common::INT32, true), 0),
    StructItem("c_id", new common::IntegerType(common::INT32, true), 0),
    StructItem("c_d_id", new common::IntegerType(common::INT32, true), 0),
    StructItem("c_w_id", new common::IntegerType(common::INT32, true), 0),
    StructItem("h_amount", new common::FloatType(common::FLOAT32), 0),
    StructItem("h_date", new common::IntegerType(common::INT64, false), 0)};

common::StructType payment_struct_type("TxItem", std::move(payment_struct_items));

struct NewOrderReq
{
    int read;
    unsigned int key1;
    unsigned int key2;

    NewOrderReq() {}
    NewOrderReq(int read, unsigned int key1, unsigned int key2)
        : read(read), key1(key1), key2(key2) {}
};

std::vector<common::StructType::StructItem> neworder_struct_items{
    StructItem("read", new common::IntegerType(common::INT32, false), 0),
    StructItem("key1", new common::IntegerType(common::INT32, true), 0),
    StructItem("key2", new common::IntegerType(common::INT32, true), 4)};

common::StructType neworder_struct_type("TxItem", std::move(neworder_struct_items));

int txcnt, warehouse_cnt, valid_txn_bitoffset, warp_cnt, batch_size, debug;
int isno;
std::string bench_name;
std::string cc_type;
common::DB_CPU db;
common::DB_GPU *db_gpu;
common::ExecInfo *exec_info;
cudaEvent_t start1, mid1, stop1;

struct TaskInfo
{
    common::TransactionSet_CPU *txinfo;
    char *gpu_txs;
    char *gpu_gaccotxs;
    void *output;

    TaskInfo() {}
    TaskInfo(common::TransactionSet_CPU *txinfo,
             char *gpu_txs,
             char *gpu_gaccotxs,
             void *output) : txinfo(txinfo), gpu_txs(gpu_txs), gpu_gaccotxs(gpu_gaccotxs), output(output) {}
};

struct ProgramInfo
{
    common::ConcurrencyControlCPUBase *cc;
    CUdeviceptr cc_device_info;
    common::RTCProgram *program;

    ProgramInfo() {}
    ProgramInfo(common::ConcurrencyControlCPUBase *cc,
                CUdeviceptr cc_device_info,
                common::RTCProgram *program) : cc(cc), cc_device_info(cc_device_info), program(program) {}
};

TaskInfo *PaymentPre0();
TaskInfo *NewOrderPre0();

ProgramInfo *Pre(TaskInfo *task_info)
{
    std::string path("./benchmark/tpcc.cu");
    std::string include_path("./include");
    common::ConcurrencyControlCPUBase *cc = nullptr;
    common::TransactionSet_CPU *txinfo = task_info->txinfo;
    char *gpu_gacco_txs = task_info->gpu_gaccotxs;

    common::TSAllocator_CPU *ts_allocator = new common::NaiveTSAllocator_CPU();
    // common::TSAllocator_CPU *ts_allocator = new common::BatchedTSAllocator_CPU();

    if (cc_type == "tictoc")
        cc = new cc::Tictoc_CPU(&db, txinfo, batch_size);
    else if (cc_type == "silo")
        cc = new cc::Silo_CPU(&db, txinfo, batch_size);
    else if (cc_type == "to")
        cc = new cc::TO_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "slowto")
        cc = new cc::Slow_TO_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "mvcc")
        cc = new cc::MVCC_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "slowmvcc")
        cc = new cc::Slow_MVCC_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "tpl_nw")
        cc = new cc::TwoPL_CPU(&db, txinfo, false, batch_size);
    else if (cc_type == "tpl_wd")
        cc = new cc::TwoPL_CPU(&db, txinfo, true, batch_size);
    else if (cc_type == "gacco")
        cc = new cc::Gacco_CPU(&db, db_gpu, txinfo, gpu_gacco_txs, batch_size);
    else if (cc_type == "gputx")
        cc = new cc::GPUTx_CPU(&db, db_gpu, txinfo, gpu_gacco_txs, batch_size);

    size_t cc_size = cc->GetMemSize();
    // std::cout << "CC SIZE " << cc_size << " " << cc_size * 1.0f / (1024 * 1024) << "\n";

    CUdeviceptr cc_device_info = (CUdeviceptr)cc->ToGPU();
    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());

    common::RTCProgram *program = new common::RTCProgram(path, include_path, "tpcc.cu", true);
    program->AddNameExpression("payment");
    program->AddNameExpression("new_order");

    std::vector<std::string> str_opts;
    str_opts.push_back(std::string("-I ") + include_path);
    str_opts.push_back("-I /usr/include");
    str_opts.push_back("-D NVRTC_COMPILE");
    str_opts.push_back("-arch=compute_89");
    // str_opts.push_back("-lineinfo");
    //  str_opts.push_back("-G");
    str_opts.push_back("-D INDEX_0_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_1_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_2_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_3_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_4_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D THREAD_TX_SHIFT=" + std::to_string(valid_txn_bitoffset));

    exec_info->GetCompileOptions(str_opts);
    txinfo->GetCompileOptions(str_opts);
    cc->GetCompileOptions(str_opts);
    db.GetCompileOptions(str_opts);

    std::vector<const char *> opts;
    for (auto &s : str_opts)
        opts.push_back(s.c_str());

    if (!program->Compile(opts))
        exit(1);

    ProgramInfo *ret = new ProgramInfo(cc, cc_device_info, program);
    return ret;
}

void Bench(ProgramInfo *aaabbb, void **txset_info, char **gpu_txs, void **output)
{
    common::ConcurrencyControlCPUBase *cc = aaabbb->cc;
    common::RTCProgram *program = aaabbb->program;
    void **args = new void *[8];
    args[0] = &db_gpu;
    args[1] = gpu_txs;
    args[2] = txset_info;
    args[6] = &(aaabbb->cc_device_info);
    args[7] = output;
    float elapsed_time = 0;

    cudaEventRecord(start1);
    cc->Init0();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&elapsed_time, start1, stop1);
    exec_info->precompute_time = elapsed_time;

    std::vector<size_t> &batches = cc->batches;
    int batch_st = 0;
    int batch_en = 0;
    size_t block_size = warp_cnt << 5;

    for (int i = 0; i < batches.size(); i++)
    {
        cudaEventRecord(start1);
        size_t batched_txnum = batches[i]; // min(batch_size, txnum - i * batch_size);
        // printf("%d\n", batched_txnum);
        if (!batched_txnum)
            continue;
        batch_en = batch_st + batched_txnum;
        size_t thread_num = batched_txnum << valid_txn_bitoffset;
        dim3 num_blocks(thread_num / block_size + (thread_num % block_size > 0), 1, 1);
        dim3 num_threads(thread_num > block_size ? block_size : thread_num, 1, 1);
        args[3] = &i;
        args[4] = &batch_st;
        args[5] = &batch_en;
        cc->Init(i, batch_st);
        cudaEventRecord(mid1);
        program->Call(bench_name, num_blocks, num_threads, args, cc->streams[i]);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&elapsed_time, start1, mid1);
        exec_info->precompute_time += elapsed_time;
        cudaEventElapsedTime(&elapsed_time, mid1, stop1);
        exec_info->processing_time += elapsed_time;
        batch_st = batch_en;
    }
    cc->Sync();
}

int main(int argc, char **argv)
{
    common::Context context;

    sscanf(argv[1], "%d", &isno);
    sscanf(argv[2], "%d", &warehouse_cnt);
    sscanf(argv[3], "%d", &txcnt);
    cc_type = std::string(argv[4]);
    sscanf(argv[5], "%d", &valid_txn_bitoffset);
    sscanf(argv[6], "%d", &warp_cnt);
    sscanf(argv[7], "%d", &batch_size);
    sscanf(argv[8], "%d", &debug);
    // std::cin >> txnum >> warehouse_cnt;

    std::string name = isno ? "tpcc_no_wh" : "tpcc_pm_wh";
    name += std::to_string(warehouse_cnt);
    // Gen Table
    common::Table_CPU *table_warehouse = new common::Table_CPU;
    common::Table_CPU *table_district = new common::Table_CPU;
    common::Table_CPU *table_customer = new common::Table_CPU;
    common::Table_CPU *table_item = new common::Table_CPU;
    common::Table_CPU *table_stock = new common::Table_CPU;
    common::Table_CPU *table_history = new common::Table_CPU;
    common::Table_CPU *table_order = new common::Table_CPU;
    common::Table_CPU *table_neworder = new common::Table_CPU;
    common::Table_CPU *table_orderline = new common::Table_CPU;

    table_warehouse->AddColumns(infos_warehouse, gens_warehouse);
    table_warehouse->LoadData(warehouse_cnt, "./tables/warehouse_wh" + std::to_string(warehouse_cnt));
    // table_warehouse->InitData(warehouse_cnt);
    table_warehouse->LoadIndex(common::SORTED_ARRAY, 0, "./tables/warehouse_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_district->AddColumns(infos_district, gens_district);
    table_district->LoadData(warehouse_cnt * DISTRICTS_PER_W, "./tables/district_wh" + std::to_string(warehouse_cnt));
    // table_district->InitData(warehouse_cnt * DISTRICTS_PER_W);
    table_district->LoadIndex(
        common::SORTED_ARRAY,
        // std::vector<int>{0, 1},
        // std::vector<int>{1, DISTRICTS_PER_W},
        "./tables/district_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_customer->AddColumns(infos_customer, gens_customer);
    table_customer->LoadData(warehouse_cnt * DISTRICTS_PER_W * CUSTOMERS_PER_D, "./tables/customer_wh" + std::to_string(warehouse_cnt));
    // table_customer->InitData(warehouse_cnt * DISTRICTS_PER_W * CUSTOMERS_PER_D);
    table_customer->LoadIndex(
        common::SORTED_ARRAY,
        // std::vector<int>{0, 1, 2},
        // std::vector<int>{1, CUSTOMERS_PER_D, DISTRICTS_PER_W * CUSTOMERS_PER_D},
        "./tables/customer_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_item->AddColumns(infos_item, gens_item);
    table_item->LoadData(ITEMS_NUM, "./tables/item_wh" + std::to_string(warehouse_cnt));
    // table_item->InitData(ITEMS_NUM);
    table_item->LoadIndex(common::SORTED_ARRAY, 0, "./tables/item_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_stock->AddColumns(infos_stock, gens_stock);
    table_stock->LoadData(warehouse_cnt * ITEMS_NUM, "./tables/stock_wh" + std::to_string(warehouse_cnt));
    // table_stock->InitData(warehouse_cnt * ITEMS_NUM);
    table_stock->LoadIndex(common::SORTED_ARRAY,
                           //    std::vector<int>{0, 1},
                           //    std::vector<int>{1, ITEMS_NUM},
                           "./tables/stock_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_history->AddColumns(infos_history, gens_history);
    table_history->LoadData(txcnt, "./tables/history_wh" + std::to_string(warehouse_cnt));
    // table_history->InitData(txcnt);

    table_order->AddColumns(infos_order, gens_order);
    table_order->LoadData(txcnt, "./tables/order_wh" + std::to_string(warehouse_cnt));
    // table_order->InitData(txcnt);

    table_neworder->AddColumns(infos_neworder, gens_neworder);
    table_neworder->LoadData(txcnt, "./tables/neworder_wh" + std::to_string(warehouse_cnt));
    // table_neworder->InitData(txcnt);

    table_orderline->AddColumns(infos_orderline, gens_orderline);
    table_orderline->LoadData(txcnt * 15, "./tables/orderline_wh" + std::to_string(warehouse_cnt));
    // table_orderline->InitData(txcnt * 15);

    db.AddTable(table_warehouse);
    db.AddTable(table_district);
    db.AddTable(table_customer);
    db.AddTable(table_item);
    db.AddTable(table_stock);
    db.AddTable(table_history);
    db.AddTable(table_order);
    db.AddTable(table_neworder);
    db.AddTable(table_orderline);

    db.Init();
    db_gpu = db.ToGPU();
    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());

    size_t db_size = db.GetMemSize();
    std::cout << "TRY " << argv[4] << "," << valid_txn_bitoffset << "," << warp_cnt << "," << batch_size << "\n";
    // std::cout << "GPUDB SIZE " << db_size << " " << db_size * 1.0f / (1024.0 * 1024.0) << "\n";

    TaskInfo *task_info = nullptr;

    if (isno)
    {
        bench_name = "new_order";
        task_info = NewOrderPre0();
    }
    else
    {
        bench_name = "payment";
        task_info = PaymentPre0();
    }

    exec_info = new common::ExecInfo(name, task_info->txinfo, txcnt, argv[4],
                                     valid_txn_bitoffset, warp_cnt, batch_size, debug);

    ProgramInfo *program_info = Pre(task_info);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&mid1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start);
    void *txset_info = ((common::DynamicTransactionSet_CPU *)(task_info->txinfo))->ToGPU();
    Bench(program_info, &txset_info, &(task_info->gpu_txs), &(task_info->output));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(exec_info->tot_time), start, stop);
    exec_info->Analyse();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

TaskInfo *PaymentPre0()
{
    common::StaticTransactionSet_CPU *txinfo = new common::StaticTransactionSet_CPU(txcnt, &payment_struct_type, &db);

    gputp_operator::Operator *op1 = new gputp_operator::StructMemberAccessOperator(
        new gputp_operator::VariableOperator("tx"),
        "w_id");
    op1->type = new common::IntegerType(common::INT32, true);

    gputp_operator::Operator *op2 = new gputp_operator::BinOperator(
        gputp_operator::ADD,
        new gputp_operator::BinOperator(
            gputp_operator::MUL,
            new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), DISTRICTS_PER_W),
            new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "w_id")),
        new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "d_id"));

    op2->type = new common::IntegerType(common::INT32, true);

    gputp_operator::Operator *op3 = new gputp_operator::BinOperator(
        gputp_operator::ADD,
        new gputp_operator::BinOperator(
            gputp_operator::ADD,
            new gputp_operator::BinOperator(
                gputp_operator::MUL,
                new gputp_operator::BinOperator(
                    gputp_operator::MUL,
                    new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "c_w_id"),
                    new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), DISTRICTS_PER_W)),
                new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), CUSTOMERS_PER_D)),
            new gputp_operator::BinOperator(
                gputp_operator::MUL,
                new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), CUSTOMERS_PER_D),
                new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "c_d_id"))),
        new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "c_id"));

    op3->type = new common::IntegerType(common::INT32, true);
    txinfo->AddAccessor("warehouser", 0, 0, op1, false);
    txinfo->AddAccessor("warehousew", 0, 0, op1, true);
    txinfo->AddAccessor("districtr", 1, 0, op2, false);
    txinfo->AddAccessor("districtw", 1, 0, op2, true);
    txinfo->AddAccessor("customerr", 2, 0, op3, false);
    txinfo->AddAccessor("customerw", 2, 0, op3, true);

    PaymentTx *txdata = new PaymentTx[txcnt];
    std::string file_name = std::string("./dataset/tpcc_pm_wh") + std::to_string(warehouse_cnt) + ".txs";
    FILE *file = fopen(file_name.c_str(), "rb");
    fread(txdata, sizeof(PaymentTx), txcnt, file);
    fclose(file);

    size_t tx_size = txcnt * sizeof(PaymentTx);
    char *gpu_txs;
    cudaMalloc(&gpu_txs, tx_size);
    cudaMemcpy(gpu_txs, txdata, tx_size, cudaMemcpyHostToDevice);

    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
    delete[] txdata;

    printf("---------------------------------- Payment\n");

    void *output;
    cudaMalloc(&output, sizeof(PaymentOutput) * txcnt);

    return new TaskInfo(txinfo, gpu_txs, gpu_txs, output);
}

TaskInfo *NewOrderPre0()
{
    std::vector<int> rcnts(txcnt);
    std::vector<int> wcnts(txcnt);
    std::vector<int> detailed_opcnts(txcnt * 2);

    NewOrderTx *txdata = new NewOrderTx[txcnt];
    std::string file_name = std::string("./dataset/tpcc_no_wh") + std::to_string(warehouse_cnt) + ".txs";
    FILE *file = fopen(file_name.c_str(), "rb");
    fread(txdata, sizeof(NewOrderTx), txcnt, file);
    fclose(file);

    char *gpu_txs;
    size_t tx_size = txcnt * sizeof(NewOrderTx);
    cudaMalloc(&gpu_txs, tx_size);
    cudaMemcpy(gpu_txs, txdata, tx_size, cudaMemcpyHostToDevice);

    std::vector<NewOrderReq> gacco_txdata;
    for (int i = 0; i < txcnt; i++)
    {
        gacco_txdata.push_back(NewOrderReq(1, txdata[i].w_id, txdata[i].d_id));
        gacco_txdata.push_back(NewOrderReq(0, txdata[i].w_id, txdata[i].d_id));
        for (int j = 0; j < txdata[i].ol_cnt; j++)
        {
            gacco_txdata.push_back(NewOrderReq(1, txdata[i].supply_w_ids[j], txdata[i].i_ids[j]));
            gacco_txdata.push_back(NewOrderReq(0, txdata[i].supply_w_ids[j], txdata[i].i_ids[j]));
        }
        rcnts[i] = wcnts[i] = txdata[i].ol_cnt + 1;
        detailed_opcnts[i * 2] = 2;
        detailed_opcnts[i * 2 + 1] = txdata[i].ol_cnt * 2;
    }

    char *gpu_gaccotxs;
    cudaMalloc(&gpu_gaccotxs, sizeof(NewOrderReq) * gacco_txdata.size());
    cudaMemcpy(gpu_gaccotxs, gacco_txdata.data(), sizeof(NewOrderReq) * gacco_txdata.size(), cudaMemcpyHostToDevice);

    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
    delete[] txdata;

    printf("-------------------------- NEW ORDER\n");

    common::DynamicTransactionSet_CPU *txinfo = new common::DynamicTransactionSet_CPU(
        txcnt,
        2,
        &neworder_struct_type,
        std::move(rcnts),
        std::move(wcnts),
        std::move(detailed_opcnts),
        &db);

    gputp_operator::Operator *op1 = new gputp_operator::BinOperator(
        gputp_operator::ADD,
        new gputp_operator::BinOperator(
            gputp_operator::MUL,
            new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), DISTRICTS_PER_W),
            new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key1")),
        new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key2"));
    op1->type = new common::IntegerType(common::INT32, true);

    gputp_operator::Operator *op2 = new gputp_operator::BinOperator(
        gputp_operator::ADD,
        new gputp_operator::BinOperator(
            gputp_operator::MUL,
            new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), ITEMS_NUM),
            new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key1")),
        new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key2"));
    op2->type = new common::IntegerType(common::INT32, true);

    txinfo->AddAccessor("district", 1, 0, op1, false);
    txinfo->AddAccessor("stock", 4, 0, op2, false);

    void *output;
    cudaMalloc(&output, sizeof(NewOrderOutput) * txcnt);

    return new TaskInfo(txinfo, gpu_txs, gpu_gaccotxs, output);
}