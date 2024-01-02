#include <db.cuh>
#include <table.cuh>
#include <runtime.cuh>
#include <env.cuh>
#include <tictoc.cuh>
#include <gacco.cuh>
#include <gputx.cuh>
#include <silo.cuh>
#include <to.cuh>
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

int txcnt, warehouse_cnt;
int isno;
std::string bench_name;
common::DB_CPU db;
common::DB_GPU *db_gpu;
common::ExecInfo *exec_info;
cudaEvent_t start1, mid1, stop1;

int main(int argc, char **argv)
{
    sscanf(argv[1], "%d", &isno);
    sscanf(argv[2], "%d", &warehouse_cnt);
    sscanf(argv[3], "%d", &txcnt);
    // std::cin >> txnum >> warehouse_cnt;

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
    table_warehouse->InitData(warehouse_cnt);
    table_warehouse->SaveData("./tables/warehouse_wh" + std::to_string(warehouse_cnt));
    table_warehouse->SaveIndex(common::SORTED_ARRAY, 0, "./tables/warehouse_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_district->AddColumns(infos_district, gens_district);
    table_district->InitData(warehouse_cnt * DISTRICTS_PER_W);
    table_district->SaveData("./tables/district_wh" + std::to_string(warehouse_cnt));
    table_district->SaveIndex(
        common::SORTED_ARRAY,
        std::vector<int>{0, 1},
        std::vector<int>{1, DISTRICTS_PER_W},
        "./tables/district_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_customer->AddColumns(infos_customer, gens_customer);
    table_customer->InitData(warehouse_cnt * DISTRICTS_PER_W * CUSTOMERS_PER_D);
    table_customer->SaveData("./tables/customer_wh" + std::to_string(warehouse_cnt));
    table_customer->SaveIndex(
        common::SORTED_ARRAY,
        std::vector<int>{0, 1, 2},
        std::vector<int>{1, CUSTOMERS_PER_D, DISTRICTS_PER_W * CUSTOMERS_PER_D},
        "./tables/customer_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_item->AddColumns(infos_item, gens_item);
    table_item->InitData(ITEMS_NUM);
    table_item->SaveData("./tables/item_wh" + std::to_string(warehouse_cnt));
    table_item->SaveIndex(common::SORTED_ARRAY, 0, "./tables/item_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_stock->AddColumns(infos_stock, gens_stock);
    table_stock->InitData(warehouse_cnt * ITEMS_NUM);
    table_stock->SaveData("./tables/stock_wh" + std::to_string(warehouse_cnt));
    table_stock->SaveIndex(common::SORTED_ARRAY,
                           std::vector<int>{0, 1},
                           std::vector<int>{1, ITEMS_NUM}, "./tables/stock_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_history->AddColumns(infos_history, gens_history);
    table_history->InitData(txcnt);
    table_history->SaveData("./tables/history_wh" + std::to_string(warehouse_cnt));

    table_order->AddColumns(infos_order, gens_order);
    table_order->InitData(txcnt);
    table_order->SaveData("./tables/order_wh" + std::to_string(warehouse_cnt));

    table_neworder->AddColumns(infos_neworder, gens_neworder);
    table_neworder->InitData(txcnt);
    table_neworder->SaveData("./tables/neworder_wh" + std::to_string(warehouse_cnt));

    table_orderline->AddColumns(infos_orderline, gens_orderline);
    table_orderline->InitData(txcnt * 15);
    table_orderline->SaveData("./tables/orderline_wh" + std::to_string(warehouse_cnt));

    return 0;
}