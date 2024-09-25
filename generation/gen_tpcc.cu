#include <vector>
#include <set>
#include <algorithm>
#include <ctime>
#include <thread>
#include <iostream>
#include <mutex>
#include <string>
#include <stdio.h>
#include <tpcc.cuh>

using namespace tpcc;

const int SEED = 141919810;
const int GENERATE_THREAD_CNT = 256;

int warehouse_max = 64;
int txcnt = 4194304;
int genno = 0;
float no_remote_ratio = 0.1f;

void generatePaymentTx(PaymentTx &cur_tx);

void generatePaymentTxBatched(int tid, PaymentTx *txs, int tx_cnt)
{
    int cnt_per_thread = tx_cnt / GENERATE_THREAD_CNT;
    int st = cnt_per_thread * tid;
    if (tid == GENERATE_THREAD_CNT - 1)
        cnt_per_thread += tx_cnt % GENERATE_THREAD_CNT;
    int en = st + cnt_per_thread;

    for (int i = st; i < min(en, tx_cnt); i++)
        generatePaymentTx(txs[i]);
}

void generateNewOrderTx(NewOrderTx &cur_tx);

void generateNewOrderTxBatched(int tid, NewOrderTx *txs, int tx_cnt)
{
    int cnt_per_thread = tx_cnt / GENERATE_THREAD_CNT;
    int st = cnt_per_thread * tid;
    if (tid == GENERATE_THREAD_CNT - 1)
        cnt_per_thread += tx_cnt % GENERATE_THREAD_CNT;
    int en = st + cnt_per_thread;

    for (int i = st; i < en; i++)
        generateNewOrderTx(txs[i]);
}

int main(int argc, char **argv)
{
    srand(SEED);
    sscanf(argv[1], "%d", &warehouse_max);
    sscanf(argv[2], "%d", &txcnt);
    sscanf(argv[3], "%d", &genno);
    sscanf(argv[4], "%f", &no_remote_ratio);

    if (genno)
    {
        NewOrderTx *txdata = new NewOrderTx[txcnt];

        std::vector<std::thread> threads;
        for (int i = 0; i < GENERATE_THREAD_CNT; i++)
            threads.push_back(
                std::thread(generateNewOrderTxBatched, i, txdata, txcnt));

        for (int i = 0; i < GENERATE_THREAD_CNT; i++)
            threads[i].join();
        std::cout << "GEN OK\n";

        std::string file_name = std::string("./dataset/tpcc_no_wh") + std::to_string(warehouse_max) + ".txs";
        FILE *file = fopen(file_name.c_str(), "wb");
        fwrite(txdata, sizeof(NewOrderTx), txcnt, file);
        fclose(file);
    }
    else
    {
        PaymentTx *txdata = new PaymentTx[txcnt];

        std::vector<std::thread> threads;
        for (int i = 0; i < GENERATE_THREAD_CNT; i++)
            threads.push_back(
                std::thread(generatePaymentTxBatched, i, txdata, txcnt));

        for (int i = 0; i < GENERATE_THREAD_CNT; i++)
            threads[i].join();
        std::cout << "GEN OK\n";

        std::string file_name = std::string("./dataset/tpcc_pm_wh") + std::to_string(warehouse_max) + ".txs";
        FILE *file = fopen(file_name.c_str(), "wb");
        fwrite(txdata, sizeof(PaymentTx), txcnt, file);
        fclose(file);
    }

    return 0;
}

int uniform_int(int x, int y)
{
    return rand() % (y - x) + x;
}

float uniform_float(float x, float y)
{
    return x + 1.0 * rand() / RAND_MAX * (y - x);
}

static unsigned int NURand(unsigned int A, unsigned int x, unsigned int y)
{
    unsigned int random_0_A = uniform_int(0, A);
    unsigned int random_xy_C = uniform_int(x, y) + 42;
    return ((random_0_A | random_xy_C) % (y - x + 1)) + x;
}

void generatePaymentTx(PaymentTx &cur_tx)
{
    cur_tx.c_w_id = uniform_int(0, warehouse_max);
    cur_tx.c_d_id = uniform_int(0, 10);
    cur_tx.c_id = NURand(1023, 0, 2999);
    cur_tx.w_id = cur_tx.c_w_id;
    cur_tx.d_id = cur_tx.c_d_id;
    if (uniform_int(0, 100) > 85)
    {
        // customer is paying in another warehouse and district
        while (warehouse_max > 1 && cur_tx.w_id == cur_tx.c_w_id)
            cur_tx.w_id = uniform_int(0, warehouse_max);
        cur_tx.d_id = uniform_int(0, 10 - 1);
        if (cur_tx.d_id >= cur_tx.c_d_id)
            cur_tx.d_id += 1;
    }

    cur_tx.h_amount = uniform_float(1.0, 5000.0);
    cur_tx.h_date = std::time(nullptr);
}

void generateNewOrderTx(NewOrderTx &cur_tx)
{
    cur_tx.w_id = uniform_int(0, warehouse_max);
    cur_tx.d_id = uniform_int(0, 10);
    cur_tx.c_id = NURand(1023, 0, 2999);

    cur_tx.ol_cnt = uniform_int(5, 15 + 1);
    cur_tx.entry_d = std::time(nullptr);
    cur_tx.all_local = 1;
    // unsigned int32_t rbk        = util::uniform_int(1,100,true);

    std::set<int> i_ids;
    std::vector<StockInfo> stocks(cur_tx.ol_cnt);

    for (int ol_idx = 0; ol_idx < cur_tx.ol_cnt; ++ol_idx)
    {
        // assume no aborts in this tx
        // if(ol_idx == cur_tx.ol_cnt-1 && rbk == 1){
        //     // last item and rollback is expected
        //     cur_tx.i_ids[ol_idx] = unsigned int32_MAX;
        //     break;
        // }
        int i_id = NURand(8191, 0, 99999);
        while (i_ids.find(i_id) != i_ids.end())
            i_id = NURand(8191, 0, 99999);
        i_ids.insert(i_id);

        StockInfo &stock = stocks[ol_idx];
        stock.i_id = i_id; // NURand(8191, 0, 99999);
        stock.quantity = uniform_int(1, 10 + 1);
        stock.supply_w_id = cur_tx.w_id;
        if (uniform_float(0.0, 1.0) <= no_remote_ratio && warehouse_max > 1)
        {
            stock.supply_w_id = uniform_int(0, warehouse_max - 1);
            if (stock.supply_w_id >= cur_tx.w_id)
                stock.supply_w_id += 1;
            cur_tx.all_local = 0;
        }
    }
    std::sort(stocks.begin(), stocks.end());
    for (int i = 0; i < cur_tx.ol_cnt; i++)
    {
        StockInfo &stock = stocks[i];
        cur_tx.i_ids[i] = stock.i_id;
        cur_tx.supply_w_ids[i] = stock.supply_w_id;
        cur_tx.quantities[i] = stock.quantity;
    }
}