#include <ctime>
#include <set>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <ycsb.cuh>

using namespace bench_ycsb;

const int SEED = 141919810;
const int GENERATE_THREAD_CNT = 256;

void generateTx(YCSBTx &cur_tx);

void generateTxBatched(int tid, YCSBTx *txs, int tx_cnt)
{
    int cnt_per_thread = tx_cnt / GENERATE_THREAD_CNT;
    int st = cnt_per_thread * tid;
    if (tid == GENERATE_THREAD_CNT - 1)
        cnt_per_thread += tx_cnt % GENERATE_THREAD_CNT;
    int en = st + cnt_per_thread;

    for (int i = st; i < min(en, tx_cnt); i++)
        generateTx(txs[i]);
}

int rownum = 10485760;
int txcnt = 4194304;
float g_zipf_theta = 0.6f;
float g_read_perc = 0.9f;

int main(int argc, char **argv)
{
    srand(SEED);
    sscanf(argv[1], "%d", &rownum);
    sscanf(argv[2], "%d", &txcnt);
    sscanf(argv[3], "%f", &g_zipf_theta);
    sscanf(argv[4], "%f", &g_read_perc);

    YCSBTx *txdata = new YCSBTx[txcnt];

    std::vector<std::thread> threads;
    for (int i = 0; i < GENERATE_THREAD_CNT; i++)
        threads.push_back(
            std::thread(generateTxBatched, i, txdata, txcnt));

    for (int i = 0; i < GENERATE_THREAD_CNT; i++)
        threads[i].join();
    std::cout << "GEN OK\n";

    FILE *file = fopen(argv[5], "wb");
    fwrite(txdata, sizeof(YCSBTx), txcnt, file);
    fclose(file);

    return 0;
}

double uniform_double(float x, float y)
{
    return x + 1.0 * rand() / RAND_MAX * (y - x);
}

template <typename T>
int uniform_int(T x, T y)
{
    return rand() % (y - x) + x;
}
// const float g_zipf_theta = 0.6;
// The following algorithm comes from the paper:
// Quickly generating billion-record synthetic databases
// However, it seems there is a small bug.
// The original paper says zeta(theta, 2.0). But I guess it should be
// zeta(2.0, theta).
double zeta(unsigned int n, double theta)
{
    double sum = 0;
    for (unsigned int i = 1; i <= n; i++)
        sum += pow(1.0 / i, theta);
    return sum;
}

unsigned int zipf(unsigned int n, double theta)
{
    static double denom = zeta(rownum - 1, g_zipf_theta);
    static double zeta_2_theta = zeta(2, g_zipf_theta);

    // assert(this->the_n == n);
    // assert(theta == g_zipf_theta);
    double alpha = 1 / (1 - theta);
    double zetan = denom;
    double eta = (1 - pow(2.0 / n, 1 - theta)) /
                 (1 - zeta_2_theta / zetan);
    double u = uniform_double(0, 1);
    double uz = u * zetan;
    if (uz < 1)
        return 1;
    if (uz < 1 + pow(0.5, theta))
        return 2;
    return 1 + (unsigned int)(n * pow(eta * u - eta + 1, alpha));
}

void generateTx(YCSBTx &cur_tx)
{
    std::set<unsigned int> all_keys;

    int rid = 0;
    for (int tmp = 0; tmp < MAX_REQUEST_CNT; tmp++)
    {
        double r = uniform_double(0, 1);
        YCSBReq &req = cur_tx.requests[rid];
        if (r < g_read_perc)
            req.read = 1;
        else
            req.read = 0;

        // the request will access part_id.
        unsigned int table_size = rownum;
        unsigned int primary_key = zipf(table_size - 1, g_zipf_theta);
        // assert(primary_key < table_size);
        req.key = primary_key;

        // long long rint64 = uniform_int<long long>(0, 1L << 31);
        // req.value = rint64 % (1 << 8);
        //  Make sure a single row is not accessed twice
        //  if (req.rtype == RD || req->rtype == WR)
        //  {
        if (all_keys.find(req.key) == all_keys.end())
            all_keys.insert(req.key);
        else
            continue;

        rid++;
    }
    cur_tx.request_cnt = rid;

    YCSBReq *requests = cur_tx.requests;

    // Sort the requests in key order.
    // if (g_key_order)
    // {
    for (int i = cur_tx.request_cnt - 1; i > 0; i--)
        for (int j = 0; j < i; j++)
            if (requests[j].key > requests[j + 1].key)
            {
                YCSBReq tmp = requests[j];
                requests[j] = requests[j + 1];
                requests[j + 1] = tmp;
            }
    // }
}