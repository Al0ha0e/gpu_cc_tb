#ifndef YCSB_H
#define YCSB_H

namespace bench_ycsb
{
    struct __align__(8) Item
    {
        unsigned int id;
        char f0[10];
        char f1[10];
        char f2[10];
        char f3[10];
        char f4[10];
        char f5[10];
        char f6[10];
        char f7[10];
        char f8[10];
        char f9[10];
    };

    struct __align__(8) YCSBReq
    {
        int read;
        unsigned int key;
    };

#define MAX_REQUEST_CNT 16

    struct __align__(8) YCSBTx
    {
        size_t request_cnt;
        YCSBReq requests[MAX_REQUEST_CNT];
    };
}

#endif