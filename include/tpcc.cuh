#ifndef TPCC_H
#define TPCC_H

#define DISTRICTS_PER_W 10
#define CUSTOMERS_PER_D 3000
#define ITEMS_NUM 100000

namespace tpcc
{
    struct __align__(8) Warehouse
    {
        unsigned int id;
        float tax;
        float ytd;
        // char name[10];
        // char street1[20];
        // char street2[20];
        // char city[20];
        // char state[2];
        // char zip[9];
        // char padding[3];
    };

    struct __align__(8) District
    {
        unsigned int id;
        unsigned int w_id;
        float tax;
        float ytd;
        unsigned int next_o_id;
        // char name[10];
        // char street1[20];
        // char street2[20];
        // char city[20];
        // char state[2];
        // char zip[9];
        // char padding[3];
    };

    struct __align__(8) Customer
    {
        unsigned int id;
        unsigned int d_id;
        unsigned int w_id;
        unsigned int since;
        float credit_lim;
        float discount;
        float balance;
        float ytd_payment;
        unsigned int payment_cnt;
        unsigned int delivery_cnt;
        // char first[16];
        // char middle[2];
        // char last[16];
        // char street1[20];
        // char street2[20];
        // char city[20];
        // char state[2];
        // char zip[9];
        // char phone[16];
        // char credit[2];
        // char data[500];
        // char padding[1];
    };

    struct __align__(8) Item
    {
        unsigned int id;
        unsigned int im_id;
        float price;
        // char name[24];
        // char data[50];
        // char padding[2];
    };

    struct __align__(8) Stock
    {
        unsigned int i_id;
        unsigned int w_id;
        unsigned int quantity;
        unsigned int ytd;
        unsigned int order_cnt;
        unsigned int remote_cnt;
        // char dist00[24];
        // char dist01[24];
        // char dist02[24];
        // char dist03[24];
        // char dist04[24];
        // char dist05[24];
        // char dist06[24];
        // char dist07[24];
        // char dist08[24];
        // char dist09[24];
        // char data[50];
    };

    struct __align__(8) History
    {
        unsigned int w_id;
        unsigned int d_id;
        unsigned int c_id;
        unsigned int c_d_id;
        unsigned int c_w_id;
        float amount;
        unsigned long long date;
        // char data[24];
    };

    struct __align__(8) Order
    {
        unsigned int id;
        unsigned int w_id;
        unsigned int d_id;
        unsigned int c_id;
        unsigned int carrier_id;
        unsigned int ol_cnt;
        unsigned int all_local;
        unsigned long long entry_d;

        __host__ __device__ Order() {}
        __host__ __device__ Order(unsigned int id, unsigned int w_id, unsigned int d_id,
                                  unsigned int c_id, unsigned int carrier_id, unsigned int ol_cnt,
                                  unsigned int all_local, unsigned long long entry_d)
            : id(id), w_id(w_id), d_id(d_id), c_id(c_id), carrier_id(carrier_id), ol_cnt(ol_cnt), all_local(all_local), entry_d(entry_d) {}
    };

    struct __align__(8) NewOrder
    {
        unsigned int o_id;
        unsigned int w_id;
        unsigned int d_id;

        __host__ __device__ NewOrder() {}
        __host__ __device__ NewOrder(unsigned int o_id, unsigned int w_id, unsigned int d_id)
            : o_id(o_id), w_id(w_id), d_id(d_id) {}
    };

    struct __align__(8) OrderLine
    {
        unsigned int o_id;
        unsigned int d_id;
        unsigned int w_id;
        unsigned int number;
        unsigned int i_id;
        unsigned int supply_w_id;
        unsigned long long delivery_d;
        unsigned int quantity;
        float amount;
        int dist_info;
    };

    struct __align__(8) PaymentTx
    {
        // input data
        // payment warehouse and district
        unsigned int w_id;
        unsigned int d_id;
        // customer information
        unsigned int c_id;
        unsigned int c_d_id;
        unsigned int c_w_id;
        float h_amount;
        unsigned long long h_date;
    };

    struct __align__(8) PaymentOutput
    {
        unsigned int w_id;
        unsigned int d_id;
        unsigned int c_id;
        unsigned int c_d_id;
        unsigned int c_w_id;
        float h_amount;
        unsigned long long h_date;
    };

    struct __align__(8) PaymentLocalStruct
    {
        Warehouse warehouse;
        District district;
        Customer customer;
    };

    struct __align__(8) NewOrderTx
    {
        // General order information
        unsigned int w_id;
        unsigned int d_id;
        unsigned int c_id;
        unsigned int carrier_id;
        unsigned int ol_cnt;
        unsigned int all_local;
        unsigned long long entry_d;
        // OrderLine items information
        unsigned int i_ids[15];
        unsigned int supply_w_ids[15];
        unsigned int quantities[15];
    };

    struct __align__(8) OrderlineOutput
    {
        unsigned int s_quantity;
        float price;
        float amount;
        char i_name[25];
        char brand_generic[2];
    };

    struct __align__(8) NewOrderOutput
    {
        float total_amount;
        unsigned int o_id;
        unsigned int w_id;
        unsigned int d_id;
        unsigned int ol_cnt;
        OrderlineOutput ol_out[15];
    };

#ifndef NVRTC_COMPILE

    struct StockInfo
    {
        unsigned int i_id;
        unsigned int supply_w_id;
        unsigned int quantity;

        bool operator<(const StockInfo &ano) const
        {
            if (supply_w_id == ano.supply_w_id)
                return i_id < ano.i_id;
            return supply_w_id < ano.supply_w_id;
        }
    };

#endif

}

#endif