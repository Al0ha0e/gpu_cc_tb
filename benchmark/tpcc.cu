#include <db.cuh>
#include <transaction.cuh>
#include <runtime.cuh>
#include <cc.cuh>
#include <tictoc.cuh>
#include <gacco.cuh>
#include <gputx.cuh>
#include <silo.cuh>
#include <to.cuh>
#include <slow_to.cuh>
#include <slow_mvcc.cuh>
#include <twopl.cuh>
#include <mvcc.cuh>

#include <tpcc.cuh>

using namespace tpcc;

TABLE_ST_ARR
// TABLE_MEM_ST_ARR

enum
{
    TABLE_WAREHOUSE,
    TABLE_DISTRICT,
    TABLE_CUSTOMER,
    TABLE_ITEM,
    TABLE_STOCK,
    TABLE_HISTORY,
    TABLE_ORDER,
    TABLE_NEWORDER,
    TABLE_ORDERLINE
};

__launch_bounds__(1024)
    __global__ void payment(
        common::DB_GPU *db,
        char *txs,
        void *txset_info,
        int batch_id,
        int batch_st,
        int batch_en,
        void *info,
        char *output)
{
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id & ((1 << THREAD_TX_SHIFT) - 1))
        return;

    size_t tid = batch_st + (thread_id >> THREAD_TX_SHIFT);
    if (tid >= batch_en)
        return;

    unsigned long long tx_start_time = clock64();

    PaymentTx &tx = ((PaymentTx *)txs)[tid];

    unsigned int warehouse_id = tx.w_id;
    unsigned int district_id = tx.w_id * DISTRICTS_PER_W + tx.d_id;
    unsigned int customer_id = tx.c_w_id * DISTRICTS_PER_W * CUSTOMERS_PER_D + tx.c_d_id * CUSTOMERS_PER_D + tx.c_id;

    common::Table_GPU table_warehouse = db->tables[TABLE_WAREHOUSE];
    common::Table_GPU table_district = db->tables[TABLE_DISTRICT];
    common::Table_GPU table_customer = db->tables[TABLE_CUSTOMER];

    INDEX_0_TYPE<unsigned int> *index_warehouse = (INDEX_0_TYPE<unsigned int> *)table_warehouse.indices[0];
    INDEX_1_TYPE<unsigned int> *index_district = (INDEX_1_TYPE<unsigned int> *)table_district.indices[0];
    INDEX_2_TYPE<unsigned int> *index_customer = (INDEX_2_TYPE<unsigned int> *)table_customer.indices[0];

    unsigned long long index_start_time = clock64();

    size_t warehouse_idx = index_warehouse->Find(warehouse_id);
    size_t district_idx = index_district->Find(district_id);
    size_t customer_idx = index_customer->Find(customer_id);

    unsigned long long index_duration = clock64() - index_start_time;

    CC_TYPE cc(txset_info, info, tid);

    Warehouse warehouse;
    District district;
    Customer customer;

    do
    {
        if (!cc.TxStart(info))
            continue;

        // // warehouse
        // warehouses[warehouse_idx].ytd += tx.h_amount;
        void *warehousep = (void *)(table_warehouse.data + warehouse_idx * sizeof(Warehouse));

        if (!cc.ReadForUpdate(
                table_st_arr[TABLE_WAREHOUSE] + warehouse_idx,
                0,
                warehousep,
                // warehousep,
                (void *)&warehouse,
                sizeof(Warehouse)))
            continue;

        warehouse.ytd += tx.h_amount;

        if (!cc.Write(table_st_arr[TABLE_WAREHOUSE] + warehouse_idx,
                      0,
                      (void *)&warehouse,
                      // warehousep,
                      warehousep,
                      sizeof(Warehouse)))
            continue;

        // // district
        // districts[district_idx].ytd += tx.h_amount;
        void *districtp = (void *)(table_district.data + district_idx * sizeof(District));

        if (!cc.ReadForUpdate(
                table_st_arr[TABLE_DISTRICT] + district_idx,
                1,
                districtp,
                // districtp,
                (void *)&district,
                sizeof(District)))
            continue;

        district.ytd += tx.h_amount;

        if (!cc.Write(
                table_st_arr[TABLE_DISTRICT] + district_idx,
                1,
                (void *)&district,
                // districtp,
                districtp,
                sizeof(District)))
            continue;

        // // customer
        // customers[customer_idx].balance -= tx.h_amount;
        // customers[customer_idx].ytd_payment += tx.h_amount;
        // customers[customer_idx].payment_cnt += 1;

        void *customerp = (void *)(table_customer.data + customer_idx * sizeof(Customer));

        if (!cc.ReadForUpdate(
                table_st_arr[TABLE_CUSTOMER] + customer_idx,
                2,
                customerp,
                // customerp,
                (void *)&customer,
                sizeof(Customer)))
            continue;

        customer.balance -= tx.h_amount;
        customer.ytd_payment += tx.h_amount;
        customer.payment_cnt += 1;

        if (!cc.Write(
                table_st_arr[TABLE_CUSTOMER] + customer_idx,
                2,
                (void *)&customer,
                // customerp,
                customerp,
                sizeof(Customer)))
            continue;

        if (!cc.TxEnd(NULL))
            continue;

        // history[tx_id] = data_row::History{tx.w_id, tx.d_id, tx.c_id, tx.c_d_id,
        //                                    tx.c_w_id, tx.h_amount, tx.h_date};
        History *historyp = ((History *)(db->tables[TABLE_HISTORY].data)) + tid;
        memcpy(historyp, &tx, sizeof(History));

        // tx_output[tx_id] = data_row::PaymentOutput{tx.w_id, tx.d_id, tx.c_id, tx.c_d_id,
        //                                            tx.c_w_id, tx.h_amount, tx.h_date};
        PaymentOutput *outp = ((PaymentOutput *)output) + tid;
        memcpy(outp, &tx, sizeof(PaymentOutput));

        break;
    } while (true);

    atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->tot_duration), clock64() - tx_start_time);
    atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->index_duration), index_duration);

    cc.Finalize();
}

__device__ unsigned int order_line_idx = 0;

__launch_bounds__(1024)
    __global__ void new_order(
        common::DB_GPU *db,
        void *txs,
        void *txset_info,
        int batch_id,
        int batch_st,
        int batch_en,
        void *info,
        void *output)
{
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id & ((1 << THREAD_TX_SHIFT) - 1))
        return;

    size_t tid = batch_st + (thread_id >> THREAD_TX_SHIFT);
    if (tid >= batch_en)
        return;
    unsigned long long tx_start_time = clock64();

    NewOrderTx &tx = ((NewOrderTx *)txs)[tid];

    unsigned int warehouse_id = tx.w_id;
    unsigned int district_id = tx.w_id * DISTRICTS_PER_W + tx.d_id;
    unsigned int customer_id = tx.w_id * DISTRICTS_PER_W * CUSTOMERS_PER_D + tx.d_id * CUSTOMERS_PER_D + tx.c_id;

    common::Table_GPU table_warehouse = db->tables[TABLE_WAREHOUSE];
    common::Table_GPU table_district = db->tables[TABLE_DISTRICT];
    common::Table_GPU table_customer = db->tables[TABLE_CUSTOMER];
    common::Table_GPU table_item = db->tables[TABLE_ITEM];
    common::Table_GPU table_stock = db->tables[TABLE_STOCK];
    common::Table_GPU table_order = db->tables[TABLE_ORDER];
    common::Table_GPU table_neworder = db->tables[TABLE_NEWORDER];
    common::Table_GPU table_orderline = db->tables[TABLE_ORDERLINE];

    INDEX_0_TYPE<unsigned int> *index_warehouse = (INDEX_0_TYPE<unsigned int> *)table_warehouse.indices[0];
    INDEX_1_TYPE<unsigned int> *index_district = (INDEX_1_TYPE<unsigned int> *)table_district.indices[0];
    INDEX_2_TYPE<unsigned int> *index_customer = (INDEX_2_TYPE<unsigned int> *)table_customer.indices[0];
    INDEX_3_TYPE<unsigned int> *index_item = (INDEX_3_TYPE<unsigned int> *)table_item.indices[0];
    INDEX_4_TYPE<unsigned int> *index_stock = (INDEX_4_TYPE<unsigned int> *)table_stock.indices[0];

    unsigned long long index_start_time = clock64();
    size_t warehouse_idx = index_warehouse->Find(warehouse_id);
    size_t district_idx = index_district->Find(district_id);
    size_t customer_idx = index_customer->Find(customer_id);
    unsigned long long index_duration = clock64() - index_start_time;

    float wh_tax = ((Warehouse *)table_warehouse.data)[warehouse_idx].tax;
    float district_tax = ((District *)table_district.data)[district_idx].tax;
    float customer_discount = ((Customer *)table_customer.data)[customer_idx].discount;

    CC_TYPE cc(txset_info, info, tid);

    Stock stock;
    NewOrderOutput *outp = ((NewOrderOutput *)output) + tid;
    float sum_ol_amount;
    unsigned int cur_o_id;
    unsigned int ol_idx_insert_start = atomicAdd(&order_line_idx, tx.ol_cnt);
    unsigned long long partial_index_duration = 0;
    while (true)
    {
        if (!cc.TxStart(info))
            continue;

        District district;

        // UPDATE district table next_o_id
        // cur_o_id = next_o_ids[district_idx];
        // next_o_ids[district_idx] += 1;

        void *districtp = (void *)(table_district.data + district_idx * sizeof(District));
        if (!cc.ReadForUpdate(
                table_st_arr[TABLE_DISTRICT] + district_idx,
                0,
                districtp,
                (void *)&district,
                sizeof(District)))
        {
            continue;
        }

        cur_o_id = district.next_o_id;
        district.next_o_id++;

        if (!cc.Write(
                table_st_arr[TABLE_DISTRICT] + district_idx,
                0,
                (void *)&district,
                districtp,
                sizeof(District)))
        {
            continue;
        }

        bool success1 = true;
        sum_ol_amount = 0;
        partial_index_duration = 0;
        for (unsigned int ol_idx = 0; ol_idx < tx.ol_cnt; ++ol_idx)
        {
            OrderLine &the_ol_item = ((OrderLine *)table_orderline.data)[ol_idx_insert_start];
            the_ol_item.i_id = tx.i_ids[ol_idx];
            the_ol_item.supply_w_id = tx.supply_w_ids[ol_idx];
            the_ol_item.quantity = tx.quantities[ol_idx];

            index_start_time = clock64();
            size_t item_idx = index_item->Find(the_ol_item.i_id);
            float item_price = ((Item *)table_item.data)[item_idx].price;

            unsigned int stock_id = the_ol_item.supply_w_id * ITEMS_NUM + the_ol_item.i_id;
            size_t stock_idx = index_stock->Find(stock_id);
            partial_index_duration += clock64() - index_start_time;

            // UPDATE stock table
            // compute address of s_dist information manually
            // (seems to be slightly faster than inlined switch?)
            // char *the_s_dist = (char *)((char *)(stocks + s_idx) + offsetof(data_row::Stock, dist00) + the_tx.d_id * sizeof(char[25]));

            void *stockp = (void *)(table_stock.data + stock_idx * sizeof(Stock));

            if (!cc.ReadForUpdate(
                    table_st_arr[TABLE_STOCK] + stock_idx,
                    ol_idx + 1,
                    stockp,
                    &stock,
                    sizeof(Stock)))
            {
                success1 = false;
                break;
            }

            unsigned int stock_quantity = stock.quantity;
            if (stock_quantity > the_ol_item.quantity + 10)
                stock.quantity -= the_ol_item.quantity;
            else
                stock.quantity -= (the_ol_item.quantity - 91);

            stock.ytd += the_ol_item.quantity;
            stock.order_cnt += 1;
            stock.remote_cnt += (unsigned int)(the_ol_item.supply_w_id != tx.w_id);

            if (!cc.Write(
                    table_st_arr[TABLE_STOCK] + stock_idx,
                    ol_idx + 1,
                    &stock,
                    stockp,
                    sizeof(Stock)))
            {
                success1 = false;
                break;
            }

            sum_ol_amount += the_ol_item.quantity * item_price;

            // INSERT into orderline table
            the_ol_item.number = ol_idx;
            the_ol_item.amount = the_ol_item.quantity * item_price;

            // add to tx_output
            OrderlineOutput &ol_output = outp->ol_out[ol_idx];
            // memcpy(ol_output.i_name, items[the_ol_item.i_id].name, 25);
            ol_output.s_quantity = stock_quantity;
            ol_output.price = item_price;
            ol_output.amount = the_ol_item.quantity * item_price;

            ol_idx_insert_start++;
        }

        if (success1 && cc.TxEnd(NULL))
            break;
    }

    // INSERT into orders and new_orders
    // ((Order *)table_order.data)[tid] = Order{cur_o_id, tx.w_id, tx.d_id, tx.c_id,
    //                                          tx.carrier_id, tx.ol_cnt, tx.all_local,
    //                                          tx.entry_d};
    ((Order *)table_order.data)[tid] = Order(cur_o_id, tx.w_id, tx.d_id, tx.c_id,
                                             tx.carrier_id, tx.ol_cnt, tx.all_local,
                                             tx.entry_d);

    // ((NewOrder *)table_neworder.data)[tid] = NewOrder{cur_o_id, tx.w_id, tx.d_id};
    ((NewOrder *)table_neworder.data)[tid] = NewOrder(cur_o_id, tx.w_id, tx.d_id);

    outp->total_amount = sum_ol_amount * (1 - customer_discount) * (1 + wh_tax + district_tax);
    outp->o_id = cur_o_id;
    outp->w_id = tx.w_id;
    outp->d_id = tx.d_id;

    atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->tot_duration), clock64() - tx_start_time);
    atomicAdd(&(((common::Metrics *)GLOBAL_METRICS)->index_duration), index_duration + partial_index_duration);

    cc.Finalize();
}
