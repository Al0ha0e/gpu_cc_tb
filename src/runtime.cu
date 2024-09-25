#include <runtime.cuh>
#include <stdio.h>

#ifndef NVRTC_COMPILE
#include <map>
#include <set>
#include <vector>
#include <stack>
#include <algorithm>
#endif

namespace common
{

#ifndef NVRTC_COMPILE

    static std::map<size_t, std::set<int>> active_txs;
    static std::vector<std::set<int>> graph;
    static std::vector<std::vector<common::Event *>> opsets;
    static std::stack<int> trace_stack;
    static bool *visit1, *visit2, *committed;

    void ExecInfo::scan_events()
    {
        std::sort(host_events, host_events + event_cnt);
        for (int i = 0; i < event_cnt; i++)
        {
            common::Event &event = host_events[i];
            // std::cout << i << " " << event.id << " " << event.tid << " " << event.oid << " " << event.type << "\n";
            if (event.type == 2) // commit
            {
                committed[event.tid] = true;
                for (common::Event *event1 : opsets[event.tid])
                    active_txs[event1->oid].erase(event.tid);
            }
            else
            {
                opsets[event.tid].push_back(host_events + i);
                if (active_txs.find(event.oid) == active_txs.end())
                    active_txs[event.oid] = std::set<int>();

                if (event.type == 0) // read
                {
                    for (int tid : active_txs[event.oid])
                    {
                        if (tid == event.tid)
                            continue;
                        auto &opset = opsets[tid];
                        for (common::Event *event1 : opset)
                        {
                            if (event1->oid == event.oid && event1->type == 1)
                            {
                                graph[event.tid].insert(tid);
                                break;
                            }
                        }
                    }
                }
                else // write
                {
                    for (int tid : active_txs[event.oid])
                    {
                        if (tid == event.tid)
                            continue;
                        graph[event.tid].insert(tid);
                    }
                }
                active_txs[event.oid].insert(event.tid);
            }
        }
    }

    void ExecInfo::check0()
    {
        for (int i = 0; i < txcnt; i++)
            if (!committed[i])
                printf("UNCOMMITTED %d\n", i);
    }

    bool ExecInfo::check(int now)
    {
        if (visit1[now])
            return true;
        if (visit2[now])
        {
            backtrack(now);
            return false;
        }
        visit2[now] = true;
        trace_stack.push(now);
        for (int nxt : graph[now])
            if (!check(nxt))
                return false;
        trace_stack.pop();
        visit2[now] = false;
        visit1[now] = true;
        return true;
    }

    void ExecInfo::backtrack(int now)
    {
        printf("!!!!!!!!! %d\n", now);
        std::vector<Event> bad_events;
        while (!trace_stack.empty())
        {
            int tx = trace_stack.top();
            trace_stack.pop();
            for (Event *event : opsets[tx])
                bad_events.push_back(*event);
            printf("BAD TX %d\n", tx);
            for (int nxt : graph[tx])
                printf("%d ", nxt);
            printf("\n");
            if (tx == now)
                break;
        }
        std::sort(bad_events.begin(), bad_events.end());
        printf("--------------------------BAD ST-----------------------\n");
        for (Event &event : bad_events)
        {
            std::cout << event.id << " tid " << event.tid << " oid" << event.oid << " " << (event.type == 0 ? 'r' : (event.type == 1 ? 'w' : 'c')) << "\n";
            cc->Explain(event.self_info, event.target_info);
            // std::cout << "ts" << event.ts << " o " << (event.ts == (event.ts & ((1UL << 31) - 1))) << " | " << tt.s.uncommited << " r " << tt.s.rts << " w " << tt.s.wts << "\n";
        }

        printf("--------------------------BAD EN-----------------------\n");
    }

    void ExecInfo::Check()
    {
        host_events = new Event[event_cnt];
        cudaMemcpy(host_events, gpu_events, event_cnt * sizeof(Event), cudaMemcpyDeviceToHost);
        graph.resize(txcnt);
        opsets.resize(txcnt);
        committed = new bool[txcnt];
        memset(committed, 0, sizeof(bool) * txcnt);
        scan_events();
        std::cout << "SCAN OK\n";
        check0();
        size_t visit_size = sizeof(bool) * txcnt;
        visit1 = (bool *)malloc(visit_size);
        visit2 = (bool *)malloc(visit_size);
        memset(visit1, 0, visit_size);
        bool fail = false;
        for (int i = 0; i < txcnt; i++)
        {
            if (!visit1[i])
            {
                memset(visit2, 0, visit_size);
                if (!check(i))
                {
                    fail = true;
                    break;
                }
            }
        }
        printf(fail ? "BAD\n" : "GOOD\n");
    }

#endif

}