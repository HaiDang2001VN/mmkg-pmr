// g++ -std=c++23 -O3 -pthread -o preprocess.o preprocess.cpp
//
// Shortest-path search over a static knowledge graph via bidirectional BFS.
// Reads a graph CSV and a queries CSV; for each query assigned to this thread,
// finds a path between head and tail within max_hops, with fanout decay.
//
// Usage:
//   ./preprocess.o <graph_csv> <queries_csv> <output_prefix>
//                  <max_hops> <max_fanout> <decay_factor>
//                  <thread_id> <num_threads> <log_dir>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// ── helpers ──────────────────────────────────────────────────────────────

vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    stringstream ss(s);
    while (getline(ss, token, delimiter))
        tokens.push_back(token);
    return tokens;
}

// ── data structures ──────────────────────────────────────────────────────

struct GraphEdge {
    int v;
    int edge_type;
    int reverse_edge_type;
    bool traversed_reverse;
};

struct ParentLink {
    int parent;
    int edge_type;
    bool traversed_reverse;
};

struct Query {
    int row_index;
    int query_id;
    int head_id;
    int tail_id;
};

// ── main ─────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 10) {
        cerr << "Usage: " << argv[0]
             << " <graph_csv> <queries_csv> <output_prefix>"
             << " <max_hops> <max_fanout> <decay_factor>"
             << " <thread_id> <num_threads> <log_dir>\n";
        return 1;
    }

    string graph_csv     = argv[1];
    string queries_csv   = argv[2];
    string output_prefix = argv[3];
    int    max_hops      = stoi(argv[4]);
    int    max_fanout    = stoi(argv[5]);
    float  decay_factor  = stof(argv[6]);
    int    thread_id     = stoi(argv[7]);
    int    num_threads   = stoi(argv[8]);
    string log_dir       = argv[9];

    // ── log stream ───────────────────────────────────────────────────────
    string log_name = log_dir + "/_" + to_string(thread_id) + "_preprocess.log";
    ofstream log(log_name, ios_base::app);
    if (!log.is_open()) {
        cerr << "Failed to open log: " << log_name << endl;
        return 1;
    }
    log << "[Info] Thread " << thread_id << "/" << num_threads
        << "  max_hops=" << max_hops << "  max_fanout=" << max_fanout
        << "  decay=" << decay_factor << endl;

    // ── 1. Load graph ────────────────────────────────────────────────────
    unordered_map<int, vector<GraphEdge>> adj;      // node → neighbours
    unordered_map<int, int>               node_type; // node_id → type

    {
        ifstream in(graph_csv);
        if (!in.is_open()) {
            log << "[Error] Cannot open graph CSV: " << graph_csv << endl;
            return 1;
        }

        string header;
        getline(in, header);  // skip header

        string line;
        int lineno = 1;
        while (getline(in, line)) {
            lineno++;
            auto cols = split(line, ',');
            if (cols.size() < 5) {
                log << "[Warn] Line " << lineno << ": expected at least 5 cols, got "
                    << cols.size() << endl;
                continue;
            }
            try {
                int u      = stoi(cols[0]);
                int v      = stoi(cols[1]);
                int u_type = stoi(cols[2]);
                int v_type = stoi(cols[3]);
                int e_type = stoi(cols[4]);
                int reverse_e_type = cols.size() >= 6 ? stoi(cols[5]) : e_type;

                // Each adjacency entry stores the relation for the direction
                // actually traversed. The opposite relation is retained so
                // backward-BFS reconstruction does not have to guess it from
                // an endpoint pair with possibly parallel edges.
                adj[u].push_back({v, e_type, reverse_e_type, false});
                adj[v].push_back({u, reverse_e_type, e_type, true});
                if (!node_type.count(u)) node_type[u] = u_type;
                if (!node_type.count(v)) node_type[v] = v_type;
            } catch (const exception &ex) {
                log << "[Warn] Line " << lineno << ": parse error — "
                    << ex.what() << endl;
            }
        }
    }
    for (auto &entry : adj) {
        auto &edges = entry.second;
        sort(edges.begin(), edges.end(), [](const GraphEdge &a,
                                            const GraphEdge &b) {
            return tie(a.v, a.edge_type, a.reverse_edge_type,
                       a.traversed_reverse)
                 < tie(b.v, b.edge_type, b.reverse_edge_type,
                       b.traversed_reverse);
        });
    }
    log << "[Info] Graph loaded: " << adj.size() << " nodes, "
        << node_type.size() << " typed" << endl;

    // ── 2. Load queries ──────────────────────────────────────────────────
    vector<Query> queries;
    {
        ifstream in(queries_csv);
        if (!in.is_open()) {
            log << "[Error] Cannot open queries CSV: " << queries_csv << endl;
            return 1;
        }
        string header;
        getline(in, header);

        string line;
        int lineno = 1;
        while (getline(in, line)) {
            lineno++;
            auto cols = split(line, ',');
            if (cols.size() < 4) {
                log << "[Warn] Line " << lineno << ": expected 4 cols "
                    << "(row_index,query_id,head_id,tail_id), got "
                    << cols.size() << endl;
                continue;
            }
            try {
                Query q;
                q.row_index = stoi(cols[0]);
                q.query_id  = stoi(cols[1]);
                q.head_id   = stoi(cols[2]);
                q.tail_id   = stoi(cols[3]);
                queries.push_back(q);
            } catch (const exception &ex) {
                log << "[Warn] Line " << lineno << ": parse error — "
                    << ex.what() << endl;
            }
        }
    }
    log << "[Info] Queries loaded: " << queries.size() << endl;

    // ── 3. Process queries (this thread's share) ─────────────────────────
    string out_name = output_prefix + "_paths_" + to_string(thread_id) + ".txt";
    ofstream out(out_name);
    if (!out.is_open()) {
        log << "[Error] Cannot open output: " << out_name << endl;
        return 1;
    }

    // count how many queries belong to this thread
    int my_total = 0;
    for (const auto &q : queries)
        if (q.query_id % num_threads == thread_id) my_total++;
    log << "[Info] My queries: " << my_total << endl;
    int processed = 0;
    int found     = 0;
    int missing   = 0;
    int self      = 0;
    int last_pct  = -1;

    auto log_progress = [&]() {
        processed++;
        if (my_total > 0) {
            int pct = (int)(100.0 * processed / my_total);
            if (pct > last_pct) {
                last_pct = pct;
                log << "[Progress] " << pct << "%  processed=" << processed
                    << "  found=" << found << "  missing=" << missing
                    << "  self=" << self << endl;
                log.flush();
            }
        }
    };

    for (const auto &q : queries) {
        if (q.query_id % num_threads != thread_id)
            continue;

        // ── self-probe: candidate tail equals the query head ────────────
        // No self-edge exists in the scoped graph, so the probe is emitted as
        // an explicit no-path record per the explicit-probe contract:
        //   qid;row_index;hops;nodes;node_types;edge_types;edge_directions
        if (q.head_id == q.tail_id) {
            out << q.query_id << ";" << q.row_index << ";-1;;;;\n";
            missing++;
            self++;
            log_progress();
            continue;
        }

        {
        // ── bidirectional BFS ────────────────────────────────────────────
        queue<int> q_fwd, q_bwd;
        unordered_map<int, ParentLink> parent_fwd, parent_bwd;
        unordered_map<int, int> depth_fwd, depth_bwd;

        q_fwd.push(q.head_id);
        parent_fwd[q.head_id] = {q.head_id, -1, false};
        depth_fwd[q.head_id]  = 0;

        q_bwd.push(q.tail_id);
        parent_bwd[q.tail_id] = {q.tail_id, -1, false};
        depth_bwd[q.tail_id]  = 0;

        int  meet_node  = -1;
        bool path_found = false;

        while (!q_fwd.empty() && !q_bwd.empty() && !path_found) {
            if (depth_fwd[q_fwd.front()] + depth_bwd[q_bwd.front()] >= max_hops)
                break;
            // ── forward ──────────────────────────────────────────────────
            {
                int cur_depth = depth_fwd[q_fwd.front()];
                int limit = static_cast<int>(
                    ceil(max_fanout * pow(decay_factor, cur_depth)));
                int sz = (int)q_fwd.size();
                for (int i = 0; i < sz && !path_found; ++i) {
                    int u = q_fwd.front(); q_fwd.pop();
                    if (!adj.count(u)) continue;
                    int added = 0;
                    for (auto it = adj.at(u).begin();
                         it != adj.at(u).end() && added < limit; ++it) {
                        int v = it->v;
                        if (!parent_fwd.count(v)) {
                            parent_fwd[v] = {
                                u, it->edge_type, it->traversed_reverse,
                            };
                            depth_fwd[v]  = depth_fwd[u] + 1;
                            q_fwd.push(v);
                            added++;
                            if (parent_bwd.count(v)) {
                                meet_node  = v;
                                path_found = true;
                                break;
                            }
                        }
                    }
                }
            }
            if (path_found) break;
            // ── backward ─────────────────────────────────────────────────
            {
                int cur_depth = depth_bwd[q_bwd.front()];
                int limit = static_cast<int>(
                    ceil(max_fanout * pow(decay_factor, cur_depth)));
                int sz = (int)q_bwd.size();
                for (int i = 0; i < sz && !path_found; ++i) {
                    int u = q_bwd.front(); q_bwd.pop();
                    if (!adj.count(u)) continue;
                    int added = 0;
                    for (auto it = adj.at(u).begin();
                         it != adj.at(u).end() && added < limit; ++it) {
                        int v = it->v;
                        if (!parent_bwd.count(v)) {
                            // The search moved u -> v, but reconstruction
                            // needs the opposite segment v -> u.
                            parent_bwd[v] = {
                                u, it->reverse_edge_type,
                                !it->traversed_reverse,
                            };
                            depth_bwd[v]  = depth_bwd[u] + 1;
                            q_bwd.push(v);
                            added++;
                            if (parent_fwd.count(v)) {
                                meet_node  = v;
                                path_found = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if (meet_node == -1) {
            // qid;row_index;hops;nodes;node_types;edge_types;edge_directions
            out << q.query_id << ";" << q.row_index << ";-1;;;;\n";
            missing++;
            log_progress();
            continue;
        }

        // reconstruct path
        vector<int> path_nodes;
        vector<int> edge_types;
        vector<int> edge_directions;
        vector<int> left_nodes;
        vector<int> left_edge_types;
        vector<int> left_edge_directions;
        int cur = meet_node;
        while (cur != q.head_id) {
            const auto &link = parent_fwd.at(cur);
            left_nodes.push_back(cur);
            left_edge_types.push_back(link.edge_type);
            left_edge_directions.push_back(link.traversed_reverse);
            cur = link.parent;
        }
        left_nodes.push_back(q.head_id);
        reverse(left_nodes.begin(), left_nodes.end());
        reverse(left_edge_types.begin(), left_edge_types.end());
        reverse(left_edge_directions.begin(), left_edge_directions.end());
        path_nodes = left_nodes;
        edge_types = left_edge_types;
        edge_directions = left_edge_directions;

        cur = meet_node;
        while (cur != q.tail_id) {
            const auto &link = parent_bwd.at(cur);
            path_nodes.push_back(link.parent);
            edge_types.push_back(link.edge_type);
            edge_directions.push_back(link.traversed_reverse);
            cur = link.parent;
        }

        int hops = (int)path_nodes.size() - 1;

        vector<int> node_types;
        for (int nid : path_nodes)
            node_types.push_back(node_type.count(nid) ? node_type.at(nid) : 0);

        out << q.query_id << ";" << q.row_index << ";" << hops << ";";
        for (size_t i = 0; i < path_nodes.size(); ++i)
            out << path_nodes[i] << (i + 1 < path_nodes.size() ? "," : "");
        out << ";";
        for (size_t i = 0; i < node_types.size(); ++i)
            out << node_types[i] << (i + 1 < node_types.size() ? "," : "");
        out << ";";
        for (size_t i = 0; i < edge_types.size(); ++i)
            out << edge_types[i] << (i + 1 < edge_types.size() ? "," : "");
        out << ";";
        for (size_t i = 0; i < edge_directions.size(); ++i)
            out << edge_directions[i]
                << (i + 1 < edge_directions.size() ? "," : "");
        out << "\n";
        found++;
        log_progress();
        }  // end BFS scope
    }

    out.close();

    log << "[Info] Done — found=" << found << "  unreachable=" << missing
        << "  (self-probes=" << self << ")  total=" << (found + missing)
        << endl;
    log.close();
    return 0;
}
