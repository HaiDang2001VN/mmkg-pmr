// g++ -std=c++23 -O3 -pthread -o preprocess.o preprocess.cpp
//
// Shortest-path search over a static knowledge graph via bidirectional BFS.
// Reads a graph CSV and a queries CSV; for each probe (query head, candidate
// tail) finds a path between them within max_hops under fanout decay.
//
// Per-probe masking (optional trailing arguments):
//   mask_own_edges=1 — the probe's own fact family between head and tail
//       (oriented relation qrel or its registry inverse, in either
//       orientation) is masked, so train evidence matches the eval-time
//       path graph (val/test scopes already exclude own-split facts).
//   allow_target_traversal=0 — traversal through ANY target-relation edge
//       is denied for every split (uniform rule, no per-split skew).
//
// Throughput: all probes of a query share the head, so the forward BFS from
// the head runs once and is reused; each probe runs only its own backward
// BFS. A probe whose own edge exists (masking active) runs a full per-probe
// masked bidirectional BFS instead. Backward expansion continues while the
// frontier is non-empty within the depth bound, so a drained frontier no
// longer forces hops=-1.
//
// Output rows (7 ';'-fields):
//   query_id;row_index;hops;nodes;node_types;edge_types;edge_directions
// Unreachable probes emit hops=-1.
//
// Usage:
//   ./preprocess.o <graph_csv> <queries_csv> <output_prefix>
//                  <max_hops> <max_fanout> <decay_factor>
//                  <thread_id> <num_threads> <log_dir>
//                  [mask_own_edges] [allow_target_traversal] [target_ids_csv]

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
    int rel_id;
    int inv_rel_id;
};

// ── main ─────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 10) {
        cerr << "Usage: " << argv[0]
             << " <graph_csv> <queries_csv> <output_prefix>"
             << " <max_hops> <max_fanout> <decay_factor>"
             << " <thread_id> <num_threads> <log_dir>"
             << " [mask_own_edges] [allow_target_traversal] [target_ids_csv]\n";
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

    bool mask_own_edges = argc > 10 ? stoi(argv[10]) != 0 : false;
    bool allow_target_traversal = argc > 11 ? stoi(argv[11]) != 0 : true;
    unordered_set<int> target_ids;
    if (argc > 12) {
        for (const auto &part : split(argv[12], ',')) {
            if (!part.empty()) target_ids.insert(stoi(part));
        }
    }

    // ── log stream ───────────────────────────────────────────────────────
    string log_name = log_dir + "/_" + to_string(thread_id) + "_preprocess.log";
    ofstream log(log_name, ios_base::app);
    if (!log.is_open()) {
        cerr << "Failed to open log: " << log_name << endl;
        return 1;
    }
    log << "[Info] Thread " << thread_id << "/" << num_threads
        << "  max_hops=" << max_hops << "  max_fanout=" << max_fanout
        << "  decay=" << decay_factor
        << "  mask_own_edges=" << mask_own_edges
        << "  allow_target_traversal=" << allow_target_traversal
        << "  target_ids=" << target_ids.size() << endl;

    // ── 1. Load graph ────────────────────────────────────────────────────
    unordered_map<int, vector<GraphEdge>> adj;       // node → neighbours
    unordered_map<int, int>               node_type;  // node_id → type

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
                log << "[Warn] Line " << lineno << ": expected 6 cols "
                    << "(row_index,query_id,head_id,tail_id,query_relation_id,"
                    "inverse_query_relation_id), got " << cols.size() << endl;
                continue;
            }
            try {
                Query q;
                q.row_index = stoi(cols[0]);
                q.query_id  = stoi(cols[1]);
                q.head_id   = stoi(cols[2]);
                q.tail_id   = stoi(cols[3]);
                // Older queries files without the relation columns keep the
                // own-edge mask inert (rel id -1 matches no edge type).
                q.rel_id     = cols.size() > 4 && !cols[4].empty()
                               ? stoi(cols[4]) : -1;
                q.inv_rel_id = cols.size() > 5 && !cols[5].empty()
                               ? stoi(cols[5]) : -1;
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
                    << "  (self-probes=" << self << ")" << endl;
                log.flush();
            }
        }
    };

    // Own-edge mask: the query's own fact family between the endpoints. The
    // undirected pair {head, tail} carrying the oriented query relation or
    // its registry inverse is masked in BOTH orientations when active.
    auto edge_allowed = [&](int u, int v, int edge_type, const Query &q) {
        if (mask_own_edges
                && (edge_type == q.rel_id || edge_type == q.inv_rel_id)
                && ((u == q.head_id && v == q.tail_id)
                    || (u == q.tail_id && v == q.head_id))) {
            return false;
        }
        if (!allow_target_traversal
                && !target_ids.empty()
                && target_ids.count(edge_type)) {
            return false;
        }
        return true;
    };

    // Whether an own edge exists between head and tail. When it does not,
    // the mask can never fire for this probe and the shared forward BFS is
    // valid as-is.
    auto has_own_edge = [&](const Query &q) {
        if (!mask_own_edges) return false;
        if (q.rel_id < 0 && q.inv_rel_id < 0) return false;
        auto side_has = [&](int u, int w) {
            auto it = adj.find(u);
            if (it == adj.end()) return false;
            for (const auto &e : it->second) {
                if (e.v > w) break;  // adjacency sorted by neighbour id
                if (e.v == w && (e.edge_type == q.rel_id
                                 || e.edge_type == q.inv_rel_id)) {
                    return true;
                }
            }
            return false;
        };
        return side_has(q.head_id, q.tail_id)
            || side_has(q.tail_id, q.head_id);
    };

    auto emit_no_path = [&](const Query &q) {
        out << q.query_id << ";" << q.row_index << ";-1;;;;\n";
        missing++;
        log_progress();
    };

    auto emit_path = [&](const Query &q, const vector<int> &path_nodes,
                         const vector<int> &edge_types,
                         const vector<int> &edge_directions) {
        int hops = (int)path_nodes.size() - 1;
        out << q.query_id << ";" << q.row_index << ";" << hops << ";";
        for (size_t i = 0; i < path_nodes.size(); ++i)
            out << path_nodes[i] << (i + 1 < path_nodes.size() ? "," : "");
        out << ";";
        for (size_t i = 0; i < path_nodes.size(); ++i) {
            auto tit = node_type.find(path_nodes[i]);
            out << (tit != node_type.end() ? tit->second : 0);
            if (i + 1 < path_nodes.size()) out << ",";
        }
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
    };

    size_t gi = 0;
    while (gi < queries.size()) {
        const Query &head_probe = queries[gi];
        if (head_probe.query_id % num_threads != thread_id) {
            ++gi;
            continue;
        }
        size_t group_end = gi;
        while (group_end < queries.size()
               && queries[group_end].query_id == head_probe.query_id
               && queries[group_end].head_id == head_probe.head_id) {
            ++group_end;
        }

        // ── shared forward BFS from the query head (no own-edge mask) ────
        // Depth bookkeeping is implicit: the frontier is level-ordered, so
        // nodes enqueued during level d sit at depth d+1.
        unordered_map<int, ParentLink> fwd_parent;
        unordered_map<int, int> fwd_depth;
        queue<int> fwd_frontier;
        fwd_parent[head_probe.head_id] = {head_probe.head_id, -1, false};
        fwd_depth[head_probe.head_id] = 0;
        fwd_frontier.push(head_probe.head_id);
        for (int depth = 0; depth < max_hops && !fwd_frontier.empty(); ++depth) {
            int limit = static_cast<int>(
                ceil(max_fanout * pow(decay_factor, depth)));
            int sz = (int)fwd_frontier.size();
            for (int i = 0; i < sz; ++i) {
                int u = fwd_frontier.front(); fwd_frontier.pop();
                auto ait = adj.find(u);
                if (ait == adj.end()) continue;
                int added = 0;
                for (auto it = ait->second.begin();
                     it != ait->second.end() && added < limit; ++it) {
                    int v = it->v;
                    if (fwd_parent.count(v)) continue;
                    if (!edge_allowed(u, v, it->edge_type, head_probe)) continue;
                    fwd_parent[v] = {u, it->edge_type, it->traversed_reverse};
                    fwd_depth[v] = depth + 1;
                    fwd_frontier.push(v);
                    ++added;
                }
            }
        }

        // ── per-probe backward BFS + meet detection ──────────────────────
        for (size_t j = gi; j < group_end; ++j) {
            const Query &q = queries[j];
            if (q.query_id % num_threads != thread_id) continue;

            if (q.head_id == q.tail_id) {
                emit_no_path(q);
                self++;
                continue;
            }

            // A probe whose own edge is in the graph needs a masked forward
            // state; every other probe reuses the shared unmasked tree.
            bool needs_masked_fwd = has_own_edge(q);
            unordered_map<int, ParentLink> probe_parent;
            unordered_map<int, int> probe_depth;
            if (needs_masked_fwd) {
                queue<int> pf;
                probe_parent[q.head_id] = {q.head_id, -1, false};
                probe_depth[q.head_id] = 0;
                pf.push(q.head_id);
                for (int depth = 0; depth < max_hops && !pf.empty(); ++depth) {
                    int limit = static_cast<int>(
                        ceil(max_fanout * pow(decay_factor, depth)));
                    int sz = (int)pf.size();
                    for (int i = 0; i < sz; ++i) {
                        int u = pf.front(); pf.pop();
                        auto ait = adj.find(u);
                        if (ait == adj.end()) continue;
                        int added = 0;
                        for (auto it = ait->second.begin();
                             it != ait->second.end() && added < limit; ++it) {
                            int v = it->v;
                            if (probe_parent.count(v)) continue;
                            if (!edge_allowed(u, v, it->edge_type, q)) continue;
                            probe_parent[v] = {u, it->edge_type,
                                               it->traversed_reverse};
                            pf.push(v);
                            ++added;
                        }
                    }
                }
            }
            const unordered_map<int, ParentLink> &fparent =
                needs_masked_fwd ? probe_parent : fwd_parent;
            const unordered_map<int, int> &fdepth =
                needs_masked_fwd ? probe_depth : fwd_depth;

            unordered_map<int, ParentLink> bwd_parent;
            unordered_map<int, int> bwd_depth;
            queue<int> bwd_frontier;
            bwd_parent[q.tail_id] = {q.tail_id, -1, false};
            bwd_depth[q.tail_id] = 0;
            bwd_frontier.push(q.tail_id);

            int meet_node = -1;
            if (fparent.count(q.tail_id)) {
                meet_node = q.tail_id;
            } else {
                bool path_found = false;
                for (int depth = 0;
                     depth < max_hops && !bwd_frontier.empty() && !path_found;
                     ++depth) {
                    int limit = static_cast<int>(
                        ceil(max_fanout * pow(decay_factor, depth)));
                    int sz = (int)bwd_frontier.size();
                    for (int i = 0; i < sz && !path_found; ++i) {
                        int u = bwd_frontier.front(); bwd_frontier.pop();
                        auto ait = adj.find(u);
                        if (ait == adj.end()) continue;
                        int added = 0;
                        for (auto it = ait->second.begin();
                             it != ait->second.end() && added < limit; ++it) {
                            int v = it->v;
                            if (bwd_parent.count(v)) continue;
                            if (!edge_allowed(u, v, it->edge_type, q)) continue;
                            // The search moves u→v, but reconstruction needs
                            // the opposite segment v→u, so the oriented
                            // relation and the direction bit flip.
                            bwd_parent[v] = {u, it->reverse_edge_type,
                                             !it->traversed_reverse};
                            bwd_depth[v] = depth + 1;
                            bwd_frontier.push(v);
                            ++added;
                            if (fparent.count(v)
                                    && fdepth.at(v) + bwd_depth[v] <= max_hops) {
                                meet_node = v;
                                path_found = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (meet_node == -1) {
                emit_no_path(q);
                continue;
            }

            vector<int> path_nodes, edge_types, edge_directions;
            // head→meet segment
            {
                vector<int> seg_nodes, seg_etypes, seg_edirs;
                int cur = meet_node;
                while (fparent.at(cur).parent != cur) {
                    const auto &link = fparent.at(cur);
                    seg_nodes.push_back(cur);
                    seg_etypes.push_back(link.edge_type);
                    seg_edirs.push_back(link.traversed_reverse ? 1 : 0);
                    cur = link.parent;
                }
                seg_nodes.push_back(cur);
                reverse(seg_nodes.begin(), seg_nodes.end());
                reverse(seg_etypes.begin(), seg_etypes.end());
                reverse(seg_edirs.begin(), seg_edirs.end());
                path_nodes.swap(seg_nodes);
                edge_types = seg_etypes;
                edge_directions = seg_edirs;
            }
            // meet→tail segment via the backward tree
            int cur = meet_node;
            while (cur != q.tail_id) {
                const auto &link = bwd_parent.at(cur);
                path_nodes.push_back(link.parent);
                edge_types.push_back(link.edge_type);
                edge_directions.push_back(link.traversed_reverse ? 1 : 0);
                cur = link.parent;
            }
            emit_path(q, path_nodes, edge_types, edge_directions);
        }

        gi = group_end;
    }

    out.close();

    log << "[Info] Done — found=" << found << "  unreachable=" << missing
        << "  (self-probes=" << self << ")  total=" << (found + missing)
        << endl;
    log.close();
    return 0;
}
