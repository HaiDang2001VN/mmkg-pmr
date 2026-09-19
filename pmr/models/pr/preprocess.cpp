// g++ -std=c++23 -O3 -pthread -o preprocess.o preprocess.cpp
//
// Shortest-path search over a static knowledge graph via bidirectional BFS
// (worker v2). Reads a graph CSV and a queries CSV; for each probe (query
// head, candidate tail) finds a path between them within max_hops under
// fanout decay.
//
// Process model (v2): ONE process parses the graph once and spawns
// num_threads internal std::threads; each thread owns a contiguous range of
// query groups, its own _paths_<tid>.txt output file, and its own
// _<tid>_preprocess.log. The merge contract (7 ';'-fields, one row per
// prepared probe) is unchanged from v1; only the thread→query partition
// differs (contiguous group ranges instead of query_id % num_threads).
//
// Memory model (v2): the adjacency is a flat CSR array of 12-byte packed
// edges (sorted and exact-duplicate-deduped per node), node types are a flat
// byte vector, BFS state lives in per-thread epoch-stamped arenas reused
// across every probe (no per-probe allocations), and the CSV is parsed in
// place from chunked reads with from_chars (no stringstream, no per-line
// string allocations).
//
// Behaviour preserved from v1: per-probe own-edge masking (query relation +
// registry inverse between head and tail, both orientations) and the
// uniform target-traversal deny rule; a probe whose own edge exists runs a
// fully masked per-probe bidirectional search; the shared forward tree per
// query is built WITHOUT the own-edge mask (cross-probe facts are
// legitimately traversable at eval time); backward expansion continues
// level-by-level within the depth bound so a drained frontier no longer
// forces hops=-1.
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
// argv[7] (thread_id) is accepted for invocation compatibility but is
// vestigial: threads are managed internally.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

// ── helpers ──────────────────────────────────────────────────────────────

static vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(delimiter, start);
        if (pos == string::npos) {
            tokens.push_back(s.substr(start));
            break;
        }
        tokens.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return tokens;
}

// Parse one integer spanning [p, field_end). Trims whitespace and a
// trailing '\r'. Succeeds only when the whole field is consumed.
static bool parse_int(const char *p, const char *field_end, int32_t &out) {
    while (p < field_end && (*p == ' ' || *p == '\t')) ++p;
    while (field_end > p && (field_end[-1] == '\r' || field_end[-1] == ' '))
        --field_end;
    auto result = from_chars(p, field_end, out);
    return result.ec == errc() && result.ptr == field_end;
}

// Buffered chunked line reader: RSS stays at one 16 MB chunk regardless of
// file size; a partial trailing line is carried across refills.
class LineReader {
public:
    explicit LineReader(const string &path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        ok_ = fd_ >= 0;
        buf_.resize(kChunk);
    }
    ~LineReader() {
        if (fd_ >= 0) ::close(fd_);
    }
    LineReader(const LineReader &) = delete;
    LineReader &operator=(const LineReader &) = delete;

    bool good() const { return ok_; }

    // Rewind to the first data line (skips the header line).
    bool rewind_to_data() {
        if (!ok_ || lseek(fd_, 0, SEEK_SET) < 0) return false;
        pos_ = len_ = 0;
        eof_ = false;
        const char *lb;
        const char *le;
        next_line(lb, le);  // header (no-op on an empty file)
        return true;
    }

    // Next '\n'-bounded line in [lb, le); a final unterminated line is
    // returned as-is; returns false at end of input.
    bool next_line(const char *&lb, const char *&le) {
        for (;;) {
            if (pos_ < len_) {
                const void *nl =
                    memchr(buf_.data() + pos_, '\n', len_ - pos_);
                if (nl) {
                    size_t nlpos =
                        static_cast<size_t>(static_cast<const char *>(nl)
                                            - buf_.data());
                    lb = buf_.data() + pos_;
                    le = buf_.data() + nlpos;
                    pos_ = nlpos + 1;
                    return true;
                }
                if (eof_) {  // final unterminated line
                    if (pos_ == len_) return false;
                    lb = buf_.data() + pos_;
                    le = buf_.data() + len_;
                    pos_ = len_;
                    return true;
                }
            } else if (eof_) {
                return false;
            }
            // Need more data: carry the partial tail, refill.
            size_t carry = len_ - pos_;
            if (carry > 0 && pos_ > 0) {
                memmove(buf_.data(), buf_.data() + pos_, carry);
            }
            pos_ = 0;
            ssize_t got = ::read(fd_, buf_.data() + carry,
                                 buf_.size() - carry);
            if (got <= 0) {
                eof_ = true;
                len_ = carry;
                continue;
            }
            len_ = carry + static_cast<size_t>(got);
        }
    }

private:
    static constexpr size_t kChunk = 16u << 20;
    int fd_ = -1;
    vector<char> buf_;
    size_t pos_ = 0, len_ = 0;
    bool eof_ = false;
    bool ok_ = false;
};

// ── packed graph structures ──────────────────────────────────────────────

// One directed adjacency entry, 12 bytes. The opposite direction's oriented
// relation is retained so backward-BFS reconstruction does not have to guess
// it from an endpoint pair with possibly parallel edges. The physical
// reverse bit of the u→v traversal rides in bit 0 of rev_and_flag.
struct PackedEdge {
    int32_t v;
    int32_t e_type;        // oriented relation for the u→v traversal
    int32_t rev_and_flag;  // (v→u oriented relation << 1) | reverse bit
};

inline int32_t pack_rev(int32_t rev, bool rev_bit) {
    return (rev << 1) | (rev_bit ? 1 : 0);
}
inline int32_t unpack_rev(int32_t packed) { return packed >> 1; }
inline uint8_t unpack_flag(int32_t packed) {
    return static_cast<uint8_t>(packed & 1);
}

// Flat epoch-stamped BFS arena. reset() is O(1); visited(v) compares stamps,
// so nothing is cleared between searches and nothing is allocated per probe.
struct Arena {
    int32_t n = 0;
    vector<uint32_t> stamp;
    vector<int32_t> parent;  // parent[v] == v marks the root
    vector<int32_t> depth;
    vector<int32_t> etype;   // oriented relation for the parent→v hop
    vector<uint8_t> dflag;
    vector<int32_t> cur, nxt;
    uint32_t epoch = 0;

    void init(int32_t nodes) {
        n = nodes;
        stamp.assign(n, 0);
        parent.assign(n, 0);
        depth.assign(n, 0);
        etype.assign(n, -1);
        dflag.assign(n, 0);
        epoch = 0;
        cur.reserve(4096);
        nxt.reserve(4096);
    }
    inline void reset() { ++epoch; }
    inline bool visited(int32_t v) const { return stamp[v] == epoch; }
    inline void visit(int32_t v, int32_t par, int32_t et, int32_t dp,
                      uint8_t fl) {
        stamp[v] = epoch;
        parent[v] = par;
        etype[v] = et;
        depth[v] = dp;
        dflag[v] = fl;
    }
};

struct Graph {
    int32_t n = 0;             // node id upper bound (max id + 1)
    vector<int32_t> offsets;   // n + 1
    vector<PackedEdge> edges;  // sorted, deduped per slice
    vector<int8_t> types;      // n, 0 = untyped default
    int32_t max_rel = -1;

    inline void slice(int32_t u, int64_t &eb, int64_t &ee) const {
        if (u < n) {
            eb = offsets[u];
            ee = offsets[u + 1];
        } else {
            eb = offsets[n];
            ee = offsets[n];
        }
    }
};

struct Query {
    int64_t row_index;
    int64_t query_id;
    int32_t head_id;
    int32_t tail_id;
    int32_t rel_id;
    int32_t inv_rel_id;
};

struct GroupRun {
    size_t lo, hi;  // [lo, hi) row range of one (query_id, head) run
};

// ── parsing ──────────────────────────────────────────────────────────────

// Splits [lb, le) on ',' and parses up to six integer fields. Mirrors the
// previous loader: every present field must parse as an integer, otherwise
// the line is a parse failure (the caller warns and skips). `min_fields`
// guards the minimum column count (graph rows: 5; query rows: 4).
static bool parse_fields(const char *lb, const char *le, int32_t out[6],
                         int &nfields, int min_fields) {
    bool ok[6] = {false, false, false, false, false, false};
    nfields = 0;
    const char *p = lb;
    while (nfields < 6) {
        const char *sep = p;
        while (sep < le && *sep != ',') ++sep;
        ok[nfields] = parse_int(p, sep, out[nfields]);
        ++nfields;
        if (sep >= le) break;
        p = sep + 1;
    }
    if (nfields < min_fields) return false;
    for (int i = 0; i < nfields; ++i) {
        if (!ok[i]) return false;
    }
    return true;
}

static void push_query(vector<Query> &queries, vector<GroupRun> &groups,
                       const Query &q) {
    if (groups.empty()
            || queries[groups.back().lo].query_id != q.query_id
            || queries[groups.back().lo].head_id != q.head_id) {
        if (!groups.empty()) groups.back().hi = queries.size();
        groups.push_back({queries.size(), queries.size()});
    }
    queries.push_back(q);
}

static bool load_graph(const string &path, ofstream &log, Graph &g) {
    LineReader reader(path);
    if (!reader.good()) {
        log << "[Error] Cannot open graph CSV: " << path << endl;
        return false;
    }

    // Pass 1: bounds only.
    int32_t max_node = -1;
    int32_t max_rel = -1;
    {
        const char *lb, *le;
        int32_t f[6];
        int nf;
        while (reader.next_line(lb, le)) {
            if (!parse_fields(lb, le, f, nf, 5)) continue;
            if (f[0] > max_node) max_node = f[0];
            if (f[1] > max_node) max_node = f[1];
            if (f[4] > max_rel) max_rel = f[4];
            if (nf >= 6 && f[5] > max_rel) max_rel = f[5];
        }
    }
    if (max_node < 0) {
        log << "[Warn] Graph CSV contains no usable edges: " << path << endl;
        g.n = 0;
        g.offsets.assign(1, 0);
        g.max_rel = max_rel;
        return true;
    }
    const int32_t nodes = max_node + 1;

    // Pass 2: degrees + first-seen types in file order.
    vector<int32_t> degree(nodes + 1, 0);
    vector<uint8_t> typed(nodes, 0);
    g.types.assign(nodes, 0);
    int32_t typed_count = 0;
    reader.rewind_to_data();
    {
        const char *lb, *le;
        int32_t f[6];
        int nf;
        while (reader.next_line(lb, le)) {
            if (!parse_fields(lb, le, f, nf, 5)) continue;
            ++degree[f[0]];
            ++degree[f[1]];
            if (!typed[f[0]]) {
                typed[f[0]] = 1;
                g.types[f[0]] = static_cast<int8_t>(f[2]);
                ++typed_count;
            }
            if (!typed[f[1]]) {
                typed[f[1]] = 1;
                g.types[f[1]] = static_cast<int8_t>(f[3]);
                ++typed_count;
            }
        }
    }

    // Prefix sums, fill (pass 3), sort + dedupe per slice (compacted in
    // place; the write cursor never passes the read cursor).
    g.offsets.assign(nodes + 1, 0);
    {
        int32_t total = 0;
        for (int32_t i = 0; i < nodes; ++i) {
            g.offsets[i] = total;
            total += degree[i];
        }
        g.offsets[nodes] = total;
    }
    vector<PackedEdge> edges(static_cast<size_t>(g.offsets[nodes]));
    {
        vector<int32_t> cursor(g.offsets.begin(), g.offsets.end() - 1);
        reader.rewind_to_data();
        const char *lb, *le;
        int32_t f[6];
        int nf;
        while (reader.next_line(lb, le)) {
            if (!parse_fields(lb, le, f, nf, 5)) continue;
            edges[cursor[f[0]]++] = {f[1], f[4], pack_rev(f[5], false)};
            edges[cursor[f[1]]++] = {f[0], f[5], pack_rev(f[4], true)};
        }
    }
    vector<int32_t> new_offsets(nodes + 1);
    int64_t wpos = 0;
    for (int32_t u = 0; u < nodes; ++u) {
        const int64_t eb = g.offsets[u];
        const int64_t ee = g.offsets[u + 1];
        new_offsets[u] = static_cast<int32_t>(wpos);
        sort(edges.begin() + eb, edges.begin() + ee,
             [](const PackedEdge &a, const PackedEdge &b) {
                 if (a.v != b.v) return a.v < b.v;
                 if (a.e_type != b.e_type) return a.e_type < b.e_type;
                 return a.rev_and_flag < b.rev_and_flag;
             });
        int64_t w = wpos;
        for (int64_t r = eb; r < ee; ++r) {
            const PackedEdge &e = edges[r];
            if (w > new_offsets[u]) {
                const PackedEdge &prev = edges[w - 1];
                if (prev.v == e.v && prev.e_type == e.e_type
                        && prev.rev_and_flag == e.rev_and_flag) {
                    continue;
                }
            }
            edges[w++] = e;
        }
        wpos = w;
    }
    new_offsets[nodes] = static_cast<int32_t>(wpos);
    edges.resize(static_cast<size_t>(wpos));
    edges.shrink_to_fit();
    g.edges = std::move(edges);
    g.offsets.swap(new_offsets);
    g.n = nodes;
    g.max_rel = max_rel;
    log << "[Info] Graph parsed: " << nodes << " nodes, " << typed_count
        << " typed, " << g.offsets[nodes] << " directed edges (deduped)"
        << endl;
    return true;
}

static bool load_queries(const string &path, ofstream &log,
                         vector<Query> &queries, vector<GroupRun> &groups) {
    LineReader reader(path);
    if (!reader.good()) {
        log << "[Error] Cannot open queries CSV: " << path << endl;
        return false;
    }
    const char *lb, *le;
    while (reader.next_line(lb, le)) {
        if (le <= lb) continue;
        int32_t f[6];
        int nfields = 0;
        if (!parse_fields(lb, le, f, nfields, 4)) {
            log << "[Warn] Query line skipped (malformed)" << endl;
            continue;
        }
        Query q;
        q.row_index = f[0];
        q.query_id  = f[1];
        q.head_id   = f[2];
        q.tail_id   = f[3];
        // Older queries files without the relation columns keep the own-edge
        // mask inert (rel id -1 matches no edge type).
        q.rel_id     = (nfields > 4) ? f[4] : -1;
        q.inv_rel_id = (nfields > 5) ? f[5] : -1;
        push_query(queries, groups, q);
    }
    if (!groups.empty()) groups.back().hi = queries.size();
    log << "[Info] Queries loaded: " << queries.size() << " rows, "
        << groups.size() << " groups" << endl;
    return true;
}

// ── BFS on arenas ────────────────────────────────────────────────────────

// Level-synchronized BFS from `root` into arena `a`. Masking applies to the
// given probe (own-edge + target-deny); `flipped` marks a backward search,
// whose parent links record the opposite (reconstruction) orientation. When
// `meet` is non-null (backward search against the forward tree), expansion
// stops at the first newly added node that is already forward-visited with
// a combined depth within max_hops; the backward parent link for that hop
// is recorded before returning, exactly as the previous loader did.
static void bfs_levels(const Graph &g, Arena &a, int32_t root, int max_hops,
                       const vector<int32_t> &fanout_limit,
                       const vector<uint8_t> &target_denied,
                       bool mask_own_edges, const Query &probe, bool flipped,
                       const Arena *meet, int32_t *meet_node) {
    a.reset();
    a.visit(root, root, -1, 0, 0);
    a.cur.clear();
    a.cur.push_back(root);
    for (int depth = 0; depth < max_hops && !a.cur.empty(); ++depth) {
        const int32_t limit = fanout_limit[depth];
        a.nxt.clear();
        for (int32_t u : a.cur) {
            if (u >= g.n) continue;
            int added = 0;
            int64_t eb, ee;
            g.slice(u, eb, ee);
            for (int64_t ei = eb; ei < ee && added < limit; ++ei) {
                const PackedEdge &e = g.edges[ei];
                if (a.visited(e.v)) continue;
                if (mask_own_edges) {
                    const int32_t et = e.e_type;
                    if ((et == probe.rel_id || et == probe.inv_rel_id)
                            && ((u == probe.head_id && e.v == probe.tail_id)
                                || (u == probe.tail_id
                                    && e.v == probe.head_id))) {
                        continue;
                    }
                }
                if (!target_denied.empty() && target_denied[e.e_type]) {
                    continue;
                }
                const bool is_meet =
                    meet_node != nullptr && meet->visited(e.v)
                    && meet->depth[e.v] + depth + 1 <= max_hops;
                if (flipped) {
                    // The search moves u→v, but reconstruction needs the
                    // opposite segment v→u: relation and direction flip.
                    a.visit(e.v, u, unpack_rev(e.rev_and_flag), depth + 1,
                            static_cast<uint8_t>(
                                1 - unpack_flag(e.rev_and_flag)));
                } else {
                    a.visit(e.v, u, e.e_type, depth + 1,
                            unpack_flag(e.rev_and_flag));
                }
                if (is_meet) {
                    *meet_node = e.v;
                    return;
                }
                a.nxt.push_back(e.v);
                ++added;
            }
        }
        a.cur.swap(a.nxt);
    }
}

// Whether an own edge exists between head and tail. When it does not, the
// mask can never fire for this probe and the shared forward tree is valid
// as-is. Slices are sorted by neighbour id, so the candidate span is found
// by binary search.
static bool has_own_edge(const Graph &g, const Query &q) {
    if (q.rel_id < 0 && q.inv_rel_id < 0) return false;
    auto side_has = [&](int32_t u, int32_t w) {
        if (u >= g.n) return false;
        int64_t eb, ee;
        g.slice(u, eb, ee);
        int64_t lo = eb, hi = ee;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            if (g.edges[mid].v < w) lo = mid + 1; else hi = mid;
        }
        for (int64_t ei = lo; ei < ee; ++ei) {
            const PackedEdge &e = g.edges[ei];
            if (e.v > w) break;
            if (e.v == w && (e.e_type == q.rel_id
                             || e.e_type == q.inv_rel_id)) {
                return true;
            }
        }
        return false;
    };
    return side_has(q.head_id, q.tail_id) || side_has(q.tail_id, q.head_id);
}

// ── per-thread query processing ──────────────────────────────────────────

static void run_thread(int tid, int num_threads, const Graph &g,
                      const vector<Query> &queries,
                      const vector<GroupRun> &groups, size_t group_lo,
                      size_t group_hi, int max_hops,
                      const vector<int32_t> &fanout_limit,
                      const vector<uint8_t> &target_denied,
                      bool mask_own_edges, bool allow_target_traversal,
                      const string &output_prefix, const string &log_dir,
                      int &rc_out) {
    const string log_name =
        log_dir + "/_" + to_string(tid) + "_preprocess.log";
    ofstream log(log_name, ios_base::app);
    if (!log.is_open()) {
        cerr << "Failed to open log: " << log_name << endl;
        rc_out = 1;
        return;
    }
    const string out_name =
        output_prefix + "_paths_" + to_string(tid) + ".txt";
    ofstream out(out_name, ios_base::binary);
    if (!out.is_open()) {
        log << "[Error] Cannot open output: " << out_name << endl;
        rc_out = 1;
        return;
    }

    int64_t my_total = 0;
    for (size_t gi = group_lo; gi < group_hi; ++gi) {
        my_total += static_cast<int64_t>(groups[gi].hi - groups[gi].lo);
    }
    log << "[Info] Thread " << tid << "/" << num_threads
        << "  max_hops=" << max_hops << "  mask_own_edges=" << mask_own_edges
        << "  allow_target_traversal=" << allow_target_traversal
        << "  my_probes=" << my_total << endl;

    int64_t processed = 0, found = 0, missing = 0, self = 0;
    int last_pct = -1;
    string buf;
    buf.reserve(1u << 20);

    auto log_progress = [&]() {
        ++processed;
        if (my_total > 0) {
            int pct = static_cast<int>(100.0 * processed / my_total);
            if (pct > last_pct) {
                last_pct = pct;
                log << "[Progress] " << pct << "%  processed=" << processed
                    << "  found=" << found << "  missing=" << missing
                    << "  (self-probes=" << self << ")" << endl;
                log.flush();
            }
        }
    };

    auto emit_no_path = [&](const Query &q) {
        buf += to_string(q.query_id);
        buf += ';';
        buf += to_string(q.row_index);
        buf += ";-1;;;;\n";
        ++missing;
        log_progress();
    };

    auto emit_path = [&](const Query &q, const vector<int32_t> &path_nodes,
                         const vector<int32_t> &edge_types,
                         const vector<int32_t> &edge_directions) {
        const int64_t hops = static_cast<int64_t>(path_nodes.size()) - 1;
        buf += to_string(q.query_id);
        buf += ';';
        buf += to_string(q.row_index);
        buf += ';';
        buf += to_string(hops);
        buf += ';';
        for (size_t i = 0; i < path_nodes.size(); ++i) {
            buf += to_string(path_nodes[i]);
            if (i + 1 < path_nodes.size()) buf += ',';
        }
        buf += ';';
        for (size_t i = 0; i < path_nodes.size(); ++i) {
            const int32_t node = path_nodes[i];
            buf += to_string(node < g.n ? g.types[node] : 0);
            if (i + 1 < path_nodes.size()) buf += ',';
        }
        buf += ';';
        for (size_t i = 0; i < edge_types.size(); ++i) {
            buf += to_string(edge_types[i]);
            if (i + 1 < edge_types.size()) buf += ',';
        }
        buf += ';';
        for (size_t i = 0; i < edge_directions.size(); ++i) {
            buf += to_string(edge_directions[i]);
            if (i + 1 < edge_directions.size()) buf += ',';
        }
        buf += '\n';
        if (buf.size() >= (1u << 20)) {
            out.write(buf.data(), static_cast<streamsize>(buf.size()));
            buf.clear();
        }
        ++found;
        log_progress();
    };

    Arena fwd, bwd, probe_fwd;
    fwd.init(g.n);
    bwd.init(g.n);
    probe_fwd.init(g.n);

    vector<int32_t> path_nodes, edge_types, edge_directions;
    size_t warned_oob = 0;

    for (size_t gi = group_lo; gi < group_hi; ++gi) {
        const GroupRun &run = groups[gi];
        const Query &head_probe = queries[run.lo];

        // ── shared forward BFS from the query head (no own-edge mask) ────
        bfs_levels(g, fwd, head_probe.head_id, max_hops, fanout_limit,
                   target_denied, false, head_probe, false, nullptr, nullptr);

        for (size_t j = run.lo; j < run.hi; ++j) {
            const Query &q = queries[j];

            if (q.head_id == q.tail_id) {
                emit_no_path(q);
                ++self;
                continue;
            }
            if (q.head_id >= g.n || q.tail_id >= g.n) {
                if (warned_oob++ == 0) {
                    log << "[Warn] Query references a node beyond the graph "
                        << "node range; emitting no-path for such probes"
                        << endl;
                }
                emit_no_path(q);
                continue;
            }

            // A probe whose own edge is in the graph needs a masked forward
            // state; every other probe reuses the shared unmasked tree.
            const bool needs_masked_fwd = mask_own_edges && has_own_edge(g, q);
            if (needs_masked_fwd) {
                bfs_levels(g, probe_fwd, q.head_id, max_hops, fanout_limit,
                           target_denied, true, q, false, nullptr, nullptr);
            }
            const Arena &fparent = needs_masked_fwd ? probe_fwd : fwd;

            int32_t meet_node = -1;
            if (fparent.visited(q.tail_id)) {
                meet_node = q.tail_id;
            } else {
                bfs_levels(g, bwd, q.tail_id, max_hops, fanout_limit,
                           target_denied, true, q, true, &fparent,
                           &meet_node);
            }

            if (meet_node < 0) {
                emit_no_path(q);
                continue;
            }

            // ── reconstruction ───────────────────────────────────────────
            // head→meet segment via the forward tree (reversed walk), then
            // meet→tail via the backward tree.
            path_nodes.clear();
            edge_types.clear();
            edge_directions.clear();
            {
                int32_t walk = meet_node;
                while (fparent.parent[walk] != walk) {
                    path_nodes.push_back(walk);
                    edge_types.push_back(fparent.etype[walk]);
                    edge_directions.push_back(fparent.dflag[walk]);
                    walk = fparent.parent[walk];
                }
                path_nodes.push_back(walk);  // the head (root link)
                reverse(path_nodes.begin(), path_nodes.end());
                reverse(edge_types.begin(), edge_types.end());
                reverse(edge_directions.begin(), edge_directions.end());
            }
            {
                int32_t cur = meet_node;
                while (cur != q.tail_id) {
                    path_nodes.push_back(bwd.parent[cur]);
                    edge_types.push_back(bwd.etype[cur]);
                    edge_directions.push_back(bwd.dflag[cur]);
                    cur = bwd.parent[cur];
                }
            }
            emit_path(q, path_nodes, edge_types, edge_directions);
        }
    }
    out.write(buf.data(), static_cast<streamsize>(buf.size()));
    buf.clear();
    out.close();

    log << "[Info] Done — found=" << found << "  unreachable=" << missing
        << "  (self-probes=" << self << ")  total=" << (found + missing)
        << endl;
    log.close();
    rc_out = 0;
}

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

    const string graph_csv     = argv[1];
    const string queries_csv   = argv[2];
    const string output_prefix = argv[3];
    const int    max_hops      = stoi(argv[4]);
    const int    max_fanout    = stoi(argv[5]);
    const float  decay_factor  = stof(argv[6]);
    const int    worker_arg    = stoi(argv[7]);  // vestigial invocation arg
    int          num_threads   = stoi(argv[8]);
    const string log_dir       = argv[9];

    const bool mask_own_edges = argc > 10 ? stoi(argv[10]) != 0 : false;
    const bool allow_target_traversal =
        argc > 11 ? stoi(argv[11]) != 0 : true;
    vector<int32_t> target_ids;
    if (argc > 12) {
        for (const auto &part : split(argv[12], ',')) {
            if (!part.empty()) target_ids.push_back(stoi(part));
        }
    }
    if (num_threads < 1) num_threads = 1;

    // Process-level log: thread 0's file hosts the load/summary lines.
    const string main_log_name =
        log_dir + "/_0_preprocess.log";
    ofstream main_log(main_log_name, ios_base::app);
    if (!main_log.is_open()) {
        cerr << "Failed to open log: " << main_log_name << endl;
        return 1;
    }
    main_log << "[Info] Worker v2 process  max_hops=" << max_hops
             << "  max_fanout=" << max_fanout << "  decay=" << decay_factor
             << "  internal_threads=" << num_threads
             << "  invocation_thread_id=" << worker_arg
             << "  mask_own_edges=" << mask_own_edges
             << "  allow_target_traversal=" << allow_target_traversal
             << "  target_ids=" << target_ids.size() << endl;

    Graph g;
    if (!load_graph(graph_csv, main_log, g)) {
        return 1;
    }

    vector<Query> queries;
    vector<GroupRun> groups;
    if (!load_queries(queries_csv, main_log, queries, groups)) {
        return 1;
    }

    // Uniform target-traversal deny table, sized to cover every relation id
    // the worker may see (graph edges + query relation fields).
    vector<uint8_t> target_denied;
    if (!allow_target_traversal && !target_ids.empty()) {
        int32_t max_rel = g.max_rel;
        for (const Query &q : queries) {
            if (q.rel_id > max_rel) max_rel = q.rel_id;
            if (q.inv_rel_id > max_rel) max_rel = q.inv_rel_id;
        }
        target_denied.assign(static_cast<size_t>(max_rel) + 1, 0);
        for (const int32_t id : target_ids) {
            if (id >= 0 && static_cast<size_t>(id) < target_denied.size()) {
                target_denied[static_cast<size_t>(id)] = 1;
            }
        }
        main_log << "[Info] Target-traversal denied for "
                 << target_denied.size() << "-sized relation table ("
                 << target_ids.size() << " target ids)" << endl;
    }

    // Fanout limits per depth (computed once; identical to per-BFS
    // ceil(max_fanout * decay^depth) in the previous loader).
    vector<int32_t> fanout_limit(max_hops);
    for (int depth = 0; depth < max_hops; ++depth) {
        fanout_limit[depth] = static_cast<int32_t>(
            ceil(static_cast<double>(max_fanout)
                 * pow(decay_factor, depth)));
    }

    // Partition query groups into contiguous static ranges.
    const size_t n_groups = groups.size();
    const size_t per_thread = (n_groups + static_cast<size_t>(num_threads) - 1)
                              / static_cast<size_t>(num_threads);
    vector<int> rc(num_threads, -1);
    vector<thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        const size_t lo = static_cast<size_t>(t) * per_thread;
        const size_t hi = min((static_cast<size_t>(t) + 1) * per_thread,
                              n_groups);
        if (lo >= hi) {
            rc[t] = 0;
            continue;
        }
        threads.emplace_back(run_thread, t, num_threads, cref(g),
                             cref(queries), cref(groups), lo, hi, max_hops,
                             cref(fanout_limit), cref(target_denied),
                             mask_own_edges, allow_target_traversal,
                             output_prefix, log_dir, ref(rc[t]));
    }
    for (auto &th : threads) th.join();

    for (int t = 0; t < num_threads; ++t) {
        if (rc[t] != 0) {
            main_log << "[Error] Thread " << t << " failed (rc=" << rc[t]
                     << ")" << endl;
            return 1;
        }
    }
    main_log << "[Info] Worker v2 complete: threads=" << num_threads << endl;
    return 0;
}
