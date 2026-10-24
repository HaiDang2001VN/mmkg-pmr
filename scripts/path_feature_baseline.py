"""Path-feature logistic baseline (non-transformer).

Builds scalar path features (hops, node-type fractions, edge-type counts,
direction fractions) from the v2 training paths, fits a logistic regression
on the train probes, and ranks the K=10 / K<=100 test candidates. Reports
full/all-path strict and non-strict MRR/Hits. CPU only.

Usage:
    python -m scripts.path_feature_baseline --v1-root ... --v2-root ... \
        --out-dir assets/report_v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RELATIONS = ["isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf"]
NUM_REL = 40
K = [1, 3, 5, 10]


def _parse_counts(series, width, prefix_shift=0):
    out = np.zeros((len(series), width), dtype=np.float32)
    for i, text in enumerate(series.fillna("")):
        if not text:
            continue
        for tok in str(text).split(","):
            try:
                v = int(float(tok))
            except ValueError:
                continue
            idx = v - prefix_shift
            if 0 <= idx < width:
                out[i, idx] += 1.0
    return out


def _features(df):
    raw = df["hops"].to_numpy(dtype=float)
    hops = np.where(raw >= 0, raw, 0.0)
    has_path = (raw >= 0).astype(float)
    valid = has_path
    n_nodes = np.where(raw >= 0, raw + 1.0, 0.0)
    n_edges = np.maximum(n_nodes - 1.0, 0.0)
    node = _parse_counts(df["node_types"], 4)
    node = node[:, 1:4]  # types 1..3
    edge = _parse_counts(df["edge_types"], NUM_REL)
    dirs = _parse_counts(df["edge_directions"], 2)
    den_n = np.maximum(n_nodes, 1)[:, None]
    den_e = np.maximum(n_edges, 1)[:, None]
    feats = np.concatenate([
        (hops / 5.0)[:, None],
        has_path[:, None],
        node / den_n,
        edge / den_e,
        dirs / den_e,
    ], axis=1)
    return feats.astype(np.float32)


def _frame(queries_csv, paths_csv):
    q = pd.read_csv(queries_csv).reset_index(drop=True)
    q["row_index"] = range(len(q))
    p = pd.read_csv(paths_csv)[
        ["row_index", "hops", "node_types", "edge_types", "edge_directions"]]
    d = q.merge(p, on="row_index")
    d["candidate_tail_id"] = d["tail_id"]
    d["has_path"] = d["hops"] >= 0
    rel = (d[d["is_positive"] == 1]
           .drop_duplicates("query_id").set_index("query_id")["relation"])
    return d, rel.to_dict()


def _ranks(frame, scores):
    d = frame.copy()
    d["score"] = scores
    strict, ns, rel = {}, {}, {}
    for qid, g in d.groupby("query_id"):
        pos = g.loc[g["is_positive"] == 1, "score"].max()
        strict[int(qid)] = 1.0 / (int((g["score"] > pos).sum()) + 1)
        ns[int(qid)] = 1.0 / max(1, int((g["score"] >= pos).sum()))
        rel[int(qid)] = frame.loc[frame["query_id"] == qid, "relation"].iloc[0]
    return strict, ns, rel


def _metrics(strict, ns, ids=None):
    ids = list(ns) if ids is None else [q for q in ids if q in ns]
    out = {"n": len(ids)}
    for tag, rk in (("strict", strict), ("ns", ns)):
        if not ids:
            out[tag] = {"MRR": 0.0}
            continue
        mrr = float(np.mean([rk[q] for q in ids]))
        out[tag] = {"MRR": round(mrr, 6),
                    **{f"H{k}": round(float(np.mean(
                        [rk[q] * k >= 1.0 for q in ids])), 6) for k in K}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    v1, v2 = Path(args.v1_root), Path(args.v2_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, _ = _frame(v2 / "random" / "queries" / "train_queries.csv",
                      v2 / "random" / "paths" / "train_paths.csv")
    X = _features(train)
    y = train["is_positive"].to_numpy()
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)
    print(f"ACCEPTANCE logistic-fit rows={len(X)} feats={X.shape[1]}")

    for protocol in ("k10", "k100"):
        if protocol == "k10":
            qcsv = v1 / "test_queries.csv"
            pcsv = v2 / "random" / "paths" / "test_paths.csv"
        else:
            qcsv = v2 / "k100" / "queries" / "test_queries.csv"
            pcsv = v2 / "k100" / "paths_prod" / "test_paths.csv"
        test, rel = _frame(qcsv, pcsv)
        prob = clf.predict_proba(scaler.transform(_features(test)))[:, 1]
        strict, ns, _ = _ranks(test, prob)
        all_path = test.groupby("query_id")["has_path"].all()
        ap_ids = set(all_path[all_path].index.tolist())
        report = {
            "protocol": protocol,
            "n_queries": int(test["query_id"].nunique()),
            "n_all_path": len(ap_ids),
            "full": _metrics(strict, ns),
            "all_path": _metrics(strict, ns, ap_ids),
            "per_relation": {},
        }
        for r in RELATIONS:
            ids = {q for q, rr in rel.items() if rr == r}
            report["per_relation"][r] = _metrics(strict, ns, ids)
        with open(out / f"logistic_{protocol}.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"ACCEPTANCE logistic {protocol} full_ns="
              f"{report['full']['ns']['MRR']:.4f} "
              f"all_path_ns={report['all_path']['ns']['MRR']:.4f}")


if __name__ == "__main__":
    main()
