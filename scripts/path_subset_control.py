"""Path-conditional metrics and the path-presence control.

Reads prepared queries, path hops, and per-candidate scores, then reports
strict MRR/Hits@K on:
  - all queries (full protocol)
  - the all-path subset (every candidate of the query has a path) — on this
    subset an additive path-presence constant is rank-invariant, so any
    difference is path-content signal
  - the complement (queries with at least one pathless candidate)

If a backbone score CSV is supplied, it also evaluates the presence control
`backbone + 1e6 * 1[has_path]` on the same subsets.

Score CSV schemas:
  PAMR      query_id, candidate_tail_id, score, is_positive
  VL-KGE    query_id, score, is_positive        (row order = prepared order)

Usage:
    python -m scripts.path_subset_control \
        --queries .../test_queries.csv --paths .../test_paths.csv \
        --scores .../full/pamr/test_scores.csv \
        --backbone-scores .../full/vlkge/TransE/test_scores.csv \
        --out .../path_control.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from pamr.ranking import compute_rank_metrics

TOP_K = [1, 3, 5, 10]


def _load_aligned(queries_csv, paths_csv, scores_csv, backbone_csv=None):
    q = pd.read_csv(queries_csv).reset_index(drop=True)
    q["row_index"] = range(len(q))
    paths = pd.read_csv(paths_csv)[["row_index", "hops"]]
    base = q.merge(paths, on="row_index")
    base["has_path"] = base["hops"] >= 0

    scores = pd.read_csv(scores_csv)
    if "candidate_tail_id" in scores.columns:
        m = base.merge(
            scores[["query_id", "candidate_tail_id", "score"]],
            left_on=["query_id", "tail_id"], right_on=["query_id", "candidate_tail_id"],
            how="left", validate="one_to_one")
        if m["score"].isna().any():
            raise ValueError("PAMR scores do not cover every prepared probe")
        base = m
    else:
        if len(scores) != len(base):
            raise ValueError(
                f"score rows {len(scores)} != probe rows {len(base)}")
        base["score"] = scores["score"].to_numpy()
    if backbone_csv is not None:
        b = pd.read_csv(backbone_csv)
        if "candidate_tail_id" in b.columns:
            merged = base.merge(
                b[["query_id", "candidate_tail_id", "score"]],
                left_on=["query_id", "tail_id"],
                right_on=["query_id", "candidate_tail_id"],
                suffixes=("", "_bb"), how="left", validate="one_to_one")
            if merged["score_bb"].isna().any():
                raise ValueError("backbone scores do not cover every probe")
            base["backbone_score"] = merged["score_bb"].to_numpy()
        else:
            if len(b) != len(base):
                raise ValueError(
                    f"backbone score rows {len(b)} != probe rows {len(base)}")
            base["backbone_score"] = b["score"].to_numpy()
    return base


def _metrics(frame):
    if len(frame) == 0:
        return {"num_queries": 0}
    m = compute_rank_metrics(
        frame["score"].to_numpy(), frame["is_positive"].to_numpy(),
        frame["query_id"].to_numpy(), top_k=TOP_K)
    return {
        "num_queries": int(m["num_queries"]),
        "MRR": round(float(m["MRR"]), 6),
        "Hits": {str(k): round(float(v), 6) for k, v in m["Hits"].items()},
    }


def _subsets(frame):
    all_path = frame.groupby("query_id")["has_path"].all()
    ap_ids = set(all_path[all_path].index.tolist())
    full_ids = set(frame["query_id"].unique().tolist())
    return ap_ids, full_ids - ap_ids


def _apply_control(frame):
    frame = frame.copy()
    frame["score"] = frame["backbone_score"] + 1e6 * frame["has_path"].astype(float)
    return frame


def _recip_ranks(frame, all_path_ids):
    out = {}
    for qid, g in frame.groupby("query_id"):
        if qid not in all_path_ids:
            continue
        pos = g.loc[g["is_positive"] == 1, "score"].max()
        out[int(qid)] = 1.0 / (int((g["score"] > pos).sum()) + 1)
    return out


def _bootstrap(a, b, n_boot=10000, seed=0):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    diffs = np.array([a[q] - b[q] for q in ids])
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return {
        "mean_diff": round(float(diffs.mean()), 6),
        "ci95": [round(float(np.percentile(means, 2.5)), 6),
                 round(float(np.percentile(means, 97.5)), 6)],
        "n_queries": len(ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--backbone-scores", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frame = _load_aligned(args.queries, args.paths, args.scores,
                          args.backbone_scores)
    ap_ids, comp_ids = _subsets(frame)
    report = {
        "num_queries_full": int(frame["query_id"].nunique()),
        "num_queries_all_path": len(ap_ids),
        "num_queries_complement": len(comp_ids),
        "model": {
            "full": _metrics(frame),
            "all_path": _metrics(frame[frame["query_id"].isin(ap_ids)]),
            "complement": _metrics(frame[frame["query_id"].isin(comp_ids)]),
        },
    }
    if args.backbone_scores is not None:
        backbone = frame.copy()
        backbone["score"] = frame["backbone_score"]
        ctrl = _apply_control(frame)
        report["backbone"] = {
            "full": _metrics(backbone),
            "all_path": _metrics(backbone[backbone["query_id"].isin(ap_ids)]),
            "complement": _metrics(
                backbone[backbone["query_id"].isin(comp_ids)]),
        }
        report["presence_control"] = {
            "full": _metrics(ctrl),
            "all_path": _metrics(ctrl[ctrl["query_id"].isin(ap_ids)]),
            "complement": _metrics(ctrl[ctrl["query_id"].isin(comp_ids)]),
        }
        model_r = _recip_ranks(frame, ap_ids)
        ctrl_r = _recip_ranks(ctrl, ap_ids)
        bb_r = _recip_ranks(backbone, ap_ids)
        report["bootstrap_all_path_vs_backbone"] = _bootstrap(model_r, bb_r)
        report["bootstrap_all_path_vs_control"] = _bootstrap(model_r, ctrl_r)
    print("ACCEPTANCE path-subset-control " + json.dumps(report))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
