"""Journal v2 statistics: paired bootstrap CIs (strict + non-strict).

Per-query reciprocal ranks for the PAMR family (5 seeds), the four
baselines (1 or 3 seeds), the presence control and the length heuristic,
on the full and all-path subsets, K=10 and K<=100. Outputs JSON + CSVs.

Usage:
    python -m scripts.stats_v2 --v1-root ... --v2-root ... --out-dir ... \
        [--baseline-seeds 42 1 2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = [1, 3, 5, 10]
PAMR_CONFIGS = {
    "PAMR-listwise": "campaign",
    "PAMR-cls": "cls",
    "PAMR-no-feats": "no_feats",
    "PAMR-lenonly": "lenonly",
    "PAR": "par",
}
BASELINES = ["TransE", "RotatE", "DistMult", "ComplEx"]
BOOT = 10000


def _protocol_dirs(v1, v2, protocol):
    if protocol == "k10":
        return {
            "queries": v1 / "test_queries.csv",
            "paths": v2 / "random" / "paths" / "test_paths.csv",
            "base_scores": v1 / "full" / "vlkge",
        }
    k100 = v2 / "k100"
    return {
        "queries": k100 / "queries" / "test_queries.csv",
        "paths": k100 / "paths_prod" / "test_paths.csv",
        "base_scores": k100 / "full" / "vlkge",
    }


def _frame(queries_csv, paths_csv):
    q = pd.read_csv(queries_csv).reset_index(drop=True)
    q["row_index"] = range(len(q))
    p = pd.read_csv(paths_csv)[["row_index", "hops"]]
    d = q.merge(p, on="row_index")
    d["candidate_tail_id"] = d["tail_id"]
    d["has_path"] = d["hops"] >= 0
    rel = (d[d["is_positive"] == 1]
           .drop_duplicates("query_id").set_index("query_id")["relation"])
    return d, rel.to_dict()


def _pamr_scores(run_dir, protocol):
    name = "test_scores.csv" if protocol == "k10" else "test_k100_scores.csv"
    p = Path(run_dir) / "paths" / name
    if not p.exists():
        return None
    return pd.read_csv(p)[["query_id", "candidate_tail_id", "score"]]


def _base_scores(base_dir, model, seed, protocol):
    if protocol == "k10":
        p = (base_dir / model / "test_scores.csv" if seed == 42
             else base_dir.parent / "seed_scores" / f"{model}_s{seed}"
             / "test_scores.csv")
    else:
        p = (base_dir / model / "test_scores.csv" if seed == 42
             else base_dir.parent / "seed_scores" / f"{model}_s{seed}"
             / "test_scores.csv")
    return pd.read_csv(p)["score"].to_numpy() if p.exists() else None


def _method_scores(frame, scores):
    if scores is None:
        return None
    if isinstance(scores, pd.DataFrame):
        m = frame.merge(scores, on=["query_id", "candidate_tail_id"],
                        how="left", validate="one_to_one")
        if m["score"].isna().any():
            raise ValueError("PAMR scores do not cover all candidates")
        return m["score"].to_numpy()
    if len(scores) != len(frame):
        raise ValueError("baseline scores length mismatch")
    return scores


def _recip_ranks(frame, scores, all_path_ids=None, rel_map=None):
    out = {"strict": {}, "ns": {}, "rel": {}}
    d = frame.copy()
    d["score"] = scores
    for qid, g in d.groupby("query_id"):
        if all_path_ids is not None and qid not in all_path_ids:
            continue
        pos = g.loc[g["is_positive"] == 1, "score"].max()
        out["strict"][int(qid)] = 1.0 / (int((g["score"] > pos).sum()) + 1)
        out["ns"][int(qid)] = 1.0 / max(1, int((g["score"] >= pos).sum()))
        out["rel"][int(qid)] = rel_map.get(int(qid), "")
    return out


def _mean(rank_map):
    if not rank_map:
        return 0.0
    return float(np.mean(list(rank_map.values())))


def _hits_at(rank_map, k):
    if not rank_map:
        return 0.0
    return float(np.mean([(1.0 / r) >= (1.0 / k) for r in rank_map.values()]))


def _bootstrap(a, b, n=BOOT, seed=0):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    diffs = np.array([a[q] - b[q] for q in ids])
    rng = np.random.default_rng(seed)
    means = rng.choice(diffs, size=(n, len(diffs)), replace=True).mean(axis=1)
    return {"mean_diff": round(float(diffs.mean()), 6),
            "ci95": [round(float(np.percentile(means, 2.5)), 6),
                     round(float(np.percentile(means, 97.5)), 6)],
            "n_queries": len(ids)}


def _seed_mean(rank_dicts, key):
    ids = set(rank_dicts[0][key])
    for d in rank_dicts[1:]:
        ids &= set(d[key])
    return {q: float(np.mean([d[key][q] for d in rank_dicts])) for q in ids}


def _metrics(rank_map, rels=None, split_rel=None):
    mrr_s = _mean(rank_map["strict"])
    mrr_ns = _mean(rank_map["ns"])
    out = {"MRR": round(mrr_s, 6), "MRR_ns": round(mrr_ns, 6)}
    for k in TOP_K:
        out[f"H{k}"] = round(_hits_at(rank_map["strict"], k), 6)
        out[f"H{k}_ns"] = round(_hits_at(rank_map["ns"], k), 6)
    if rels is not None:
        per = {}
        for rel in rels:
            ids = {q for q, r in rank_map["rel"].items() if r == rel}
            sub = {"strict": {q: rank_map["strict"][q] for q in ids},
                   "ns": {q: rank_map["ns"][q] for q in ids},
                   "rel": {q: rel for q in ids}}
            per[rel] = _metrics(sub)
        out["per_relation"] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--baseline-seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    v1 = Path(args.v1_root)
    v2 = Path(args.v2_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rels = ["isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf"]

    for protocol in ("k10", "k100"):
        d = _protocol_dirs(v1, v2, protocol)
        frame, rel_map = _frame(d["queries"], d["paths"])
        all_path = frame.groupby("query_id")["has_path"].all()
        ap_ids = set(all_path[all_path].index.tolist())
        report = {"protocol": protocol,
                  "n_queries": int(frame["query_id"].nunique()),
                  "n_all_path": len(ap_ids),
                  "methods": {}, "baselines": {},
                  "comparisons": {}, "controls": {}}

        # baselines: per-seed ranks -> seed-mean
        base_rank = {}
        for model in BASELINES:
            per_seed = []
            for seed in args.baseline_seeds:
                sc = _base_scores(d["base_scores"], model, seed, protocol)
                if sc is None:
                    continue
                per_seed.append(_recip_ranks(
                    frame, _method_scores(frame, sc),
                    rel_map=rel_map))
            if not per_seed:
                continue
            base_rank[model] = _seed_mean(per_seed, "ns")
            base_rank[model + "_strict"] = _seed_mean(per_seed, "strict")
            # full/all-path metrics on the seed-mean
            full = {"strict": base_rank[model + "_strict"],
                    "ns": base_rank[model],
                    "rel": per_seed[0]["rel"]}
            sub_ids = set(full["ns"]) & ap_ids
            sub = {"strict": {q: full["strict"][q] for q in sub_ids},
                   "ns": {q: full["ns"][q] for q in sub_ids},
                   "rel": {q: full["rel"][q] for q in sub_ids}}
            mrr_seeds = [_mean(dd["ns"]) for dd in per_seed]
            report["baselines"][model] = {
                "seeds": args.baseline_seeds,
                "MRR_ns_std": round(float(np.std(mrr_seeds, ddof=1))
                                    if len(mrr_seeds) > 1 else 0.0, 6),
                "full": _metrics(full, rels),
                "all_path": _metrics(sub, rels),
            }

        # PAMR family: per-config per-seed ranks
        method_rank = {}
        for name, cfg in PAMR_CONFIGS.items():
            per_seed = []
            for seed in args.seeds:
                run_dir = v2 / "campaign" / cfg / f"seed{seed}"
                sc = _pamr_scores(run_dir, protocol)
                if sc is None:
                    continue
                per_seed.append(_recip_ranks(
                    frame, _method_scores(frame, sc),
                    rel_map=rel_map))
            if not per_seed:
                continue
            ns = _seed_mean(per_seed, "ns")
            st = _seed_mean(per_seed, "strict")
            method_rank[name] = ns
            full = {"strict": st, "ns": ns, "rel": per_seed[0]["rel"]}
            sub_ids = set(ns) & ap_ids
            sub = {"strict": {q: st[q] for q in sub_ids},
                   "ns": {q: ns[q] for q in sub_ids},
                   "rel": {q: full["rel"][q] for q in sub_ids}}
            mrr_seeds = [_mean(dd["ns"]) for dd in per_seed]
            report["methods"][name] = {
                "seeds": args.seeds,
                "MRR_ns_std": round(float(np.std(mrr_seeds, ddof=1))
                                    if len(mrr_seeds) > 1 else 0.0, 6),
                "full": _metrics(full, rels),
                "all_path": _metrics(sub, rels),
            }

        # controls
        lengths = frame.copy()
        lengths["score"] = np.where(lengths["has_path"],
                                    -lengths["hops"].astype(float), -1e6)
        len_rank = _recip_ranks(frame, lengths["score"].to_numpy(),
                                rel_map=rel_map)
        report["controls"]["length_only"] = {
            "full": _metrics(len_rank),
            "all_path": _metrics({k: {q: v[q] for q in set(v) & ap_ids}
                                  for k, v in len_rank.items()}),
        }
        bb = _base_scores(d["base_scores"], "TransE", 42, protocol)
        pres_rank = None
        if bb is not None:
            pres = np.where(frame["has_path"], bb + 1e6, bb)
            pres_rank = _recip_ranks(frame, pres, rel_map=rel_map)
            report["controls"]["presence_control"] = {
                "full": _metrics(pres_rank),
                "all_path": _metrics({k: {q: v[q] for q in set(v) & ap_ids}
                                      for k, v in pres_rank.items()}),
            }

        # comparisons: every PAMR config vs every baseline + controls
        for name, ns in method_rank.items():
            comp = {}
            for model in BASELINES:
                if model not in base_rank:
                    continue
                b = base_rank[model]
                sub_ids = ap_ids
                comp[f"vs_{model}"] = {
                    "full": _bootstrap(ns, b),
                    "all_path": _bootstrap(
                        {q: ns[q] for q in ns if q in sub_ids},
                        {q: b[q] for q in b if q in sub_ids}),
                }
            lr = len_rank["ns"]
            comp["vs_length"] = {
                "full": _bootstrap(ns, lr),
                "all_path": _bootstrap(
                    {q: ns[q] for q in ns if q in ap_ids},
                    {q: lr[q] for q in lr if q in ap_ids}),
            }
            if pres_rank is not None:
                pr = pres_rank["ns"]
                comp["vs_presence"] = {
                    "full": _bootstrap(ns, pr),
                    "all_path": _bootstrap(
                        {q: ns[q] for q in ns if q in ap_ids},
                        {q: pr[q] for q in pr if q in ap_ids}),
                }
            if name in ("PAMR-listwise", "PAMR-cls"):
                for other in ("PAMR-no-feats", "PAR"):
                    if other in method_rank:
                        ob = method_rank[other]
                        comp[f"vs_{other}"] = {
                            "full": _bootstrap(ns, ob),
                            "all_path": _bootstrap(
                                {q: ns[q] for q in ns if q in ap_ids},
                                {q: ob[q] for q in ob if q in ap_ids}),
                        }
            report["comparisons"][name] = comp

        with open(out / f"stats_{protocol}.json", "w") as f:
            json.dump(report, f, indent=2)
        rows = []
        for name, m in report["methods"].items():
            rows.append({"method": name, "subset": "full",
                         "MRR_ns": m["full"]["MRR_ns"],
                         "MRR": m["full"]["MRR"],
                         "std": m["MRR_ns_std"]})
        for name, b in report["baselines"].items():
            rows.append({"method": f"VLKGE-{name}", "subset": "full",
                         "MRR_ns": b["full"]["MRR_ns"],
                         "MRR": b["full"]["MRR"], "std": None})
        pd.DataFrame(rows).to_csv(out / f"stats_{protocol}.csv", index=False)
        print(f"ACCEPTANCE stats {protocol} methods={len(report['methods'])} "
              f"baselines={len(report['baselines'])} "
              f"n={report['n_queries']} all_path={report['n_all_path']}")


if __name__ == "__main__":
    main()
