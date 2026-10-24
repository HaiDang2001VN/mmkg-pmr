"""Journal v2 aggregation: non-strict (tie-aware) primary, strict alongside.

Reads the campaign run metrics (K=10 test + extra_tests.k100), the four
baseline metrics, and the path-control JSONs, then writes:

  main_k10.csv / main_k100.csv          method x {MRR, H1, H3, H5} (+std)
  ablations_k10.csv / ablations_k100.csv
  all_path_k10.csv / all_path_k100.csv  all-path subset (non-strict)
  controls.csv                          backbone/presence/length rows
  summary.json

Usage:
    python -m scripts.aggregate_v2 --v2-root ... --v1-root ... --out-dir ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RELATIONS = ["isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf"]
K = [1, 3, 5, 10]
MAIN_CONFIGS = [("PAMR-listwise", "campaign"), ("PAMR-cls", "cls"),
                ("PAR", "par"), ("PAMR-no-feats", "no_feats")]
ABLATIONS = [("finetune", "finetune"), ("depth12", "depth12"),
             ("cls", "cls"), ("lf_temp", "lf_temp"),
             ("hop3", "hop3"), ("no_mask", "no_mask"),
             ("lenonly", "lenonly")]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _hit_pamr(entry, k):
    return entry.get("Hits_ns", {}).get(str(k), 0.0)


def _hit_base(entry, k):
    return entry.get(f"Hits_ns@{k}", 0.0)


def _pamr_metrics(metrics_path, protocol):
    m = _load(metrics_path)
    test = m["test"] if protocol == "k10" else (
        (m.get("extra_tests") or {}).get("k100") or {})
    if not test:
        return None
    per = test.get("per_relation", {})
    return {
        "MRR_ns": test.get("MRR_ns"),
        "MRR": test.get("MRR"),
        "Hits_ns": {k: test.get("Hits_ns", {}).get(str(k)) for k in K},
        "Hits": {k: test.get("Hits", {}).get(str(k)) for k in K},
        "per_relation": {
            rel: {
                "MRR_ns": per[rel].get("MRR_ns"),
                "MRR": per[rel].get("MRR"),
                "Hits_ns": {k: _hit_pamr(per[rel], k) for k in K},
                "Hits": {k: per[rel].get("Hits", {}).get(str(k)) for k in K},
            } for rel in RELATIONS if rel in per
        },
    }


def _base_metrics(metrics_path, protocol):
    m = _load(metrics_path)
    test = m.get("test") or {}
    per = test.get("per_relation", {})
    if not test:
        return None
    return {
        "MRR_ns": test.get("MRR_ns"),
        "MRR": test.get("MRR"),
        "Hits_ns": {k: test.get("Hits_ns", {}).get(str(k)) for k in K},
        "Hits": {k: test.get("Hits", {}).get(str(k)) for k in K},
        "per_relation": {
            rel: {
                "MRR_ns": per[rel].get("MRR_ns"),
                "MRR": per[rel].get("MRR"),
                "Hits_ns": {k: _hit_base(per[rel], k) for k in K},
                "Hits": {k: per[rel].get(f"Hits@{k}") for k in K},
            } for rel in RELATIONS if rel in per
        },
    }


def _control(run_dir, protocol):
    for p in (Path(run_dir) / f"path_control_{protocol}.json",
              Path(run_dir) / "paths" / f"path_control_{protocol}.json"):
        if p.exists():
            return _load(p)
    return None


def _rows_for(config, seeds, v2, protocol, loader):
    vals = []
    for seed in seeds:
        p = (Path(v2) / "campaign" / config / f"seed{seed}" / "paths" /
             "pamr_training_metrics.json")
        if not p.exists():
            continue
        m = loader(str(p), protocol)
        if m is not None:
            vals.append(m)
    return vals


def _mean_std(vals, key, sub=None):
    xs = []
    for v in vals:
        if sub is None:
            x = v.get(key)
        else:
            x = (v.get("per_relation", {}).get(sub) or {}).get(key)
        if x is not None:
            xs.append(float(x))
    if not xs:
        return None, None
    return round(float(np.mean(xs)), 4), (
        round(float(np.std(xs, ddof=1)), 4) if len(xs) > 1 else 0.0)


def _hit_mean(vals, k, sub=None):
    xs = []
    for v in vals:
        if sub is None:
            h = v.get("Hits_ns", {}).get(k)
        else:
            h = ((v.get("per_relation", {}).get(sub) or {})
                 .get("Hits_ns", {}).get(k))
        if h is not None:
            xs.append(float(h))
    return round(float(np.mean(xs)), 4) if xs else None


def _row(name, vals, n_seeds):
    mrr, mrr_sd = _mean_std(vals, "MRR_ns")
    row = {"Method": name, "MRR_ns": mrr, "MRR_ns_std": mrr_sd,
           "n_seeds": n_seeds}
    for k in K:
        row[f"H{k}_ns"] = _hit_mean(vals, k)
    mrr_s, _ = _mean_std(vals, "MRR")
    row["MRR_strict"] = mrr_s
    return row


def _pr_row(name, vals, n_seeds):
    row = {"Method": name, "n_seeds": n_seeds}
    for rel in RELATIONS:
        mrr, sd = _mean_std(vals, "MRR_ns", sub=rel)
        row[f"{rel}_MRR_ns"] = mrr
        row[f"{rel}_MRR_ns_std"] = sd
        for k in K:
            row[f"{rel}_H{k}_ns"] = _hit_mean(vals, k, sub=rel)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    ap.add_argument("--baseline-seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    v2, v1 = Path(args.v2_root), Path(args.v1_root)
    k100 = v2 / "k100"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for protocol in ("k10", "k100"):
        rows, pr_rows, all_path_rows = [], [], []
        # baselines (seed 42 + optional extra seeds; mean/std over seeds)
        for model in ("TransE", "RotatE", "DistMult", "ComplEx"):
            vals, ctrls = [], []
            for seed in args.baseline_seeds:
                if protocol == "k10":
                    mp = (v1 / "full" / "vlkge" / model / "metrics.json"
                          if seed == 42 else
                          v1 / "full" / "seed_scores" / f"{model}_s{seed}"
                          / "metrics.json")
                    cp = (v1 / "full" / "vlkge" / model / "path_control.json"
                          if seed == 42 else
                          v1 / "full" / "seed_scores" / f"{model}_s{seed}"
                          / "path_control_k10.json")
                else:
                    mp = (k100 / "full" / "vlkge" / model / "metrics.json"
                          if seed == 42 else
                          k100 / "full" / "seed_scores" / f"{model}_s{seed}"
                          / "metrics.json")
                    cp = (k100 / "full" / "vlkge" / model / "path_control.json"
                          if seed == 42 else
                          k100 / "full" / "seed_scores" / f"{model}_s{seed}"
                          / "path_control_k100.json")
                if not mp.exists():
                    continue
                vals.append(_base_metrics(str(mp), protocol))
                if cp.exists():
                    ctrls.append(_load(cp)["model"]["all_path"])
            if not vals:
                continue
            rows.append(_row(f"VLKGE-{model}", vals, len(vals)))
            pr_rows.append(_pr_row(f"VLKGE-{model}", vals, len(vals)))
            if ctrls:
                all_path_rows.append({
                    "Method": f"VLKGE-{model}",
                    "all_path_MRR_ns": round(float(np.mean(
                        [x.get("MRR_ns") for x in ctrls])), 4),
                    "all_path_MRR_strict": round(float(np.mean(
                        [x.get("MRR") for x in ctrls])), 4),
                    "n_seeds": len(ctrls),
                })
        # PAMR / PAR / no-feats
        for name, cfg in MAIN_CONFIGS:
            vals = _rows_for(cfg, args.seeds, v2, protocol, _pamr_metrics)
            if not vals:
                continue
            rows.append(_row(name, vals, len(vals)))
            pr_rows.append(_pr_row(name, vals, len(vals)))
            ap_vals = []
            for seed in args.seeds:
                c = _control(v2 / "campaign" / cfg / f"seed{seed}", protocol)
                if c:
                    ap_vals.append(c["model"]["all_path"])
            if ap_vals:
                all_path_rows.append({
                    "Method": name,
                    "all_path_MRR_ns": round(float(np.mean(
                        [x.get("MRR_ns") for x in ap_vals])), 4),
                    "all_path_MRR_strict": round(float(np.mean(
                        [x.get("MRR") for x in ap_vals])), 4),
                    "n_seeds": len(ap_vals),
                })
        lgp = out / f"logistic_{protocol}.json"
        if lgp.exists():
            lg = _load(lgp)
            full = lg.get("full", {})
            ns, st = full.get("ns", {}), full.get("strict", {})
            rows.append({
                "Method": "Path-logistic", "MRR_ns": ns.get("MRR"),
                "MRR_ns_std": 0.0, "n_seeds": 1,
                **{f"H{k}_ns": ns.get(f"H{k}") for k in K},
                "MRR_strict": st.get("MRR"),
            })
            ap_l = (lg.get("all_path") or {}).get("ns", {})
            all_path_rows.append({
                "Method": "Path-logistic",
                "all_path_MRR_ns": ap_l.get("MRR"),
                "all_path_MRR_strict": (lg.get("all_path") or {})
                .get("strict", {}).get("MRR"),
            })
        pd.DataFrame(rows).to_csv(out / f"main_{protocol}.csv", index=False)
        pd.DataFrame(pr_rows).to_csv(out / f"per_relation_{protocol}.csv",
                                     index=False)

        # controls (protocol-level; from the main PAMR seed 42)
        ctrl = _control(v2 / "campaign" / "campaign" / "seed42", protocol)
        if ctrl:
            for key, label in (("backbone", "Backbone"),
                               ("presence_control", "Presence control"),
                               ("length_only", "Length heuristic")):
                if key in ctrl:
                    c = ctrl[key]
                    all_path_rows.append({
                        "Method": label,
                        "all_path_MRR_ns": c["all_path"].get("MRR_ns"),
                        "all_path_MRR_strict": c["all_path"].get("MRR"),
                    })
            with open(out / f"controls_{protocol}.json", "w") as f:
                json.dump({k: ctrl[k] for k in
                           ("model", "backbone", "presence_control",
                            "length_only") if k in ctrl}, f, indent=2)
        pd.DataFrame(all_path_rows).to_csv(
            out / f"all_path_{protocol}.csv", index=False)

        # ablations
        abl_rows = []
        for name, cfg in ABLATIONS:
            vals = _rows_for(cfg, args.seeds, v2, protocol, _pamr_metrics)
            if vals:
                r = _row(name, vals, len(vals))
                r["Method"] = name
                abl_rows.append(r)
        pd.DataFrame(abl_rows).to_csv(out / f"ablations_{protocol}.csv",
                                      index=False)
        summary[protocol] = {"rows": len(rows), "ablations": len(abl_rows)}

    # single-seed diagnostics
    diag = []
    for name, cfg in (("v1-fixed", "v1fixed"),
                      ("loser-hard", "loser_hard")):
        for protocol in ("k10", "k100"):
            p = (v2 / "campaign" / cfg / "seed42" / "paths" /
                 "pamr_training_metrics.json")
            if p.exists():
                m = _pamr_metrics(str(p), protocol)
                if m:
                    diag.append({"run": name, "protocol": protocol,
                                 "MRR_ns": m["MRR_ns"],
                                 "MRR_strict": m["MRR"]})
    pd.DataFrame(diag).to_csv(out / "diagnostics.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("ACCEPTANCE aggregate-v2 " + json.dumps(summary))


if __name__ == "__main__":
    main()
