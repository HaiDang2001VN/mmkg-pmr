"""Fresh-campaign aggregation — June-format faithful.

Builds (from the fresh Phase D/E artifacts):
  result.csv   grouped two-row header (relation x {MRR,H1,H3,H5} + MRR
               Overall), one row per method — same layout as
               assets/result.csv (June).
  count.csv    distinct test queries per target relation (fresh weights).
  average.csv  per-relation metrics + macro avg/std (ddof=1) +
               count-weighted avg/std — same math as assets/average_macro.py
               (W-rel weighting on the 4 artist relations).
  delta report vs the June (paperv7) numbers, labelled by row.

Also a --june-selftest mode: rebuild the June rows from the RESTORED
June artifacts and diff against assets/result.csv — validates the
extractor/schema before any fresh artifact exists.

Per-relation metric sources (one schema per family):
  evaluate.py output  full/vlkge/<Model>/metrics.json per_relation with
                      flat 'Hits@k' keys + top-level 'MRR'/'MRR_ns'
  train_pamr output   .../pamr_training_metrics.json test.per_relation
                      with nested .Hits[k] (strict) and model-level MRR

Method rows (fresh defaults):
  VLKGE-{TransE,RotatE,DistMult,ComplEx} | Path-Aware Reasoning (PAR)
  | Path-Aware Multi-modal Reasoning (PAMR seed42/seed1/seed-mean)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent

RELATIONS = ["isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf"]
METRICS = ["MRR", "H1", "H3", "H5"]
METHODS_JUNE = [
    "VLKGE-TransE", "VLKGE-RotatE", "VLKGE-DistMult", "VLKGE-ComplEx",
    "Path-Aware Reasoning", "Path-Aware Multi-modal Reasoning",
]


def per_relation_from_evaluate(metrics_path):
    m = json.load(open(metrics_path))
    test = m.get("test") if "test" in m else m
    per_rel = test.get("per_relation", {})
    out = {}
    for rel in RELATIONS:
        e = per_rel.get(rel)
        if e is None:
            continue
        out[rel] = {
            "MRR": e.get("MRR"),
            "H1": (e.get("Hits") or {}).get(1, e.get("Hits@1")),
            "H3": (e.get("Hits") or {}).get(3, e.get("Hits@3")),
            "H5": (e.get("Hits") or {}).get(5, e.get("Hits@5")),
        }
    return out, test.get("MRR")


def per_relation_from_pamr(metrics_path):
    m = json.load(open(metrics_path))
    test = m["test"]
    per_rel = test.get("per_relation", {})
    out = {}
    hits_map = {"H1": 1, "H3": 3, "H5": 5}
    for rel in RELATIONS:
        e = per_rel.get(rel)
        if e is None:
            continue
        row = {"MRR": e.get("MRR")}
        for k, kk in hits_map.items():
            row[k] = (e.get("Hits") or {}).get(kk)
        out[rel] = row
    return out, test.get("MRR")


def load_june_rows(june_pre):
    """Rows from the RESTORED June artifacts (selftest baseline)."""
    rows = {}
    for model in ("TransE", "RotatE", "DistMult", "ComplEx"):
        p = june_pre / "full" / "vlkge" / model / "full" / "metrics.json"
        if not p.exists():
            p = june_pre / "full" / "vlkge" / model / "metrics.json"
        if p.exists():
            out, overall = per_relation_from_evaluate(p)
            rows[f"VLKGE-{model}"] = (out, overall)
    pm = june_pre / "full" / "pamr" / "pamr_training_metrics.json"
    if pm.exists():
        out, overall = per_relation_from_pamr(pm)
        rows["Path-Aware Multi-modal Reasoning"] = (out, overall)
    return rows


def rows_to_runframe(rows):
    recs = []
    for method, (per_rel, overall) in rows.items():
        rec = {"Method": method}
        for rel in RELATIONS:
            for i, metric in enumerate(METRICS):
                rec[f"{rel}_{metric}"] = (per_rel.get(rel) or {}).get(metric)
        rec["MRR Overall"] = overall
        recs.append(rec)
    df = pd.DataFrame(recs)
    return df


def write_grouped_result(df, out_path):
    """Grouped two-row header, June layout:
    row1: '', rel1,,,,rel2,,,,... , 'MRR Overall'
    row2: Method, MRR,H1,H3,H5, ..."""
    cols = ["Method"]
    header1 = [""]
    for rel in RELATIONS:
        header1 += [rel, "", "", ""]
        cols += [f"{rel}_{m}" for m in METRICS]
    header1 += ["MRR Overall"]
    cols += ["MRR Overall"]
    df = df[cols]
    lines = [",".join(header1), ",".join(cols)]
    for rec in df.to_numpy().tolist():
        lines.append(",".join("" if v is None or (isinstance(v, float)
                                                 and np.isnan(v)) else v
                              for v in rec))
    out_path.write_text("\n".join(lines) + "\n")


def macro_average(df, counts):
    """assets/average_macro.py math, verbatim (ddof=1 + count-weighted)."""
    result = df.reset_index(drop=True)
    W = np.array([float(counts[rel]) for rel in RELATIONS])
    for metric in METRICS:
        mat = df[[f"{rel}_{metric}" for rel in RELATIONS]].astype(
            float).to_numpy()
        keep = ~np.isnan(mat).any(axis=1)
        result[f"{metric}_avg"] = np.round(
            np.nanmean(mat, axis=1), 4)
        result[f"{metric}_std"] = np.round(
            np.nanstd(mat, axis=1, ddof=1), 4)
        with np.errstate(divide="ignore", invalid="ignore"):
            wavg = np.where(keep[:, None], mat, 0).dot(W) / np.where(
                keep[:, None], 1.0, 0.0).dot(W)
            dev = np.where(keep[:, None], mat - wavg[:, None], 0.0)
            wstd = np.sqrt(
                (dev ** 2 * np.where(keep[:, None], W[None, :], 0.0)).sum(
                    axis=1)
                / np.max(np.where(keep[:, None], W[None, :], 0.0).sum(axis=1),
                         1e-9))
        result[f"{metric}_wavg"] = np.round(wavg, 4)
        result[f"{metric}_wstd"] = np.round(wstd, 4)
    if "MRR Overall" in df:
        result["MRR_Overall"] = df["MRR Overall"].values
    return result


def fresh_counts(prepared_dir):
    # fresh layout: queries live at the preproc root; June layout kept
    # them under full/ — accept either so the selftest can run on the
    # June mirror unchanged.
    qpath = prepared_dir / "test_queries.csv"
    if not qpath.exists():
        qpath = prepared_dir / "full" / "test_queries.csv"
    q = pd.read_csv(qpath)
    q = q[q["relation"].isin(RELATIONS)]
    counts = q.groupby("relation")["query_id"].nunique().to_dict()
    # internally-inverted target directions carry the public name in
    # `relation` (prepare writes the raw name) — verified by June counts.
    return counts


def delta_report(june_df, fresh_df):
    lines = ["# Fresh-vs-June (paperv7) per-relation delta", ""]
    merged = june_df.merge(fresh_df, on="Method", how="outer",
                           suffixes=("_june", "_fresh"))
    for metric in METRICS:
        for rel in RELATIONS:
            col_j = f"{rel}_{metric}_june"
            col_f = f"{rel}_{metric}_fresh"
            if col_j in merged and col_f in merged:
                d = (merged[col_f].astype(float)
                     - merged[col_j].astype(float))
                for meth, dv in zip(merged["Method"], d):
                    if not np.isnan(dv):
                        lines.append(
                            f"- {meth} | {rel} | {metric}: "
                            f"{dv:+.4f}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="checkpoints/preproc/wikiart_mkg_v2")
    ap.add_argument("--out-dir", default="assets/report")
    ap.add_argument("--mode", choices=("fresh", "june-selftest"),
                    default="june-selftest")
    args = ap.parse_args()

    prepared = (SRC / args.prepared) if not Path(args.prepared).is_absolute() \
        else Path(args.prepared)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "june-selftest":
        rows = load_june_rows(prepared)
        df = rows_to_runframe(rows)
        counts = fresh_counts(prepared)
        write_grouped_result(df, out_dir / "result_june_selftest.csv")
        val = pd.read_csv(out_dir / "result_june_selftest.csv",
                          header=[0, 1], index_col=0)
        # normalize and compare against June's own result.csv
        june = SRC / "assets" / "result.csv"
        flat_june = pd.read_csv(june, header=[0, 1], index_col=0)
        flat_cols = {}
        current = None
        for rel, metric in flat_june.columns:
            metric = str(metric).strip()
            if metric == "MRR Overall":
                flat_cols[("overall", "MRR")] = flat_june[(rel, metric)]
                continue
            rel = str(rel).strip()
            if rel and not rel.startswith("Unnamed"):
                current = rel
            flat_cols[(current, metric)] = flat_june[(rel, metric)]
        diffs = []
        for (rel, metric) in flat_cols:
            for meth in methods_in(flat_june):
                a = float(flat_cols[(rel, metric)][meth])
                b = float(per_rel_lookup(rows, meth, rel, metric))
                if not (np.isnan(b) or abs(a - b) < 2e-4):
                    diffs.append(f"{meth}|{rel}|{metric}: {a} vs {b}")
        print("ACCEPTANCE june-selftest rows=", len(rows),
              "counts=", counts, "metric_diffs=", len(diffs))
        for d in diffs[:10]:
            print("  DIFF", d)
        sys.exit(0 if not diffs else 1)

    # fresh path: fill when artifacts exist
    rows = {}
    for model in ("TransE", "RotatE", "DistMult", "ComplEx"):
        p = prepared / "full" / "vlkge" / model / "metrics.json"
        if p.exists():
            out, overall = per_relation_from_evaluate(p)
            rows[f"VLKGE-{model}"] = (out, overall)
    pm = prepared / "full" / "pamr" / "pamr_training_metrics.json"
    if pm.exists():
        out, overall = per_relation_from_pamr(pm)
        rows["Path-Aware Multi-modal Reasoning"] = (out, overall)
    par = prepared / "full" / "par" / "pamr_training_metrics.json"
    if par.exists():
        out, overall = per_relation_from_pamr(par)
        rows["Path-Aware Reasoning"] = (out, overall)
    seed1 = prepared / "full" / "ablations" / "seed1" / "pamr_training_metrics.json"
    if seed1.exists():
        out, overall = per_relation_from_pamr(seed1)
        rows["PAMR-seed1"] = (out, overall)
    df = rows_to_runframe(rows)
    counts = fresh_counts(prepared)
    write_grouped_result(df, out_dir / "result.csv")
    avg = macro_average(df, counts)
    avg.to_csv(out_dir / "average.csv", index=False, float_format="%.4f")
    # June-compatible: headerless (count, relation order preserved)
    with open(out_dir / "count.csv", "w") as f:
        for rel in RELATIONS:
            if rel in counts:
                f.write(f"{rel},{counts[rel]}\n")
    print("ACCEPTANCE aggregation rows=", len(rows), "counts=", counts)


def methods_in(flat_june):
    return list(flat_june.index)


def per_rel_lookup(rows, meth, rel, metric):
    per_rel, overall = rows.get(meth, ({}, None))
    if metric == "MRR" and rel in ("overall", None):
        return overall
    if rel == "overall":
        return overall
    return (per_rel.get(rel) or {}).get(metric, np.nan)


if __name__ == "__main__":
    main()
