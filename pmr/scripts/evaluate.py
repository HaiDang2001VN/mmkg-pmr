"""
Evaluate negative-sampled link prediction from pre-computed scores.

Reads prepare-generated query CSVs and externally-computed score CSVs,
then computes local MRR, Hits@K, and binary accuracy within each sampled
candidate set (positive + K negatives).

Expects scores to be pre-computed by a model-specific implementation.
The score file format is:

    query_id,score,is_positive
    0,3.14,1
    0,-0.50,0
    ...

Usage:
    python -m vlkge.scripts.evaluate --config configs/eval/wikiart_mkg_v2_neg_eval.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import time

import numpy as np
import pandas as pd

from pmr import utils
from pmr.ranking import compute_rank_metrics

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate negative-sampled link prediction from pre-computed scores"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to eval YAML config file")
    return parser.parse_args()


def resolve_config(config_path_str):
    """Resolve and load a YAML config with path resolution."""
    cfg_path = Path(config_path_str).expanduser()
    if not cfg_path.is_absolute():
        for candidate in [
            Path.cwd() / cfg_path,
            REPO_ROOT / cfg_path,
        ]:
            if candidate.exists():
                cfg_path = candidate.resolve()
                break
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path_str}")

    config = utils.load_yaml_config(cfg_path)
    config = utils.resolve_config_paths(config, base_dir=cfg_path.parent, repo_root=REPO_ROOT)
    return config, cfg_path.parent


def load_prepare_config(prepare_config_path_str, eval_config_base_dir):
    """Resolve and load the referenced prepare config."""
    pp = Path(prepare_config_path_str).expanduser()
    if not pp.is_absolute():
        for candidate in [
            eval_config_base_dir / pp,
            REPO_ROOT / pp,
        ]:
            if candidate.exists():
                pp = candidate.resolve()
                break
    if not pp.exists():
        raise FileNotFoundError(
            f"Prepare config referenced in eval config not found: {prepare_config_path_str}\n"
            f"  (resolved from eval config dir: {eval_config_base_dir})"
        )
    prepare_cfg = utils.load_yaml_config(pp)
    prepare_cfg = utils.resolve_config_paths(prepare_cfg, base_dir=pp.parent, repo_root=REPO_ROOT)
    return prepare_cfg


def evaluate_split(queries_df, scores_df, top_k, split_name, exclude_relations=None):
    """
    Evaluate a single split through the shared ranker (pmr.ranking).

    Scores are grouped per query_id and ranked with the centralized
    strict/non-strict semantics (best-positive selection, corrected
    non-strict rank). No intermediate per-row Python bookkeeping.

    Args:
        queries_df: DataFrame from prepare.py CSV
        scores_df: DataFrame from score CSV
        top_k: list of k values for Hits@K
        split_name: "val" or "test"
        exclude_relations: set of relation names to skip in metric computation
    """
    t_start = time.perf_counter()
    exclude_set = set(exclude_relations or [])

    # ── 1. Fast relation lookup: one row per query_id from positives ────
    t1 = time.perf_counter()
    pos_rows = queries_df[queries_df["is_positive"] == 1]
    qrel = {}
    for row in pos_rows.itertuples(index=False):
        qrel[int(row.query_id)] = row.relation
    n_prepared = len(qrel)
    print(f"  [{split_name}] built qrel: {n_prepared:,} entries "
          f"in {time.perf_counter() - t1:.1f}s", flush=True)

    # ── 2. Alignment check ──────────────────────────────────────────────
    score_qids = set(int(x) for x in scores_df["query_id"].unique())
    query_qids = set(qrel.keys())
    common_qids = query_qids & score_qids
    missing = query_qids - score_qids
    extra = score_qids - query_qids

    if missing:
        print(f"  WARNING: {len(missing):,} query_ids in queries but not in scores")
    if extra:
        print(f"  WARNING: {len(extra):,} query_ids in scores but not in queries")
    if not common_qids:
        print(f"  ERROR: No common query_ids — cannot evaluate {split_name}")
        return None

    n_common = len(common_qids)
    print(f"  [{split_name}] aligned: {n_common:,} common queries "
          f"in {time.perf_counter() - t1:.1f}s", flush=True)

    # ── 3. Rank through the shared evaluator ────────────────────────────
    t2 = time.perf_counter()
    scores = scores_df["score"].to_numpy()
    labels = scores_df["is_positive"].to_numpy()
    qids = scores_df["query_id"].to_numpy()

    keep = np.isin(qids, np.array(sorted(common_qids)))
    scores = scores[keep]
    labels = labels[keep]
    qids = qids[keep]
    print(f"  [{split_name}] ranking {len(scores):,} score rows "
          f"({len(np.unique(qids)):,} queries) via shared ranker ...",
          flush=True)

    ranked = compute_rank_metrics(
        scores, labels, qids,
        top_k=top_k,
        exclude_relations=exclude_set,
        split=split_name,
        rel_for_qid={split_name: qrel},
    )
    t_eval = time.perf_counter() - t2
    print(f"  [{split_name}] ranked {ranked['num_queries']:,} queries "
          f"in {t_eval:.1f}s  "
          f"(total {time.perf_counter() - t_start:.1f}s)", flush=True)

    n_evaluated = ranked["num_queries"]
    if n_evaluated == 0:
        print(f"  ERROR: No evaluable queries in {split_name} "
              f"(skipped_no_positive={ranked.get('skipped_no_positive', 0):,} "
              f"skipped_excluded={ranked.get('skipped_excluded', 0):,})")
        return None

    print(f"  [{split_name}] skipped_excluded={ranked.get('skipped_excluded', 0):,}  "
          f"skipped_no_pos={ranked.get('skipped_no_positive', 0):,}", flush=True)

    # ── 4. Map the shared ranker output onto the legacy schema ──────────
    per_rel = {}
    for rel, m in ranked.get("per_relation", {}).items():
        entry = {
            "count": m["count"],
            "MRR": m["MRR"],
            "MRR_ns": m["MRR_ns"],
            "Accuracy": m["Accuracy"],
            "Accuracy_ns": m["Accuracy_ns"],
        }
        for k in top_k:
            entry[f"Hits@{k}"] = m["Hits"].get(k, 0.0)
            entry[f"Hits_ns@{k}"] = m["Hits_ns"].get(k, 0.0)
        per_rel[rel] = entry

    diagnostics = {
        "num_queries_in_prepared": n_prepared,
        "num_queries_in_scores": len(score_qids),
        "num_queries_evaluated": n_evaluated,
        "num_skipped_missing_in_scores": len(missing),
        "num_skipped_extra_in_scores": len(extra),
        "num_skipped_no_positive": ranked.get("skipped_no_positive", 0),
        "num_skipped_excluded_relation": ranked.get("skipped_excluded", 0),
    }

    return {
        "split": split_name,
        "num_queries": n_evaluated,
        "MRR": ranked["MRR"],
        "MRR_ns": ranked["MRR_ns"],
        "Hits": ranked["Hits"],
        "Hits_ns": ranked["Hits_ns"],
        "Accuracy": ranked["Accuracy"],
        "Accuracy_ns": ranked["Accuracy_ns"],
        "per_relation": per_rel,
        "diagnostics": diagnostics,
    }


def main():
    args = parse_args()
    eval_config, eval_base_dir = resolve_config(args.config)

    # ── Load prepare config (referenced — no path duplication) ────────────
    prepare_config_path = eval_config["prepare_config"]
    prepare_cfg = load_prepare_config(prepare_config_path, eval_base_dir)
    output_dir = prepare_cfg["output_dir"]

    # ── Optional separate scores directory (when scores aren't in output_dir) ─
    scores_dir = eval_config.get("scores_dir")
    if scores_dir:
        if not Path(scores_dir).is_absolute():
            scores_dir = str((REPO_ROOT / scores_dir).resolve())
    else:
        scores_dir = output_dir

    # Read relations to exclude from metric computation (honours the prepare
    # config's exclude_relations_eval, which may contain inverse relations).
    exclude_from_metrics = set(prepare_cfg.get("exclude_relations_eval", []))

    top_k = eval_config.get("top_k", [1, 3, 5, 10])

    print("=" * 80)
    print("NEGATIVE-SAMPLED EVALUATION — EVALUATE")
    print("=" * 80)
    print(f"Prepare config:  {prepare_config_path}")
    print(f"Data dir:        {output_dir}")
    if scores_dir != output_dir:
        print(f"Scores dir:      {scores_dir}")
    print(f"Top-K:           {top_k}")
    if exclude_from_metrics:
        print(f"Exclude rels:    {sorted(exclude_from_metrics)}")
    print("=" * 80)

    # ── Evaluate each split ───────────────────────────────────────────────
    all_results = {}
    splits_to_eval = []

    for split_name in ["train", "val", "test"]:
        queries_path = os.path.join(output_dir, f"{split_name}_queries.csv")
        scores_path = os.path.join(scores_dir, f"{split_name}_scores.csv")

        if not os.path.exists(queries_path):
            print(f"\n  SKIP {split_name}: queries CSV not found at {queries_path}")
            continue
        if not os.path.exists(scores_path):
            print(f"\n  SKIP {split_name}: scores CSV not found at {scores_path}")
            continue

        splits_to_eval.append((split_name, queries_path, scores_path))

    if not splits_to_eval:
        print("\nERROR: No split had both queries and scores. Nothing to evaluate.")
        sys.exit(1)

    for split_name, queries_path, scores_path in splits_to_eval:
        print(f"\n{'─' * 60}")
        print(f"  {split_name.upper()}")
        print(f"{'─' * 60}")
        print(f"  Queries: {queries_path}")
        print(f"  Scores:  {scores_path}")

        queries_df = pd.read_csv(queries_path)
        scores_df = pd.read_csv(scores_path)

        print(f"  Loaded {len(queries_df):,} query rows,  {scores_df['query_id'].nunique():,} unique query_ids in scores")

        result = evaluate_split(queries_df, scores_df, top_k, split_name,
                                exclude_relations=exclude_from_metrics)
        if result is None:
            continue

        all_results[split_name] = result

        # Print per split
        print(f"\n  Evaluated queries:  {result['num_queries']:,}")
        print(f"  MRR:                {result['MRR']:.4f}  (ns={result['MRR_ns']:.4f})")
        for k in sorted(result["Hits"]):
            ns_v = result["Hits_ns"].get(k, 0)
            print(f"  Hits@{k}:             {result['Hits'][k]:.4f}  (ns={ns_v:.4f})")
        print(f"  Accuracy:           {result['Accuracy']:.4f}")

        # Per-relation (summarized)
        per_rel = result.get("per_relation", {})
        if per_rel:
            print(f"\n  Per-Relation ({len(per_rel)} relations):")
            print(f"  {'Relation':<40s} {'Count':>8s} {'MRR':>8s} {'Acc':>8s} "
                  + " ".join(f"{'H@{k}':>8s}" for k in sorted(top_k)))
            print(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 8} "
                  + " ".join("─" * 8 for _ in top_k))
            for rel, m in sorted(per_rel.items()):
                parts = [f"  {rel:<40s} {m['count']:>8d} {m['MRR']:>8.4f} {m['Accuracy']:>8.4f}"]
                for k in sorted(top_k):
                    parts.append(f"{m[f'Hits@{k}']:>8.4f}")
                print("".join(parts))

        # Diagnostics
        diag = result.get("diagnostics", {})
        if any(v > 0 for v in [
            diag.get("num_skipped_missing_in_scores", 0),
            diag.get("num_skipped_no_positive", 0),
            diag.get("num_skipped_excluded_relation", 0),
        ]):
            print(f"\n  Diagnostics:")
            for k, v in diag.items():
                if v:
                    print(f"    {k}: {v:,}")

    # ── Save metrics.json ─────────────────────────────────────────────────
    metrics_out = {}
    for split_name, result in all_results.items():
        metrics_out[split_name] = {
            "num_queries": result["num_queries"],
            "MRR": result["MRR"],
            "MRR_ns": result["MRR_ns"],
            "Hits": {str(k): v for k, v in result["Hits"].items()},
            "Hits_ns": {str(k): v for k, v in result["Hits_ns"].items()},
            "Accuracy": result["Accuracy"],
            "Accuracy_ns": result["Accuracy_ns"],
            "per_relation": result["per_relation"],
            "diagnostics": result["diagnostics"],
        }

    # Write to scores_dir when provided, else output_dir
    metrics_out_dir = scores_dir if scores_dir and scores_dir != output_dir else output_dir
    os.makedirs(metrics_out_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics → {metrics_path}")

    # ── Final summary across splits ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    for split_name, result in all_results.items():
        print(f"\n  {split_name.upper()}  ({result['num_queries']:,} queries)")
        print(f"    MRR:      {result['MRR']:.4f}  (ns={result['MRR_ns']:.4f})")
        for k in sorted(result["Hits"]):
            ns_v = result["Hits_ns"].get(k, 0)
            print(f"    Hits@{k}:   {result['Hits'][k]:.4f}  (ns={ns_v:.4f})")
        print(f"    Accuracy: {result['Accuracy']:.4f}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
