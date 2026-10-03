"""
PAMR Path-Length Baseline Scorer

Assigns scores based solely on path length:  score = -hops + offset
where offset is applied only to positive samples, per the tie-breaking mode.

The resulting score CSV (query_id, score, is_positive) is compatible with
evaluate.py.

Usage:
    python -m pamr.models.pr.score --config configs/pr/pamr_score.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

from pamr import utils

SCRIPT_DIR = Path(__file__).parent.resolve()
PKG_ROOT = SCRIPT_DIR.parent.parent
REPO_ROOT = PKG_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="PAMR Path-Length baseline scorer"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to pamr_score.yaml config file")
    return parser.parse_args()


def resolve_config(config_path_str):
    cfg_path = Path(config_path_str).expanduser()
    if not cfg_path.is_absolute():
        for candidate in [Path.cwd() / cfg_path, REPO_ROOT / cfg_path]:
            if candidate.exists():
                cfg_path = candidate.resolve()
                break
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path_str}")
    config = utils.load_yaml_config(cfg_path)
    config = utils.resolve_config_paths(
        config, base_dir=cfg_path.parent, repo_root=REPO_ROOT)
    return config, cfg_path.parent


def main():
    args = parse_args()
    cfg, _ = resolve_config(args.config)

    paths_dir = cfg["paths_dir"]
    if not Path(paths_dir).is_absolute():
        paths_dir = str((REPO_ROOT / paths_dir).resolve())
    if not os.path.isdir(paths_dir):
        raise FileNotFoundError(f"Paths directory not found: {paths_dir}")

    queries_dir = cfg.get("queries_dir")
    if queries_dir is not None:
        if not Path(queries_dir).is_absolute():
            queries_dir = str((REPO_ROOT / queries_dir).resolve())
    else:
        queries_dir = str(Path(paths_dir).parent)
    if not os.path.isdir(queries_dir):
        raise FileNotFoundError(f"Queries directory not found: {queries_dir}")

    output_dir = cfg["output_dir"]
    if not Path(output_dir).is_absolute():
        output_dir = str((REPO_ROOT / output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    tie_break = cfg.get("tie_break", "middle")
    eps = 0.5
    offset_map = {"first": +eps, "middle": 0.0, "last": -eps}
    if tie_break not in offset_map:
        print(f"ERROR: tie_break must be in {list(offset_map)}, "
              f"got '{tie_break}'")
        sys.exit(1)
    pos_offset = offset_map[tie_break]

    # ── Optional relation filter ─────────────────────────────────────────
    rel_filter = cfg.get("relation_filter") or {}
    rel_mode = rel_filter.get("mode", "none")
    rel_list = set(rel_filter.get("relations", [])) if rel_mode != "none" else None

    print("PAMR Path-Length Baseline Scorer")
    print(f"  paths_dir:    {paths_dir}")
    print(f"  queries_dir:  {queries_dir}")
    print(f"  output_dir:   {output_dir}")
    print(f"  tie_break:    {tie_break}  (pos_offset={pos_offset:+.1f})")
    if rel_list:
        print(f"  rel_filter:   {sorted(rel_list)}")
    print()

    for split in ["train", "val", "test"]:
        paths_csv = os.path.join(paths_dir, f"{split}_paths.csv")
        queries_csv = os.path.join(queries_dir, f"{split}_queries.csv")
        if not os.path.exists(paths_csv) or not os.path.exists(queries_csv):
            print(f"  SKIP {split}: missing CSV")
            continue

        t0 = time.perf_counter()
        print(f"  Scoring {split} ...", flush=True)

        queries_df = pd.read_csv(queries_csv)
        pos_df = queries_df[queries_df["is_positive"] == 1]
        if rel_list:
            pos_df = pos_df[pos_df["relation"].isin(rel_list)]
        qinfo = {}
        for row in pos_df.itertuples(index=False):
            qinfo[int(row.query_id)] = int(row.tail_id)
        print(f"    qinfo: {len(qinfo):,} queries with positive tail",
              flush=True)

        paths_df = pd.read_csv(paths_csv)
        records = []
        stats = {"rows": 0, "skip_hops0": 0, "skip_no_qinfo": 0,
                 "skip_parse": 0}

        for row in paths_df.itertuples(index=False):
            qid = int(row.query_id)
            true_tail = qinfo.get(qid)
            if true_tail is None:
                stats["skip_no_qinfo"] += 1
                continue

            hops = int(row.hops)
            if hops <= 0:
                stats["skip_hops0"] += 1
                continue

            nodes_str = str(row.nodes)
            try:
                tail_id = int(nodes_str.split(",")[-1])
            except (ValueError, IndexError):
                stats["skip_parse"] += 1
                continue

            is_positive = int(tail_id == true_tail)
            score = -hops + (pos_offset if is_positive else 0.0)
            records.append((qid, score, is_positive))
            stats["rows"] += 1

        records.sort(key=lambda x: x[0])

        scores_csv = os.path.join(output_dir, f"{split}_scores.csv")
        df = pd.DataFrame(records, columns=["query_id", "score", "is_positive"])
        df.to_csv(scores_csv, index=False)

        unique_qids = df["query_id"].nunique()
        print(f"    → {scores_csv}")
        print(f"      {stats['rows']:,} rows, {unique_qids:,} queries  "
              f"skip: hops≤0={stats['skip_hops0']}  "
              f"no_qinfo={stats['skip_no_qinfo']}  "
              f"parse={stats['skip_parse']}  "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
