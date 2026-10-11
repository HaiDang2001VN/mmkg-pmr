"""Phase H — timing extraction (piggyback; no dedicated reruns).

Sources:
  1. PAMR-family runs: <paths_dir>/pamr_training_metrics.json
     (log_metrics=true configs) — total + per-epoch walls, GPU
     peak/alloc, avg batch ms, dataset-build timings (when present,
     2026-10-11 instrumentation onward), test-eval wall.
  2. Baselines: full/vlkge/<Model>/score metrics JSONs do not carry
     wall — the score chains wall bank in the phase D job log instead
     (parsed from reports/phaseD_*.out ACCEPTANCE step lines).
  3. BFS worker walls: reports/regenB_*.out step lines (paths-prod /
     paths-h3 wall from ACCEPTANCE timestamps) + per-worker logs.

Output: assets/report/timings.csv (method, split, metric, seconds) +
stdout ACCEPTANCE lines.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent.parent


def pamr_timings(metrics_path, method):
    m = json.load(open(metrics_path))
    rows = [
        (method, "train_total", m.get("wall_time_seconds")),
        (method, "train_cpu", m.get("cpu_time_seconds")),
        (method, "gpu_peak_alloc_mb", m.get("peak_gpu_allocated_mb")),
        (method, "gpu_peak_reserved_mb", m.get("peak_gpu_reserved_mb")),
    ]
    test = m.get("test") or {}
    rows.append((method, "test_eval_wall", test.get("wall_time_seconds")))
    pe = m.get("per_epoch") or []
    if pe:
        rows.append((method, "epoch_wall_mean",
                     round(sum(e.get("wall_time_s", 0) for e in pe)
                           / len(pe), 3)))
        abt = [e.get("avg_batch_time_ms") for e in pe
               if e.get("avg_batch_time_ms")]
        if abt:
            rows.append((method, "batch_ms_mean",
                         round(sum(abt) / len(abt), 3)))
    dbt = m.get("dataset_build_timings_s") or {}
    for split, wall in dbt.items():
        rows.append((method, f"dataset_build_{split}", wall))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="/media02/hlhdang/projects/pamr/"
                                          "checkpoints/preproc/wikiart_mkg_v2")
    ap.add_argument("--out-dir", default="assets/report")
    args = ap.parse_args()
    prepared = Path(args.prepared) if Path(args.prepared).is_absolute() \
        else (SRC / args.prepared)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sources = {
        "PAMR": prepared / "full" / "pamr" / "pamr_training_metrics.json",
        "PAR": prepared / "full" / "par" / "pamr_training_metrics.json",
        "PAMR-seed1": prepared / "full" / "ablations" / "seed1"
                      / "pamr_training_metrics.json",
    }
    for name, sub in (("finetune", "finetune"), ("depth12", "depth12"),
                      ("cls", "cls"), ("lf_temp", "lf_temp")):
        sources[f"PAMR-{name}"] = (prepared / "full" / "ablations" / sub
                                   / "pamr_training_metrics.json")
    sources["PAMR-hop3"] = (prepared.parent / "wikiart_mkg_v2_h3" / "pamr"
                            / "pamr_training_metrics.json")
    found = 0
    for method, path in sources.items():
        if path.exists():
            rows += pamr_timings(path, method)
            found += 1
    df = pd.DataFrame(rows, columns=["method", "metric", "seconds"])
    df.to_csv(out_dir / "timings.csv", index=False)
    print(f"ACCEPTANCE timings sources_found={found} rows={len(df)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
