"""Journal v2 runtime table from the campaign metrics (CPU)."""

import argparse
import json
from pathlib import Path

import pandas as pd

CONFIGS = ["campaign", "cls", "no_feats", "par", "lenonly",
           "finetune", "depth12", "lf_temp", "hop3", "no_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    camp = Path(args.v2_root) / "campaign"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in CONFIGS:
        for d in sorted(camp.glob(f"{cfg}/seed*/paths/pamr_training_metrics.json")):
            m = json.load(open(d))
            test = m.get("test") or {}
            k100 = (m.get("extra_tests") or {}).get("k100") or {}
            pe = m.get("per_epoch") or []
            rows.append({
                "config": cfg,
                "seed": int(d.parent.parent.name.replace("seed", "")),
                "train_total_s": m.get("wall_time_seconds"),
                "train_cpu_s": m.get("cpu_time_seconds"),
                "gpu_peak_alloc_mb": m.get("peak_gpu_allocated_mb"),
                "test_wall_s": test.get("wall_time_seconds"),
                "k100_wall_s": k100.get("wall_time_seconds"),
                "epoch_mean_s": round(
                    sum(e.get("wall_time_s", 0) for e in pe) / len(pe), 2)
                if pe else None,
                "build_train_s": (m.get("dataset_build_timings_s") or {}).get("train"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "timings_v2.csv", index=False)
    print(f"ACCEPTANCE runtime-v2 rows={len(df)}")
    if len(df):
        print(df.groupby("config")[["train_total_s", "test_wall_s",
                                    "k100_wall_s", "epoch_mean_s"]]
              .mean().round(2).to_string())


if __name__ == "__main__":
    main()
