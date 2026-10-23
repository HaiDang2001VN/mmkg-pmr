"""Pilot gate: path-content gain over the backbone and the presence control.

Each run is described by (label, training-metrics JSON, path-control JSON).
A run passes when:
  - the trained model's path-residual std >= --min-std, and
  - the paired-bootstrap 95% CI lower bound of all-path MRR minus backbone
    is > 0, and
  - the same against the presence control is > 0, and
  - the same against the path-length heuristic (-hops) is > 0.
The best passing run (highest all-path MRR) is selected; exit 0 iff one
passes, 2 otherwise.
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=3, action="append",
                    metavar=("LABEL", "METRICS", "CONTROL"), required=True)
    ap.add_argument("--min-std", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = []
    best = None
    for label, metrics_path, control_path in args.run:
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(control_path) as f:
            control = json.load(f)
        std = float(metrics.get("test", {})
                    .get("residual_stats", {}).get("path_std", 0.0))
        ap_mrr = control.get("model", {}).get("all_path", {}).get("MRR")
        bb = control.get("bootstrap_all_path_vs_backbone") or {}
        ct = control.get("bootstrap_all_path_vs_control") or {}
        ln = control.get("bootstrap_all_path_vs_length") or {}
        bb_lo = (bb.get("ci95") or [None])[0]
        ct_lo = (ct.get("ci95") or [None])[0]
        ln_lo = (ln.get("ci95") or [None])[0]
        passed = (std >= args.min_std and bb_lo is not None and bb_lo > 0
                  and ct_lo is not None and ct_lo > 0
                  and ln_lo is not None and ln_lo > 0)
        entry = {
            "label": label,
            "path_residual_std": round(std, 8),
            "all_path_mrr": ap_mrr,
            "full_mrr": control.get("model", {}).get("full", {}).get("MRR"),
            "vs_backbone_ci_lo": bb_lo,
            "vs_control_ci_lo": ct_lo,
            "vs_length_ci_lo": ln_lo,
            "passed": bool(passed),
        }
        runs.append(entry)
        if passed and (best is None or (ap_mrr or 0) > best["all_path_mrr"]):
            best = entry

    report = {"min_std": args.min_std, "runs": runs, "selected": best}
    print("ACCEPTANCE pilot-gate " + json.dumps(report))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    sys.exit(0 if best is not None else 2)


if __name__ == "__main__":
    main()
