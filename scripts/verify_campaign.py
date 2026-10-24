"""Campaign verification gate and checkpoint report (J1/J2).

Checks every campaign run's metrics, K=10/K<=100 presence, coverage, the
no-feats == backbone equivalence, and the main point-estimate claims, then
writes reports/checkpoint_<phase>.md.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

CONFIGS = ["campaign", "cls", "no_feats", "par", "lenonly",
           "finetune", "depth12", "lf_temp", "hop3", "no_mask"]
REQUIRED = ["campaign", "cls", "no_feats", "par", "lenonly"]
SEEDS = [42, 1, 2, 3, 4]


def _load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-root", required=True)
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--phase", default="J1")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    v2, v1 = Path(args.v2_root), Path(args.v1_root)
    camp = v2 / "campaign"
    failures, warnings, rows = [], [], []

    for cfg in CONFIGS:
        for seed in args.seeds:
            p = camp / cfg / f"seed{seed}" / "paths" / "pamr_training_metrics.json"
            if not p.exists():
                if cfg in REQUIRED:
                    failures.append(f"missing {cfg} seed{seed}")
                continue
            m = _load(p)
            test = m.get("test") or {}
            k100 = (m.get("extra_tests") or {}).get("k100") or {}
            cov = test.get("coverage")
            if int(test.get("num_queries", 0)) != 239:
                failures.append(f"{cfg} seed{seed} K10 queries != 239")
            if cov is not None and abs(float(cov) - 1.0) > 1e-9:
                failures.append(f"{cfg} seed{seed} K10 coverage {cov}")
            if int(k100.get("num_queries", 0)) != 239:
                failures.append(f"{cfg} seed{seed} K100 queries != 239")
            if "residual_stats" not in test:
                failures.append(f"{cfg} seed{seed} missing residual_stats")
            rows.append({"config": cfg, "seed": seed,
                         "k10_mrr_ns": (test.get("MRR_ns")),
                         "k100_mrr_ns": (k100.get("MRR_ns")),
                         "resid_std": (test.get("residual_stats") or {})
                         .get("path_std")})

    # no-feats == backbone on the all-path subset
    ctrl = camp / "no_feats" / "seed42" / "path_control_k10.json"
    nofeat_eq = None
    if ctrl.exists():
        c = _load(ctrl)
        a = c["model"]["all_path"].get("MRR_ns")
        b = c["backbone"]["all_path"].get("MRR_ns")
        nofeat_eq = (a is not None and b is not None
                     and abs(float(a) - float(b)) < 1e-9)
        if not nofeat_eq:
            warnings.append(
                f"no-feats all-path {a} != backbone {b} (expected equality)")
    else:
        warnings.append("missing no-feats seed42 path_control_k10.json")

    # main claims (point estimates)
    claims = {}
    for cfg in ("campaign", "cls"):
        c = camp / cfg / "seed42" / "path_control_k10.json"
        if not c.exists():
            failures.append(f"missing {cfg} path_control")
            continue
        cc = _load(c)
        m = cc["model"]["all_path"].get("MRR_ns")
        bb = cc["backbone"]["all_path"].get("MRR_ns")
        nf = cc.get("presence_control", {}).get("all_path", {}).get("MRR_ns")
        claims[cfg] = {"all_path_mrr_ns": m, "backbone": bb,
                       "presence_control": nf}
        if m is None or bb is None or m <= bb:
            warnings.append(f"{cfg} all-path not above backbone")

    stats_files = sorted((Path(__file__).resolve().parent.parent /
                          "assets" / "report_v2").glob("stats/stats_*.json"))
    if not stats_files:
        warnings.append("stats_v2 outputs missing")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    md = [f"# Campaign checkpoint {args.phase}", ""]
    md.append(f"- runs checked: {len(rows)}")
    md.append(f"- no-feats == backbone (all-path): {nofeat_eq}")
    md.append(f"- failures: {len(failures)}; warnings: {len(warnings)}")
    md += ["", "## Main all-path claims (seed 42)"] + [
        f"- {k}: {v}" for k, v in claims.items()]
    if failures:
        md += ["", "## FAILURES"] + [f"- {x}" for x in failures]
    if warnings:
        md += ["", "## WARNINGS"] + [f"- {x}" for x in warnings]
    md += ["", "## Run table", ""]
    if rows:
        md.append(pd.DataFrame(rows).to_markdown(index=False))
    (outdir / f"checkpoint_{args.phase}.md").write_text("\n".join(md) + "\n")
    report = {"phase": args.phase, "runs": len(rows),
              "failures": failures, "warnings": warnings,
              "no_feats_equals_backbone": nofeat_eq,
              "claims": claims}
    with open(outdir / f"checkpoint_{args.phase}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"ACCEPTANCE verify-{args.phase} runs={len(rows)} "
          f"failures={len(failures)} warnings={len(warnings)}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
