"""Journal v2 LaTeX tables from the frozen campaign CSVs (CPU)."""

import argparse
from pathlib import Path

import pandas as pd


def _main_table(df):
    cols = ["Method", "MRR_ns", "MRR_ns_std", "H1_ns", "H3_ns", "H5_ns",
            "MRR_strict"]
    df = df[cols].copy()
    df["MRR_ns_std"] = df["MRR_ns_std"].fillna(0.0)
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Method & MRR & $\\pm$ & H@1 & H@3 & H@5 & MRR (strict) \\\\",
             "\\midrule"]
    for _, r in df.iterrows():
        lines.append(
            f"{r['Method']} & {r['MRR_ns']:.4f} & {r['MRR_ns_std']:.4f} & "
            f"{r['H1_ns']:.4f} & {r['H3_ns']:.4f} & {r['H5_ns']:.4f} & "
            f"{r['MRR_strict']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="assets/report_v2")
    args = ap.parse_args()
    rep = Path(args.report_dir)
    out = rep / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for proto in ("k10", "k100"):
        main = pd.read_csv(rep / f"main_{proto}.csv")
        (out / f"main_{proto}.tex").write_text(_main_table(main) + "\n")
        abl = pd.read_csv(rep / f"ablations_{proto}.csv")
        (out / f"ablations_{proto}.tex").write_text(_main_table(abl) + "\n")
    print("ACCEPTANCE tables done")


if __name__ == "__main__":
    main()
