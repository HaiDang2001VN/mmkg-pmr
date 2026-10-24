"""Journal v2 figures from the frozen campaign tables (CPU)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bar(ax, df, title, err_col=None):
    names = df["Method"].tolist()
    vals = df["MRR_ns"].astype(float).tolist()
    yerr = (df[err_col].fillna(0).astype(float).tolist()
            if err_col and err_col in df else None)
    colors = ["#4c72b0" if n.startswith("VLKGE") else
              "#dd8452" if n.startswith("PAR") else "#55a868"
              for n in names]
    ax.bar(range(len(names)), vals, yerr=yerr, color=colors, capsize=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, min(1.0, max(vals) * 1.25 if vals else 1.0))
    ax.set_ylabel("MRR (non-strict)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="assets/report_v2")
    ap.add_argument("--v2-root", required=True)
    args = ap.parse_args()
    rep = Path(args.report_dir)
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    for proto in ("k10", "k100"):
        main = pd.read_csv(rep / f"main_{proto}.csv")
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        _bar(ax, main, f"Main results ({proto.upper()})", "MRR_ns_std")
        fig.tight_layout()
        fig.savefig(figs / f"main_{proto}.pdf")
        plt.close(fig)
        apth = pd.read_csv(rep / f"all_path_{proto}.csv")
        apth = apth.rename(columns={"all_path_MRR_ns": "MRR_ns"})
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        _bar(ax, apth, f"All-path subset ({proto.upper()})")
        fig.tight_layout()
        fig.savefig(figs / f"all_path_{proto}.pdf")
        plt.close(fig)

    abl = pd.read_csv(rep / "ablations_k10.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _bar(ax, abl, "Ablations (K=10)", "MRR_ns_std")
    fig.tight_layout()
    fig.savefig(figs / "ablations_k10.pdf")
    plt.close(fig)

    # strict vs non-strict tie artifact
    try:
        c = json.load(open(rep / "controls_k10.json"))
        labels = ["Length", "Presence\ncontrol", "PAMR"]
        strict = [c["length_only"]["all_path"]["MRR"],
                  c["presence_control"]["all_path"]["MRR"],
                  c["model"]["all_path"]["MRR"]]
        ns = [c["length_only"]["all_path"]["MRR_ns"],
              c["presence_control"]["all_path"]["MRR_ns"],
              c["model"]["all_path"]["MRR_ns"]]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.bar(x - 0.18, strict, 0.36, label="strict", color="#4c72b0")
        ax.bar(x + 0.18, ns, 0.36, label="non-strict", color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("all-path MRR")
        ax.set_title("Tie handling: strict vs non-strict", fontsize=10)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / "tie_strict_vs_ns.pdf")
        plt.close(fig)
    except FileNotFoundError:
        pass

    # path-length distributions before/after the all-pair mask
    bk = Path(args.v2_root) / "oldmask_backup"
    new = Path(args.v2_root).parent / "wikiart_mkg_v2" / "full" / "pamr"
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bins = np.arange(-1.5, 6.5, 1)
    for label, path in (("old train pos", bk / "v1" / "train_paths.csv"),
                        ("new train pos", new / "train_paths.csv"),
                        ("new test pos", new / "test_paths.csv")):
        if not path.exists():
            continue
        p = pd.read_csv(path)
        p = p[p["is_positive"] == 1]["hops"].replace(-1, 0)
        ax.hist(p, bins=bins, alpha=0.5, density=True, label=label)
    ax.set_xlabel("path hops")
    ax.set_ylabel("density")
    ax.set_title("Positive-path length before/after the direct-edge mask",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "path_length_shift.pdf")
    plt.close(fig)
    print("ACCEPTANCE figures done")


if __name__ == "__main__":
    main()
