"""
VLKGE + Path-Length Comparison Table Generator

Reads metrics.json files and generates LaTeX + CSV comparison tables.
Supports both batch mode (multiple sources) and single-file mode (--metrics).

Usage (batch):   python pmr/scripts/compare_vlkge.py
Usage (single):  python pmr/scripts/compare_vlkge.py --metrics path/to/metrics.json
"""

import argparse
import csv
import json
import os

METRICS = ["MRR", "H@1", "H@3", "H@5"]

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "../../assets")

SOURCES = {
    "TransE (full)":       "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/vlkge/TransE/full/metrics.json",
    "TransE (artist)":     "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/vlkge/TransE/artist/metrics.json",
    "RotatE (full)":       "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/vlkge/RotatE/full/metrics.json",
    "RotatE (artist)":     "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/vlkge/RotatE/artist/metrics.json",
    "DistMult (full)":     "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/vlkge/DistMult/full/metrics.json",
    "DistMult (artist)":   "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/vlkge/DistMult/artist/metrics.json",
    "ComplEx (artist)":    "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/vlkge/ComplEx/artist/metrics.json",
    "Path-Length (first)": "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/pamr/pathlen/artist/pathlen_first/metrics.json",
    "Path-Length (last)":  "/workspace/checkpoints/preproc/wikiart_mkg_v2/full/pamr/pathlen/all/pathlen_last/metrics.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="VLKGE Comparison Table Generator")
    parser.add_argument("--metrics", type=str, default=None,
                        help="Path to a single metrics.json file")
    parser.add_argument("--name", type=str, default=None,
                        help="Method name (only with --metrics)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for .tex / .csv files")
    parser.add_argument("--prefix", type=str, default="comparison",
                        help="Output filename prefix (default: comparison)")
    return parser.parse_args()


def load_method(name, path, relations=None):
    """Load one metrics.json, return {rel: {metric: (strict, nonstrict)}}"""
    if not os.path.exists(path):
        print(f"  SKIP {name}: file not found: {path}")
        return None

    d = json.load(open(path))
    t = d["test"]
    pr = t.get("per_relation", {})

    has_ns = "MRR_ns" in t
    method_data = {"__overall__": {}}

    # Overall
    for m in METRICS:
        if m == "MRR":
            s = t.get("MRR", 0.0)
            ns = t.get("MRR_ns", 0.0) if has_ns else 0.0
        else:
            k = m.split("@")[1]
            s = t.get("Hits", {}).get(k, 0.0)
            ns = t.get("Hits_ns", {}).get(k, 0.0) if has_ns else 0.0
        method_data["__overall__"][m] = (s, ns)

    # Per-relation
    rel_list = relations or sorted(pr.keys())
    for rel in rel_list:
        if rel not in pr:
            continue
        r = pr[rel]
        method_data[rel] = {}
        for m_orig in METRICS:
            if m_orig == "MRR":
                s = r.get("MRR", 0.0)
                ns = r.get("MRR_ns", 0.0) if has_ns else 0.0
            else:
                k = m_orig.split("@")[1]
                s = r.get(f"Hits@{k}", 0.0)
                ns = r.get(f"Hits_ns@{k}", 0.0) if has_ns else 0.0
            method_data[rel][m_orig] = (s, ns)

    return method_data


def find_best_and_second(values, key_idx=0):
    """Return (best_idx, second_idx) sorted by values[key_idx] descending.
    Exclude collapsed values (MRR=1.0, nonstrict < 0.5 for key_idx=0)."""
    indexed = []
    for i, v in enumerate(values):
        s, ns = v
        val = s if key_idx == 0 else ns
        if key_idx == 0 and s >= 0.999 and ns < 0.5:
            continue
        indexed.append((i, val))
    indexed.sort(key=lambda x: -x[1])
    best = indexed[0][0] if indexed else None
    second = indexed[1][0] if len(indexed) > 1 else None
    return best, second


def format_cell(s, ns, has_ns=True, bold_s=False, ul_s=False, bold_ns=False, ul_ns=False):
    """Format cell with independent bold/underline for strict and non-strict."""
    def wrap(val, b, u):
        if b: return f"\\textbf{{{val:.4f}}}"
        if u: return f"\\underline{{{val:.4f}}}"
        return f"{val:.4f}"

    strict = wrap(s, bold_s, ul_s)
    if has_ns:
        nonstrict = wrap(ns, bold_ns, ul_ns)
        return f"{strict} / {nonstrict}"
    return strict


def build_latex(data, relations, has_ns=True):
    """Build LaTeX table string."""
    methods = list(data.keys())
    n_cols = len(relations) * len(METRICS) + 1

    # Pre-compute rankings
    columns = []
    for rel in relations:
        for m in METRICS:
            columns.append((rel, m))
    columns.append(("__overall__", "MRR"))

    rankings_s = {}
    rankings_ns = {}
    for (rel, m) in columns:
        values = [data[method].get(rel, {}).get(m, (0.0, 0.0)) for method in methods]
        rankings_s[(rel, m)] = find_best_and_second(values, key_idx=0)
        rankings_ns[(rel, m)] = find_best_and_second(values, key_idx=1) if has_ns else (None, None)

    # Build LaTeX
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{VLKGE + Path-Length Comparison (strict" +
             (r" / non-strict)" if has_ns else r")"),
             r"\label{tab:vlkge_compare}"]
    col_spec = "l" + "c" * n_cols
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: relation groups
    header1 = [r"\textbf{Method}"]
    for rel in relations:
        header1.append(r"\multicolumn{" + str(len(METRICS)) + r"}{c}{\textbf{" + rel + r"}}")
    header1.append(r"\textbf{Overall}")
    lines.append(" & ".join(header1) + r" \\")
    # cmidrules
    cmids = []
    start = 2
    for _ in relations:
        end = start + len(METRICS) - 1
        cmids.append(r"\cmidrule(lr){" + str(start) + "-" + str(end) + "}")
        start = end + 1
    lines.append(" ".join(cmids))

    # Header row 2: metric names
    header2 = [""]
    for _ in relations:
        for m in METRICS:
            header2.append(m)
    header2.append("MRR")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for method_idx, method in enumerate(methods):
        row = [method]
        for (rel, m) in columns:
            s, ns = data[method].get(rel, {}).get(m, (0.0, 0.0))
            best_s, second_s = rankings_s[(rel, m)]
            best_ns, second_ns = rankings_ns[(rel, m)]
            cell = format_cell(
                s, ns, has_ns=has_ns,
                bold_s=(method_idx == best_s),
                ul_s=(method_idx == second_s),
                bold_ns=(method_idx == best_ns) if has_ns else False,
                ul_ns=(method_idx == second_ns) if has_ns else False,
            )
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def export_csv(data, relations, output_path):
    """Export CSV (no headers, strict/non-strict split into columns)."""
    columns = []
    for rel in relations:
        for m in METRICS:
            columns.append((rel, m))
    columns.append(("__overall__", "MRR"))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for method in data:
            row = [method]
            for (rel, m) in columns:
                s, ns = data[method].get(rel, {}).get(m, (0.0, 0.0))
                row.extend([f"{s:.4f}", f"{ns:.4f}"])
            writer.writerow(row)


def main():
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT
    prefix = args.prefix
    os.makedirs(output_dir, exist_ok=True)

    data = {}
    relations = None

    if args.metrics:
        # Single-file mode — same 4-artist-relation format as batch
        name = args.name or os.path.splitext(os.path.basename(args.metrics))[0]
        print(f"Loading single metrics.json: {args.metrics}")
        relations = ["isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf"]
        method_data = load_method(name, args.metrics, relations=relations)
        if method_data:
            data[name] = method_data
            print(f"  Loaded {name} ({len(relations)} artist relations)")
    else:
        # Batch mode: all pre-defined sources
        print("Loading metrics (batch mode)...")
        artist_rels = ["isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf"]
        for name, path in SOURCES.items():
            method_data = load_method(name, path, relations=artist_rels)
            if method_data:
                data[name] = method_data
        relations = artist_rels

    if not data:
        print("ERROR: No data loaded")
        return

    print(f"\nBuilding table ({len(data)} methods, {len(relations)} relations)...")

    has_ns = any("MRR_ns" in json.load(open(p))["test"] for p in 
                 ([args.metrics] if args.metrics else SOURCES.values()) if os.path.exists(p))
    latex = build_latex(data, relations, has_ns=has_ns)

    tex_path = os.path.join(output_dir, f"{prefix}_table.tex")
    csv_path = os.path.join(output_dir, f"{prefix}_table.csv")

    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  → {tex_path}")

    export_csv(data, relations, csv_path)
    print(f"  → {csv_path}")

    print(f"\n{latex}\nDone.")


if __name__ == "__main__":
    main()
