"""Artifact-level verification of the path pipeline's leakage-safety contracts.

All three assertions are evaluated from committed artifacts only (no
training, no C++ invocation):

A. Scope hygiene — the split-scoped path graph of a split may not contain
   any fact belonging to a split outside its declared scope (its own facts
   in particular). Checked by streaming graph_{split}.csv and matching
   each row against the excluded facts of the raw triples CSV, in both
   orientations and for the query relation or its registry inverse.

B. Own-edge masking — no emitted path may use the probe's own edge (the
   query relation or its registry inverse between head and tail) in ANY
   split. A train positive whose own edge exists in the graph must
   therefore emit either no path or a path that avoids the own pair as a
   consecutive hop.

C. Candidate identity (optional) — when both score CSVs are supplied,
   PAMR and VL-KGE must have scored exactly the same rows: same query
   set, same per-query row counts, same per-query positive multiset, and
   PAMR's candidate tails must match the prepared probe rows.

Usage:
    python -m pamr.scripts.verify_paths \
        --run-dir checkpoints/preproc/wikiart_mkg_v2/full/<ts> \
        --triples data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
        [--splits train,val,test] \
        [--pamr-scores .../test_scores.csv --vlkge-scores .../test_scores.csv \
         --scores-split test]

Exit code 0 = every check passed; 1 = at least one violation.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd


def _iter_graph_rows(graph_csv, chunksize=1_000_000):
    for chunk in pd.read_csv(graph_csv, chunksize=chunksize):
        for u, v, et in zip(chunk["u"].values, chunk["v"].values,
                            chunk["edge_type"].values):
            yield int(u), int(v), int(et)


def check_meta(run_dir, split, failures):
    """Meta-manifest consistency + policy identity."""
    problems = []
    graph_meta_path = os.path.join(run_dir, f"graph_{split}_meta.json")
    paths_meta_path = os.path.join(run_dir, f"{split}_paths_meta.json")
    graph_meta = paths_meta = None
    if os.path.exists(graph_meta_path):
        with open(graph_meta_path) as f:
            graph_meta = json.load(f)
    else:
        problems.append(f"missing {graph_meta_path}")
    if os.path.exists(paths_meta_path):
        with open(paths_meta_path) as f:
            paths_meta = json.load(f)
    else:
        problems.append(f"missing {paths_meta_path}")

    # Real artifact shapes: graph meta carries the top-level
    # relation_registry_hash; paths meta nests the full registry manifest
    # (to_dict adds registry_hash INSIDE relation_registry).
    graph_hash = (graph_meta or {}).get("relation_registry_hash")
    paths_registry = (paths_meta or {}).get("relation_registry") or {}
    paths_hash = paths_registry.get("registry_hash")
    if graph_hash and paths_hash and graph_hash != paths_hash:
        problems.append(
            f"graph/paths registry hashes differ: {graph_hash!r} vs "
            f"{paths_hash!r}")

    if graph_meta is not None:
        scope = list(graph_meta.get("path_graph_scope", []))
        # Train normally depends on its own facts (per-probe own-edge
        # masking removes them at search time); val/test must never.
        if split in scope and split != "train":
            problems.append(
                f"graph meta scope {scope} includes the split's own facts")
        if not graph_meta.get("mask_query_edges", False):
            problems.append("graph meta records mask_query_edges=false")
        if not graph_meta.get("relation_registry_hash"):
            problems.append("graph meta lacks relation_registry_hash")
    if paths_meta is not None:
        scope = list(paths_meta.get("path_graph_scope", []))
        if split in scope and split != "train":
            problems.append(
                f"paths meta scope {scope} includes the split's own facts")
        if not paths_meta.get("mask_query_edges", False):
            problems.append("paths meta records mask_query_edges=false")
        if not paths_meta.get("relation_registry"):
            problems.append("paths meta lacks relation_registry manifest")
    if graph_meta is not None and paths_meta is not None:
        for key in ("path_graph_scope", "mask_query_edges",
                    "allow_target_relation_traversal"):
            if graph_meta.get(key) != paths_meta.get(key):
                problems.append(
                    f"graph/paths meta disagree on {key}: "
                    f"{graph_meta.get(key)!r} vs {paths_meta.get(key)!r}")

    if problems:
        failures.extend(f"meta[{split}]: {p}" for p in problems)
    return bool(problems)


def build_exclusion_index(run_dir, split, triples_csv, failures):
    """Map each excluded-fact pair to the set of forbidden relation IDs.

    Excluded facts = raw triples whose mode is outside the split's declared
    path-graph scope. Returns ({frozenset(pair): {r_id, inv_id, ...}},
    n_excluded) or (None, 0) on failure.
    """
    graph_meta_path = os.path.join(run_dir, f"graph_{split}_meta.json")
    if not os.path.exists(graph_meta_path):
        failures.append(f"scope[{split}]: cannot load scope (no graph meta)")
        return None, 0
    with open(graph_meta_path) as f:
        scope = set(json.load(f)["path_graph_scope"])

    mapping_csv = os.path.join(run_dir, "entity_mapping.csv")
    rel_mapping_csv = os.path.join(run_dir, "relation_mapping.csv")
    if not (os.path.exists(mapping_csv) and os.path.exists(rel_mapping_csv)):
        failures.append(
            f"scope[{split}]: entity/relation mapping CSVs missing in run dir")
        return None, 0
    ent_to_id = dict(zip(
        pd.read_csv(mapping_csv)["entity"].values,
        pd.read_csv(mapping_csv)["id"].values))
    rel_to_id = dict(zip(
        pd.read_csv(rel_mapping_csv)["relation"].values,
        pd.read_csv(rel_mapping_csv)["id"].values))

    # Registry-inverse map from the artifact's own manifest (to_dict nests
    # inverse_relation_id). Guard against a missing manifest by degrading
    # to raw-relation-only coverage.
    inverse_map = {}
    paths_meta_path = os.path.join(run_dir, f"{split}_paths_meta.json")
    if os.path.exists(paths_meta_path):
        with open(paths_meta_path) as f:
            registry = (json.load(f).get("relation_registry") or {})
        for key, value in (registry.get("inverse_relation_id") or {}).items():
            try:
                inverse_map[int(key)] = int(value)
            except (TypeError, ValueError):
                continue

    forbidden = {}
    n_excluded = 0
    try:
        reader = pd.read_csv(triples_csv, chunksize=2_000_000)
    except FileNotFoundError:
        failures.append(f"scope[{split}]: triples CSV not found: {triples_csv}")
        return None, 0
    for chunk in reader:
        for h, r, t, mode in zip(chunk["head"].values, chunk["relation"].values,
                                 chunk["tail"].values, chunk["mode"].values):
            mode = str(mode)
            if mode in scope:
                continue
            hid = ent_to_id.get(h)
            tid = ent_to_id.get(t)
            rid = rel_to_id.get(r)
            if hid is None or tid is None or rid is None:
                failures.append(
                    f"scope[{split}]: excluded fact not in mapping: "
                    f"{h!r},{r!r},{t!r}")
                continue
            pair = frozenset((int(hid), int(tid)))
            rels = forbidden.setdefault(pair, set())
            rels.add(int(rid))
            inv = inverse_map.get(int(rid))
            if inv is not None:
                rels.add(inv)
            n_excluded += 1
    return forbidden, n_excluded


def check_scope(run_dir, split, triples_csv, failures, violations_out):
    """A — no excluded fact's own edge inside the split-scoped graph."""
    forbidden, n_excluded = build_exclusion_index(
        run_dir, split, triples_csv, failures)
    if forbidden is None:
        return
    graph_csv = os.path.join(run_dir, f"graph_{split}.csv")
    if not os.path.exists(graph_csv):
        failures.append(f"scope[{split}]: graph CSV missing: {graph_csv}")
        return
    n_rows = 0
    for u, v, et in _iter_graph_rows(graph_csv):
        n_rows += 1
        rels = forbidden.get(frozenset((u, v)))
        if rels is not None and et in rels:
            key = (u, v, et)
            violations_out.append(key)
            if len(violations_out) <= 20:
                failures.append(
                    f"scope[{split}]: graph row ({u},{v},edge_type={et}) "
                    "is an excluded fact's own edge")
    print(f"  scope[{split}]: graph rows={n_rows:,} excluded facts="
          f"{n_excluded:,} violations={len(violations_out):,}")


def load_probe_index(run_dir, split, failures):
    """Probe queries CSV -> row_index -> (h, t, qr, iqr)."""
    path = os.path.join(run_dir, f"{split}_queries.csv")
    if not os.path.exists(path):
        failures.append(f"own_edge[{split}]: probe queries missing: {path}")
        return None, 0
    qdf = pd.read_csv(path)
    idx = {}
    for row in qdf.itertuples(index=False):
        idx[int(row.row_index)] = (
            int(row.head_id), int(row.tail_id),
            int(row.query_relation_id),
            int(getattr(row, "inverse_query_relation_id", -1)))
    return idx, len(qdf)


def check_own_edge_and_completeness(run_dir, split, failures,
                                    own_violations, counts_out):
    """B — emitted paths never use the probe's own edge; every probe row
    appears exactly once (completeness)."""
    probe_idx, n_probes = load_probe_index(run_dir, split, failures)
    if probe_idx is None:
        return
    paths_csv = os.path.join(run_dir, f"{split}_paths.csv")
    if not os.path.exists(paths_csv):
        failures.append(f"own_edge[{split}]: paths CSV missing: {paths_csv}")
        return
    seen = defaultdict(int)
    n_found = 0
    str_cols = {"nodes": str, "node_types": str, "edge_types": str,
                "edge_directions": str}
    for chunk in pd.read_csv(paths_csv, chunksize=200_000, dtype=str_cols):
        for row in chunk.itertuples(index=False):
            ri = int(row.row_index)
            seen[ri] += 1
            meta = probe_idx.get(ri)
            if meta is None:
                failures.append(
                    f"own_edge[{split}]: paths row_index {ri} unknown")
                continue
            h, t, qr, iqr = meta
            own_rels = {qr}
            if iqr is not None and iqr >= 0:
                own_rels.add(iqr)
            try:
                hops = int(float(row.hops))
            except (ValueError, TypeError):
                hops = -1
            if hops <= 0:
                continue
            n_found += 1
            nodes = [int(x) for x in str(row.nodes).split(",") if x]
            etypes = [int(x) for x in str(row.edge_types).split(",") if x]
            if len(nodes) != hops + 1 or len(etypes) != hops:
                failures.append(
                    f"own_edge[{split}]: probe ri={ri} path contract "
                    f"violated (hops={hops}, nodes={len(nodes)}, "
                    f"edge_types={len(etypes)})")
                continue
            for i in range(hops):
                pair = frozenset((nodes[i], nodes[i + 1]))
                if pair == frozenset((h, t)) and etypes[i] in own_rels:
                    own_violations.append((ri, i))
                    if len(own_violations) <= 20:
                        failures.append(
                            f"own_edge[{split}]: probe ri={ri} uses its own "
                            f"edge at hop {i} "
                            f"(edge_type={etypes[i]})")
    counts_out["probes"] = n_probes
    counts_out["found"] = n_found
    duplicate = {ri: c for ri, c in seen.items() if c != 1}
    missing = [ri for ri in probe_idx if seen[ri] == 0]
    if duplicate:
        failures.append(
            f"completeness[{split}]: {len(duplicate):,} probe row_indexes "
            "appear more than once "
            f"(sample {sorted(duplicate.items())[:5]})")
    if missing:
        failures.append(
            f"completeness[{split}]: {len(missing):,} probes missing from "
            f"paths CSV (sample {missing[:5]})")
    print(f"  own_edge[{split}]: probes={n_probes:,} found={n_found:,} "
          f"own-edge violations={len(own_violations):,} "
          f"duplicate/missing probes={len(duplicate) + len(missing):,}")


def check_candidate_identity(run_dir, scores_split, pamr_scores,
                             vlkge_scores, failures, prepared_queries=None):
    """C — PAMR and VL-KGE scored exactly the same rows over the same
    candidate universe."""
    if not pamr_scores or not vlkge_scores:
        print("  candidate_identity: SKIPPED (score CSVs not supplied)")
        return
    for p in (pamr_scores, vlkge_scores):
        if not os.path.exists(p):
            failures.append(
                f"candidate_identity[{scores_split}]: file missing: {p}")
            return

    # The probe queries CSV lacks query_id grouping and is_positive; group
    # prepared tails per query here. Gold counts cross-check against the
    # PREPARED queries CSV when supplied (is_positive lives there).
    qdf = pd.read_csv(os.path.join(run_dir, f"{scores_split}_queries.csv"))
    prepared_tails = defaultdict(list)
    for row in qdf.itertuples(index=False):
        prepared_tails[int(row.query_id)].append(int(row.tail_id))
    prepared_pos = {}
    if prepared_queries and os.path.exists(prepared_queries):
        pdf = pd.read_csv(prepared_queries)
        prepared_pos = defaultdict(int)
        for row in pdf.itertuples(index=False):
            prepared_pos[int(row.query_id)] += int(row.is_positive)

    pamr_df = pd.read_csv(pamr_scores)
    vlkge_df = pd.read_csv(vlkge_scores)
    pamr_qids = sorted(pamr_df["query_id"].unique().tolist())
    vlkge_qids = sorted(vlkge_df["query_id"].unique().tolist())
    if pamr_qids != vlkge_qids:
        failures.append(
            f"candidate_identity[{scores_split}]: query sets differ "
            f"(pamr={len(pamr_qids):,}, vlkge={len(vlkge_qids):,}, "
            f"symmetric difference="
            f"{len(set(pamr_qids) ^ set(vlkge_qids)):,})")

    problems = 0
    pamr_counts = pamr_df.groupby("query_id").size()
    vlkge_counts = vlkge_df.groupby("query_id").size()
    joined = pd.concat([pamr_counts.rename("pamr"),
                        vlkge_counts.rename("vlkge")], axis=1).fillna(0)
    bad_counts = joined[joined["pamr"] != joined["vlkge"]]
    if len(bad_counts):
        problems += len(bad_counts)
        failures.append(
            f"candidate_identity[{scores_split}]: {len(bad_counts):,} "
            "queries with different row counts "
            f"(sample {bad_counts.head(3).to_dict()})")

    for name, df in (("pamr", pamr_df), ("vlkge", vlkge_df)):
        pos = df.groupby("query_id")["is_positive"].sum()
        if not prepared_pos:
            continue
        for qid, n_pos in pos.items():
            if int(n_pos) != prepared_pos.get(int(qid), 0):
                problems += 1
                failures.append(
                    f"candidate_identity[{scores_split}]: {name} positive "
                    f"count for qid={qid}: {int(n_pos)} vs prepared "
                    f"{prepared_pos.get(int(qid), 0)}")

    pamr_tails = defaultdict(list)
    for row in pamr_df.itertuples(index=False):
        pamr_tails[int(row.query_id)].append(int(row.candidate_tail_id))
    tail_mismatches = 0
    for qid, tails in prepared_tails.items():
        if sorted(tails) != sorted(pamr_tails.get(qid, [])):
            tail_mismatches += 1
    if tail_mismatches:
        problems += tail_mismatches
        failures.append(
            f"candidate_identity[{scores_split}]: {tail_mismatches:,} "
            "queries whose PAMR candidate tails differ from prepared rows")
    print(f"  candidate_identity[{scores_split}]: queries="
          f"{len(pamr_qids):,} problems={problems:,}")


def main():
    ap = argparse.ArgumentParser(
        description="Verify path-pipeline leakage-safety contracts")
    ap.add_argument("--run-dir", required=True,
                    help="Preprocess run dir (graph_{s}.csv, {s}_queries.csv, "
                         "{s}_paths.csv, meta manifests, mappings)")
    ap.add_argument("--triples", required=True,
                    help="Raw triples CSV with the 'mode' column")
    ap.add_argument("--splits", default="train,val,test",
                    help="Comma-separated splits to verify")
    ap.add_argument("--pamr-scores", default=None,
                    help="Optional PAMR scores CSV for candidate identity")
    ap.add_argument("--vlkge-scores", default=None,
                    help="Optional VL-KGE scores CSV for candidate identity")
    ap.add_argument("--scores-split", default="test",
                    help="Split the score CSVs belong to")
    ap.add_argument("--prepared-queries", default=None,
                    help="Optional prepared queries CSV "
                         "(<split>_queries.csv from prepare.py) used to "
                         "cross-check gold counts per query")
    args = ap.parse_args()

    failures = []
    print("=" * 80)
    print("PATH PIPELINE VERIFICATION")
    print("=" * 80)
    print(f"  run dir: {args.run_dir}")
    print(f"  triples: {args.triples}")
    print(f"  splits:  {args.splits}")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        print(f"\n── {split} ──")
        failures_before = len(failures)
        check_meta(args.run_dir, split, failures)
        scope_violations = []
        check_scope(args.run_dir, split, args.triples, failures,
                    scope_violations)
        own_violations = []
        counts = {}
        check_own_edge_and_completeness(
            args.run_dir, split, failures, own_violations, counts)
        added = len(failures) - failures_before
        status = "PASS" if added == 0 else f"FAIL ({added})"
        print(f"  [{split}] {status}")

    # Candidate identity is per score-CSV pair, not per verified split —
    # run it once whenever both CSVs are supplied.
    if args.pamr_scores and args.vlkge_scores:
        print(f"\n── candidate identity ({args.scores_split}) ──")
        check_candidate_identity(
            args.run_dir, args.scores_split, args.pamr_scores,
            args.vlkge_scores, failures,
            prepared_queries=args.prepared_queries)

    print("\n" + "=" * 80)
    if failures:
        print(f"VERIFICATION FAILED — {len(failures)} problem(s):")
        for line in failures[:50]:
            print(f"  - {line}")
        sys.exit(1)
    print("VERIFICATION PASSED — all checks clean")
    sys.exit(0)


if __name__ == "__main__":
    main()
