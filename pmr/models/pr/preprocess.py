"""
Path-Relation (PR) preprocessor — Python wrapper for BFS path extraction.

Reads a YAML config, converts prepare.py negative-sampled query CSVs into
graph and query input files for the compiled C++ worker, runs the worker
(one process with num_threads internal threads), and merges the per-thread
part files into per-split path CSVs.

Usage:
    python -m pmr.models.pr.preprocess --config configs/pr/wikiart_mkg_v2_paths.yaml
"""

import argparse
import csv as csv_module
import datetime
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pmr import utils
from pmr.dataloader import KnowledgeGraphDataLoader

SCRIPT_DIR = Path(__file__).parent.resolve()      # pmr/models/pr/
PKG_ROOT   = SCRIPT_DIR.parent.parent             # mmkg-pmr/pmr/
REPO_ROOT  = PKG_ROOT.parent                      # mmkg-pmr/
WS_ROOT    = REPO_ROOT.parent                     # pmr/ (workspace)

# ── Entity-type derivation: which relations mark artwork / artist roles ──

ARTWORK_HEAD_RELATIONS = {
    "isCreatedByArtist", "hasStyle", "belongsToGenre",
    "isAssociatedWithTag", "isCreatedWithMedium", "isLocatedIn",
    "isCreatedInYear", "isRelatedToArtwork",
}

ARTIST_HEAD_RELATIONS = {
    "hasNationality", "isAssociatedWithArtMovement", "isAssociatedWithField",
    "wasBornIn", "diedIn", "isAffiliatedWithArtInstitution",
    "isMemberOfPaintingSchool", "wasBornOnDate", "diedOnDate",
    "isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf",
    "hasCreatedArtwork",
}

ARTIST_TAIL_RELATIONS = {"isCreatedByArtist"}

# ── helpers ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Path-Relation preprocessor — BFS path extraction")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to PR YAML config file")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip C++ compilation (assume binary exists)")
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated subset of splits to process "
                             "(default: all of train,val,test)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep per-run query/thread output files "
                             "(overrides the keep_temp config key)")
    return parser.parse_args()


def _resolve_config(path_str, base_dir):
    """Resolve a YAML config to a dict with path resolution."""
    pp = Path(path_str).expanduser()
    if not pp.is_absolute():
        for candidate in [base_dir / pp, PKG_ROOT / pp, REPO_ROOT / pp, Path.cwd() / pp]:
            if candidate.exists():
                pp = candidate.resolve()
                break
    if not pp.exists():
        raise FileNotFoundError(f"Config not found: {path_str}")
    cfg = utils.load_yaml_config(pp)
    cfg = utils.resolve_config_paths(cfg, base_dir=pp.parent, repo_root=REPO_ROOT)
    return cfg


def compile_cpp(source, binary, flags):
    """Compile the C++ source if the binary is stale or missing."""
    src = Path(source).expanduser()
    if not src.is_absolute():
        src = (REPO_ROOT / src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"C++ source not found: {source}")

    binp = Path(binary).expanduser()
    if not binp.is_absolute():
        binp = (REPO_ROOT / binp).resolve()

    if binp.exists() and binp.stat().st_mtime >= src.stat().st_mtime:
        print(f"  Binary up-to-date: {binp}")
        return str(binp)

    print(f"  Compiling {src.name} ...")
    flags_list = flags.strip().split() if isinstance(flags, str) else flags
    cmd = ["g++"] + flags_list + ["-o", str(binp), str(src)]
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  COMPILE ERROR:\n{result.stderr}")
        raise RuntimeError(f"C++ compilation failed:\n{result.stderr}")
    print(f"  → {binp}")
    return str(binp)


def _run_one_thread(args_tuple):
    """Run the C++ worker once (worker v2 manages its threads internally).

    Returns (thread_id_marker, ok)."""
    (binary, graph_csv, queries_csv, output_prefix,
     max_hops, max_fanout, decay_factor,
     thread_id, num_threads, log_dir,
     mask_own_edges, allow_target_traversal, target_ids_csv) = args_tuple

    cmd = [binary, graph_csv, queries_csv, output_prefix,
           str(max_hops), str(max_fanout), str(decay_factor),
           str(thread_id), str(num_threads), log_dir,
           str(int(mask_own_edges)), str(int(allow_target_traversal)),
           target_ids_csv]
    result = subprocess.run(cmd, capture_output=True, text=True)
    ok = (result.returncode == 0)
    if not ok:
        print(f"  [worker] FAILED (rc={result.returncode})")
        if result.stderr.strip():
            print(f"    stderr: {result.stderr.strip()[:400]}")
    return thread_id, ok


def audit_probe_completeness(row_meta, merged_row_indexes, split_name):
    """Fail if any prepared probe is missing from, or duplicated in, the
    merged worker output.

    Args:
        row_meta: {row_index: probe record} built from the queries CSV
        merged_row_indexes: row_index values observed in worker outputs
        split_name: split label for error messages
    """
    merged_counts = {}
    for row_index in merged_row_indexes:
        merged_counts[row_index] = merged_counts.get(row_index, 0) + 1
    absent = sorted(set(row_meta) - set(merged_counts))
    duplicated = sorted(
        idx for idx, count in merged_counts.items() if count > 1)
    if absent or duplicated:
        raise RuntimeError(
            f"Path merge completeness violation for split {split_name}: "
            f"{len(absent):,} probes missing"
            + (f" (first row_indexes: {absent[:8]})" if absent else "")
            + f", {len(duplicated):,} probes duplicated"
            + (f" (first row_indexes: {duplicated[:8]})"
               if duplicated else "")
            + ". The worker outputs and the queries CSV disagree."
        )


def main():
    args = parse_args()

    # ── 1. Load configs ───────────────────────────────────────────────────
    pr_path = Path(args.config).expanduser()
    if not pr_path.is_absolute():
        for candidate in [Path.cwd() / pr_path, REPO_ROOT / pr_path]:
            if candidate.exists():
                pr_path = candidate.resolve()
                break
    if not pr_path.exists():
        raise FileNotFoundError(f"PR config not found: {args.config}")

    pr_cfg = utils.load_yaml_config(pr_path)
    pr_cfg = utils.resolve_config_paths(pr_cfg, base_dir=pr_path.parent, repo_root=REPO_ROOT)

    prepare_cfg = _resolve_config(pr_cfg["prepare_config"], pr_path.parent)
    prepare_output_dir = prepare_cfg["output_dir"]
    if not Path(prepare_output_dir).is_absolute():
        prepare_output_dir = str(WS_ROOT / prepare_output_dir)

    seed = pr_cfg.get("seed", 42)
    utils.set_seed(seed)

    max_hops     = pr_cfg["max_hops"]
    max_fanout   = pr_cfg["max_fanout"]
    decay_factor = pr_cfg["decay_factor"]
    num_threads  = pr_cfg["num_threads"]
    keep_temp    = pr_cfg.get("keep_temp", False) or args.keep_temp

    # Split-aware path graph scope. Main protocol: validation paths may only
    # use train facts, test paths may use train+val facts. The graph source
    # itself always carries a 'mode' column so forbidden edges can be cut.
    path_graph_cfg = pr_cfg.get("path_graph") or {}
    path_graph_scopes = {
        "train": tuple(path_graph_cfg.get("train", ("train",))),
        "val": tuple(path_graph_cfg.get("val", ("train",))),
        "test": tuple(path_graph_cfg.get("test", ("train", "val"))),
    }
    allowed_modes = {"train", "val", "test"}
    for split_name, scope in path_graph_scopes.items():
        unknown = set(scope) - allowed_modes
        if unknown:
            raise ValueError(
                f"path_graph.{split_name} contains invalid modes: "
                f"{sorted(unknown)}"
            )
        if split_name in scope and not path_graph_cfg:
            # Defensive default: a split may never depend on its own facts in
            # the default leakage-safe table.
            raise ValueError(
                f"path_graph.{split_name} must not include its own facts "
                "under the default leakage-safe policy; pass an explicit "
                "path_graph section to disclose transductive diagnostics"
            )

    # Per-probe query-edge masking. Own relations (the query relation and its
    # registry inverse, between the probe's head and tail) are masked in
    # every split, matching the scope behaviour val/test already get. The
    # target-traversal rule is one uniform flag for all splits: when false,
    # BFS may not traverse any target-relation edge anywhere.
    mask_query_edges = bool(path_graph_cfg.get("mask_query_edges", True))
    allow_target_traversal = bool(
        path_graph_cfg.get("allow_target_relation_traversal", True))

    # ── Output: datetime folder under workspace checkpoints ─────────────
    base_output = pr_cfg["output_dir"]
    if not Path(base_output).is_absolute():
        base_output = str(WS_ROOT / base_output)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output, ts)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print("PATH-RELATION PREPROCESSOR")
    print("=" * 80)
    print(f"Graph CSV:        {pr_cfg['graph_csv']}")
    print(f"Prepare config:   {pr_cfg['prepare_config']}")
    print(f"max_hops={max_hops}  max_fanout={max_fanout}  decay={decay_factor}")
    print(f"Threads:           {num_threads}")
    print(f"Run dir:           {run_dir}")
    print("=" * 80)
    print("=" * 80)
    sys.stdout.flush()

    # ── 2. Load triples CSV and share the prepare/DataLoader mappings ─────
    print("\nLoading knowledge graph ...", flush=True)
    graph_df = pd.read_csv(pr_cfg["graph_csv"])
    if "mode" not in graph_df.columns:
        raise ValueError(
            "Path graph source must contain the raw 'mode' column so a "
            "split-aware scope can be applied"
        )
    print(f"  {len(graph_df):,} triples", flush=True)

    # Query IDs were generated by prepare.py through
    # KnowledgeGraphDataLoader. Reusing that exact mapping is mandatory.
    # Building sorted mappings here silently permutes every interior path
    # node and edge relation.
    mapping_dl = KnowledgeGraphDataLoader(
        data_path=prepare_cfg["data_path"],
        dataset_name=prepare_cfg["dataset"],
        exclude_relations=prepare_cfg.get("exclude_relations"),
        exclude_relations_eval=prepare_cfg.get("exclude_relations_eval"),
        inverse_relation_pairs=prepare_cfg.get("inverse_relation_pairs"),
        symmetric_relations=prepare_cfg.get("symmetric_relations"),
        use_per_relation_candidates=prepare_cfg.get(
            "use_per_relation_candidates", True),
        artist2artist_relations=prepare_cfg.get("artist2artist_relations"),
        bidirectional_eval=False,
    )
    entity_to_id, relation_to_id = mapping_dl.get_entities_and_relations()
    relation_registry = mapping_dl.get_relation_registry()
    print(f"  {len(entity_to_id):,} entities,  {len(relation_to_id):,} relations"
          " (shared with prepare.py)", flush=True)

    # Target-relation IDs for the uniform traversal rule (from the prepare
    # config's relation filter, falling back to artist2artist relations).
    target_relation_names = (
        (prepare_cfg.get("relation_filter") or {}).get("relations")
        or prepare_cfg.get("artist2artist_relations")
        or [])
    unknown_targets = [name for name in target_relation_names
                       if name not in relation_to_id]
    if unknown_targets:
        raise ValueError(
            "path_graph target relations missing from the shared mapping: "
            f"{unknown_targets}"
        )
    target_relation_ids = sorted(
        relation_to_id[name] for name in target_relation_names)
    target_ids_csv = ",".join(str(rid) for rid in target_relation_ids)
    print(f"  Path policy: mask_query_edges={mask_query_edges}  "
          f"allow_target_traversal={allow_target_traversal}  "
          f"target_relations={len(target_relation_ids)}", flush=True)

    graph_relations = set(graph_df["relation"].unique())
    missing_relations = sorted(graph_relations - set(relation_to_id))
    if missing_relations:
        # Relations removed by the shared DataLoader configuration must not be
        # available to BFS, otherwise paths contain unknown relation IDs.
        print(f"  Filtering {len(missing_relations):,} graph relations not present "
              "in the shared mapping", flush=True)
        graph_df = graph_df[~graph_df["relation"].isin(missing_relations)].copy()
    if graph_df.empty:
        raise ValueError("No graph edges remain after applying the shared mapping")
    graph_entities = set(pd.unique(graph_df[["head", "tail"]].values.ravel()))
    missing_entities = sorted(graph_entities - set(entity_to_id))
    if missing_entities:
        raise ValueError(
            "Graph contains entities absent from the prepare/DataLoader "
            f"mapping, for example: {missing_entities[:3]}"
        )

    # ── 3. Derive entity types from the full graph (split-independent) ────
    print("Deriving entity types ...", flush=True)

    etype = {eid: 3 for eid in entity_to_id.values()}  # default: attribute

    # Derive all types before writing any row. The previous one-pass writer
    # could serialize a stale default type before a later edge identified the
    # entity as artwork or artist.
    for row in graph_df.itertuples(index=False):
        hid = entity_to_id[row.head]
        tid = entity_to_id[row.tail]
        rel = row.relation
        if rel in ARTWORK_HEAD_RELATIONS and etype[hid] == 3:
            etype[hid] = 1
        if rel in ARTIST_TAIL_RELATIONS and etype[tid] == 3:
            etype[tid] = 2
        if rel in ARTIST_HEAD_RELATIONS and etype[hid] == 3:
            etype[hid] = 2

    type_counts = defaultdict(int)
    for t in etype.values():
        type_counts[t] += 1
    print(f"  Artwork (1):  {type_counts[1]:,}", flush=True)
    print(f"  Artist  (2):  {type_counts[2]:,}", flush=True)
    print(f"  Attr    (3):  {type_counts[3]:,}", flush=True)

    # Keep an auditable copy of the numeric mappings beside the graph.
    entity_mapping_path = os.path.join(run_dir, "entity_mapping.csv")
    with open(entity_mapping_path, "w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["id", "entity"])
        for entity, entity_id in sorted(entity_to_id.items(), key=lambda item: item[1]):
            w.writerow([entity_id, entity])
    relation_mapping_path = os.path.join(run_dir, "relation_mapping.csv")
    with open(relation_mapping_path, "w", newline="") as f:
        w = csv_module.writer(f)
        w.writerow(["id", "relation"])
        for relation, relation_id in sorted(relation_to_id.items(), key=lambda item: item[1]):
            w.writerow([relation_id, relation])
    print(f"  Mapping files: {entity_mapping_path}, {relation_mapping_path}",
          flush=True)
    relation_manifest_path = os.path.join(run_dir, "relation_manifest.json")
    with open(relation_manifest_path, "w") as f:
        json.dump(relation_registry.to_dict(), f, indent=2)
    print(f"  Relation manifest: {relation_manifest_path}", flush=True)

    # ── 4. Compile C++ (once) ─────────────────────────────────────────────
    if not args.no_compile:
        print("\nCompilation ...")
        binary = compile_cpp(pr_cfg["cpp_source"], pr_cfg["binary_path"],
                             pr_cfg["compile_flags"])
    else:
        binary = str((REPO_ROOT / pr_cfg["binary_path"]).resolve())
        print(f"\nSkipping compilation (--no-compile), using: {binary}")
        if not os.path.exists(binary):
            raise FileNotFoundError(f"Binary not found: {binary}")

    # ── 5. Process each split ─────────────────────────────────────────────

    requested_splits = (
        [s.strip() for s in args.splits.split(",")] if args.splits
        else ["train", "val", "test"])
    invalid_splits = set(requested_splits) - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(
            f"Unknown splits requested via --splits: {sorted(invalid_splits)}")

    for split_name in requested_splits:
        queries_path = os.path.join(prepare_output_dir, f"{split_name}_queries.csv")
        if not os.path.exists(queries_path):
            print(f"\n  SKIP {split_name}: queries CSV not found")
            continue

        print(f"\n{'─' * 60}")
        print(f"  {split_name.upper()}")
        print(f"{'─' * 60}")

        qdf = pd.read_csv(queries_path)
        print(f"  Loaded {len(qdf):,} query rows")

        # Query IDs must also be in the shared mapping. Fail before spawning
        # C++ workers instead of producing numerically plausible wrong paths.
        for row in qdf.itertuples(index=False):
            if (entity_to_id.get(row.head) != int(row.head_id)
                    or entity_to_id.get(row.tail) != int(row.tail_id)
                    or relation_to_id.get(row.relation) != int(row.relation_id)):
                raise ValueError(
                    f"Query mapping mismatch at query_id={row.query_id}. "
                    "Regenerate prepared queries with the same DataLoader "
                    "configuration used for path preprocessing."
                )

        # ── 5a. Write the split-scoped graph ──────────────────────────────
        scope_df = graph_df[
            graph_df["mode"].isin(path_graph_scopes[split_name])
        ].copy()
        if scope_df.empty:
            raise ValueError(
                f"Path graph scope for {split_name} is empty "
                f"({path_graph_scopes[split_name]})"
            )

        graph_csv_path = os.path.join(run_dir, f"graph_{split_name}.csv")
        n_total = len(scope_df)
        with open(graph_csv_path, "w", newline="") as f:
            w = csv_module.writer(f)
            w.writerow([
                "u", "v", "u_type", "v_type", "edge_type",
                "reverse_edge_type",
            ])
            for row in tqdm(scope_df.itertuples(index=False), total=n_total,
                            desc=f"  Writing graph[{split_name}] + types",
                            unit="rows"):
                hid = entity_to_id[row.head]
                tid = entity_to_id[row.tail]
                rid = relation_to_id[row.relation]
                reverse_rid = relation_registry.inverse_id(rid)
                w.writerow([
                    hid, tid, etype[hid], etype[tid], rid, reverse_rid,
                ])

        print(f"    Graph scope {split_name} → {graph_csv_path} "
              f"({os.path.getsize(graph_csv_path):,} bytes)", flush=True)

        # Persist the graph policy with the artifact so downstream stages can
        # verify the scope of edges used for this split's paths.
        graph_meta = {
            "split": split_name,
            "path_graph_scope": list(path_graph_scopes[split_name]),
            "mask_query_edges": mask_query_edges,
            "allow_target_relation_traversal": allow_target_traversal,
            "target_relation_ids": target_relation_ids,
            "relation_registry_hash": relation_registry.registry_hash,
            "num_entities": len(entity_to_id),
            "num_relations": relation_registry.num_relations,
        }
        graph_meta_path = os.path.join(run_dir, f"graph_{split_name}_meta.json")
        with open(graph_meta_path, "w") as f:
            json.dump(graph_meta, f, indent=2)

        # ── 5b. Build queries CSV with explicit row identity ─────────────
        # Every query CSV row is one (head, candidate tail) probe. The BFS
        # returns paths per probe; carrying row_index back through the merge
        # makes candidate_tail_id and query_relation_id explicit instead of
        # re-inferring them from path endpoints. The per-probe query relation
        # and its registry inverse drive the own-edge mask in the worker.
        queries_csv_path = os.path.join(run_dir, f"{split_name}_queries.csv")
        print(f"  Writing queries CSV ...", flush=True)
        with open(queries_csv_path, "w", newline="") as f:
            w = csv_module.writer(f)
            w.writerow([
                "row_index", "query_id", "head_id", "tail_id",
                "query_relation_id", "inverse_query_relation_id",
            ])
            for row_position, row in enumerate(
                    tqdm(qdf.itertuples(index=False), total=len(qdf),
                         desc=f"  Writing queries [{split_name}]",
                         unit="row", leave=False)):
                qrel_id = int(row.relation_id)
                w.writerow([
                    row_position,
                    int(row.query_id),
                    int(row.head_id),
                    int(row.tail_id),
                    qrel_id,
                    relation_registry.inverse_id(qrel_id),
                ])
        n_unique = qdf["query_id"].nunique()
        print(f"    → {queries_csv_path}  ({len(qdf):,} rows, {n_unique:,} unique queries)", flush=True)

        # ── 5c. Run the C++ worker ────────────────────────────────────────
        # Worker v2: ONE process parses the graph once and spawns num_threads
        # internal threads over contiguous query-group ranges; each thread
        # still writes its own _paths_<tid>.txt part file for the merge.
        output_prefix = os.path.join(run_dir, f"{split_name}")
        print(f"  Running worker (1 process, {num_threads} internal threads) ...",
              flush=True)
        print(f"    C++ logs: {run_dir}/_*_preprocess.log", flush=True)

        rc = _run_one_thread((binary, graph_csv_path, queries_csv_path,
                              output_prefix,
                              max_hops, max_fanout, decay_factor,
                              0, num_threads, run_dir,
                              mask_query_edges, allow_target_traversal,
                              target_ids_csv))
        _, ok = rc
        if not ok:
            raise RuntimeError(
                f"The C++ worker failed for split {split_name}; its probes "
                "would silently vanish from the paths CSV. Inspect the logs "
                f"under {run_dir}/_*_preprocess.log and rerun the split."
            )

        # ── 5d. Merge outputs ─────────────────────────────────────────────
        merge_path = os.path.join(run_dir, f"{split_name}_paths.csv")
        print(f"  Merging → {merge_path}")

        found_total = 0
        missing_total = 0
        self_total = 0
        hop_sum = 0
        merged_row_indexes = []

        # qdf row identity: row_index -> explicit probe metadata, stored as a
        # keyed record so no positional unpack can silently swap columns.
        qdf = qdf.reset_index(drop=True)
        qdf["row_index"] = range(len(qdf))
        row_meta = {
            int(row.row_index): {
                "query_id": int(row.query_id),
                "query_head_id": int(row.head_id),
                "candidate_tail_id": int(row.tail_id),
                "query_relation_id": int(row.relation_id),
                "is_positive": int(row.is_positive),
            }
            for row in qdf.itertuples(index=False)
        }

        with open(merge_path, "w", newline="") as fout:
            w = csv_module.writer(fout)
            w.writerow([
                "query_id", "row_index", "query_head_id",
                "candidate_tail_id", "query_relation_id", "is_positive",
                "hops", "nodes", "node_types", "edge_types",
                "edge_directions",
            ])
            for tid in range(num_threads):
                part_path = f"{output_prefix}_paths_{tid}.txt"
                if not os.path.exists(part_path):
                    continue
                with open(part_path) as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(";")
                        if len(parts) < 2:
                            continue
                        if len(parts) != 7:
                            raise RuntimeError(
                                "Path worker output has "
                                f"{len(parts)} fields; expected 7 "
                                "(query_id;row_index;hops;nodes;node_types;"
                                "edge_types;edge_directions). Recompile "
                                "preprocess.cpp before merging."
                            )
                        qid = parts[0]
                        row_index_part = parts[1]
                        hops = parts[2]
                        nodes = parts[3] if parts[3] else ""
                        ntypes = parts[4] if parts[4] else ""
                        etypes = parts[5] if parts[5] else ""
                        directions = parts[6] if parts[6] else ""
                        try:
                            h = int(hops)
                        except ValueError:
                            h = -1
                        if h > 0 and not directions:
                            raise RuntimeError(
                                "Path worker output lacks edge_directions. "
                                "Recompile preprocess.cpp before merging."
                            )
                        try:
                            row_index = int(row_index_part)
                        except ValueError:
                            raise RuntimeError(
                                "Path worker output lacks row_index. "
                                "Recompile preprocess.cpp before merging."
                            )
                        if row_index not in row_meta:
                            raise RuntimeError(
                                f"Path row_index {row_index} is unknown for "
                                f"split {split_name}; the queries file and "
                                "worker outputs disagree"
                            )
                        meta = row_meta[row_index]
                        query_id = meta["query_id"]
                        if int(qid) != query_id:
                            raise RuntimeError(
                                f"Path qid {qid} does not match the queries "
                                f"row_index {row_index} (qid {query_id})"
                            )
                        w.writerow([
                            query_id, row_index, meta["query_head_id"],
                            meta["candidate_tail_id"],
                            meta["query_relation_id"], meta["is_positive"],
                            hops, nodes, ntypes, etypes, directions,
                        ])
                        merged_row_indexes.append(row_index)

                        if h == -1:
                            missing_total += 1
                        elif h == 0 and len(nodes.split(",")) <= 2:
                            # Worker v2 emits self-probes as hops=-1, so this
                            # branch stays at 0; retained for any legacy
                            # worker output that still encodes 0-hop rows.
                            self_total += 1
                        else:
                            found_total += 1
                            hop_sum += h

        # ── 5d'. Probe-completeness audit ────────────────────────────────
        # Every prepared probe must appear exactly once in the merged output.
        audit_probe_completeness(row_meta, merged_row_indexes, split_name)

        paths_meta = {
            "split": split_name,
            "path_graph_scope": list(path_graph_scopes[split_name]),
            "mask_query_edges": mask_query_edges,
            "allow_target_relation_traversal": allow_target_traversal,
            "target_relation_ids": target_relation_ids,
            "relation_registry": relation_registry.to_dict(),
            "max_hops": max_hops,
            "num_query_rows": len(qdf),
            "num_found": found_total,
            "num_missing": missing_total,
            "num_self": self_total,
        }
        paths_meta_path = os.path.join(run_dir, f"{split_name}_paths_meta.json")
        with open(paths_meta_path, "w") as f:
            json.dump(paths_meta, f, indent=2)
        print(f"    Paths meta → {paths_meta_path}")

        # ── 5e. Stats ─────────────────────────────────────────────────────
        total_eval = found_total + missing_total + self_total
        print(f"    Total:    {total_eval:,}")
        print(f"    Found:    {found_total:,}")
        if found_total > 0:
            print(f"    Avg hops: {hop_sum / found_total:.1f}")
        print(f"    Self:     {self_total:,}")
        print(f"    Missing:  {missing_total:,}")

        # ── 5f. Cleanup temp files ────────────────────────────────────────
        if not keep_temp:
            for tid in range(num_threads):
                part_path = f"{output_prefix}_paths_{tid}.txt"
                if os.path.exists(part_path):
                    os.remove(part_path)
            if os.path.exists(queries_csv_path):
                os.remove(queries_csv_path)
            print(f"    (temp files cleaned)")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
