"""
Generate positive-negative query CSVs for negative-sampled evaluation protocol.

For each split triple (h, r, t), samples K negative tails from a per-relation
candidate pool (with artist2artist policy, excluding known true tails), and writes
a query-grouped CSV with both string entity/relation names and integer IDs.

Parallelised via multiprocessing for large splits (train).
Usage:
    python -m pamr.scripts.prepare --config configs/eval/wikiart_mkg_v2_neg_sampling.yaml
"""

import argparse
import csv as csv_module
import json
import multiprocessing
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pamr.dataloader import KnowledgeGraphDataLoader
from pamr import utils

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

# ── module-level globals set by worker initializer ─────────────────────────

_shared_pool_data = None   # dict: {r_id: set} for set_subtraction, {r_id: np.array} for rejection
_shared_filter_map = None
_shared_id_to_entity = None
_shared_K = None
_shared_split_name = None
_shared_base_seed = None
_shared_field_names = None
_shared_sampling_method = None


def _init_worker(data_dir, K, split_name, base_seed, field_names, sampling_method):
    global _shared_pool_data, _shared_filter_map, _shared_id_to_entity
    global _shared_K, _shared_split_name, _shared_base_seed, _shared_field_names
    global _shared_sampling_method

    with open(os.path.join(data_dir, "pool_data.pkl"), "rb") as f:
        _shared_pool_data = pickle.load(f)
    with open(os.path.join(data_dir, "filter_map.pkl"), "rb") as f:
        _shared_filter_map = pickle.load(f)
    with open(os.path.join(data_dir, "id_to_entity.pkl"), "rb") as f:
        _shared_id_to_entity = pickle.load(f)

    _shared_K = K
    _shared_split_name = split_name
    _shared_base_seed = base_seed
    _shared_field_names = field_names
    _shared_sampling_method = sampling_method


def _process_chunk(args):
    chunk_pairs, chunk_triples = args

    pool_data   = _shared_pool_data
    filter_map  = _shared_filter_map
    id_to_entity = _shared_id_to_entity
    K           = _shared_K
    split_name  = _shared_split_name
    field_names = _shared_field_names
    method      = _shared_sampling_method
    rng = np.random.default_rng(_shared_base_seed + multiprocessing.current_process()._identity[0])

    rows = []
    skipped_empty = 0
    actual_neg_counts = []

    for (h_id, r_id) in chunk_pairs:
        if r_id not in pool_data:
            for _ in chunk_triples.get((h_id, r_id), []):
                skipped_empty += 1
            continue

        known_tails = filter_map.get((h_id, r_id), set())
        pool = pool_data[r_id]

        if method == "set_subtraction":
            # Pre-compute filtered candidates once per (h,r), then sample per triple
            candidates = sorted(set(pool) - known_tails)
            for (t_id, t_name, h_name, r_name) in chunk_triples.get((h_id, r_id), []):
                available = [c for c in candidates if c != t_id]
                if len(available) < K:
                    skipped_empty += 1; continue
                sampled = rng.choice(available, size=K, replace=False)
                actual_neg_counts.append(K)
                _emit_query(rows, field_names, h_name, r_name, t_name,
                            h_id, r_id, t_id, 1, split_name)
                for neg_t_id in sampled:
                    neg_t_name = id_to_entity.get(int(neg_t_id), f"entity_{neg_t_id}")
                    _emit_query(rows, field_names, h_name, r_name, neg_t_name,
                                h_id, r_id, int(neg_t_id), 0, split_name)

        elif method == "rejection":
            pool_arr = pool  # numpy array
            pool_size = len(pool_arr)
            pool_set = set(int(x) for x in pool_arr.tolist())
            for (t_id, t_name, h_name, r_name) in chunk_triples.get((h_id, r_id), []):
                excluded = pool_set.intersection(known_tails)
                available_count = pool_size - len(excluded)
                if t_id not in excluded:
                    available_count -= 1
                if available_count < K:
                    skipped_empty += 1; continue
                sampled = set()
                max_attempts = min(K * 100, pool_size * 2)
                attempts = 0
                while len(sampled) < K and attempts < max_attempts:
                    cand = int(rng.integers(0, pool_size))
                    tid_c = int(pool_arr[cand])
                    if tid_c != t_id and tid_c not in known_tails:
                        sampled.add(tid_c)
                    attempts += 1
                if len(sampled) < K:
                    # Deterministic completion prevents bounded rejection
                    # from silently changing the benchmark cardinality.
                    available = sorted(pool_set - known_tails - {t_id})
                    if len(available) < K:
                        skipped_empty += 1
                        continue
                    sampled = set(int(x) for x in rng.choice(
                        available, size=K, replace=False))
                actual_neg_counts.append(K)
                _emit_query(rows, field_names, h_name, r_name, t_name,
                            h_id, r_id, t_id, 1, split_name)
                for neg_t_id in sampled:
                    neg_t_name = id_to_entity.get(neg_t_id, f"entity_{neg_t_id}")
                    _emit_query(rows, field_names, h_name, r_name, neg_t_name,
                                h_id, r_id, neg_t_id, 0, split_name)

        else:
            raise ValueError(f"Unknown sampling method: {method}")

    return rows, skipped_empty, actual_neg_counts, len(chunk_pairs)


def _emit_query(rows, field_names, h_name, r_name, t_name,
                h_id, r_id, t_id, is_pos, split_name):
    """Append one row dict to rows list."""
    rows.append(dict(zip(field_names, [-1, h_name, r_name, t_name,
                                        h_id, r_id, t_id, is_pos, split_name])))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate positive-negative query CSVs for sampled evaluation"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count)")
    return parser.parse_args()


def resolve_config(config_path_str):
    """Resolve and load the config YAML with path resolution."""
    cfg_path = Path(config_path_str).expanduser()
    if not cfg_path.is_absolute():
        for candidate in [
            Path.cwd() / cfg_path,
            REPO_ROOT / cfg_path,
        ]:
            if candidate.exists():
                cfg_path = candidate.resolve()
                break
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path_str}")

    config = utils.load_yaml_config(cfg_path)
    config = utils.resolve_config_paths(config, base_dir=cfg_path.parent, repo_root=REPO_ROOT)
    return config


def _id_lookup(mapping, key, label):
    """Look up an ID with a friendly error message on miss."""
    val = mapping.get(key)
    if val is None:
        raise KeyError(f"{label} '{key}' not found in mapping")
    return val


def _chunk_list(lst, n):
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    args = parse_args()
    config = resolve_config(args.config)

    seed = config.get("seed", 42)
    utils.set_seed(seed)

    num_workers = args.workers or multiprocessing.cpu_count()

    print("=" * 80)
    print("NEGATIVE-SAMPLED EVALUATION — PREPARE")
    print("=" * 80)
    print(f"Dataset:  {config['dataset']}")
    print(f"Data:     {config['data_path']}")
    print(f"Output:   {config['output_dir']}")
    print(f"Workers:  {num_workers}")
    for split, n in config["negatives_per_query"].items():
        print(f"  {split}: {n} negatives per query")
    print("=" * 80)

    # ── 1. Load dataset via KnowledgeGraphDataLoader ──────────────────────
    print("\nLoading dataset...")
    dl = KnowledgeGraphDataLoader(
        data_path=config["data_path"],
        dataset_name=config["dataset"],
        exclude_relations=config.get("exclude_relations"),
        exclude_relations_eval=config.get("exclude_relations_eval"),
        inverse_relation_pairs=config.get("inverse_relation_pairs"),
        symmetric_relations=config.get("symmetric_relations"),
        use_per_relation_candidates=config.get("use_per_relation_candidates", True),
        artist2artist_relations=config.get("artist2artist_relations"),
        bidirectional_eval=False,
    )

    entity_to_id, relation_to_id = dl.get_entities_and_relations()
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    print(f"  Entities:  {len(entity_to_id):,}")
    print(f"  Relations: {dl.get_num_relations():,}")

    # ── 2. Split data and build filter_map ────────────────────────────────
    train_data, val_data, test_data = dl.split_data()
    split_sizes = dl.get_split_sizes()
    print(f"  Splits:  Train={split_sizes['train']:,}  "
          f"Val={split_sizes['val']:,}  Test={split_sizes['test']:,}")

    # ── 3. Evaluation candidate pools (artist2artist policy applied) ──────
    eval_pools = {
        "train": dl.relation_to_valid_tails_eval_train,
        "val":   dl.relation_to_valid_tails_eval_val,
        "test":  dl.relation_to_valid_tails_eval_test,
    }

    # ── 4. Generate queries per split ─────────────────────────────────────
    negatives_cfg = config["negatives_per_query"]
    sampling_method = config.get("sampling_method", "set_subtraction")
    sample_fraction = config.get("sample_fraction", 1.0)
    relation_filter = config.get("relation_filter") or {}
    relation_filter_mode = relation_filter.get("mode", "none")
    target_relations = set(relation_filter.get("relations", []))
    if relation_filter_mode not in ("none", "eval_only", "train_eval"):
        raise ValueError(
            "relation_filter.mode must be one of none, eval_only, train_eval"
        )
    if relation_filter_mode != "none" and not target_relations:
        raise ValueError(
            "relation_filter.relations is required when relation_filter.mode "
            "is not none"
        )

    # Split-aware truth filtering is built per split below; building an
    # all-splits filter map here would cost a multi-GB pass over the
    # materialized data and is never consumed.
    # Under train_eval every prepared query uses the target relations, so the
    # scope truth maps only need those relations — restricting the subset
    # shrinks them from gigabytes to kilobytes. eval_only keeps the full maps
    # because its train split prepares queries for every relation.
    if relation_filter_mode == "train_eval":
        filter_relation_ids = sorted(
            relation_to_id[name] for name in target_relations
            if name in relation_to_id)
        print(f"  Scope truth maps restricted to "
              f"{len(filter_relation_ids)} target relations")
    else:
        filter_relation_ids = None
    scope_filter_maps = dl.compute_filter_maps_by_scope(
        relation_ids=filter_relation_ids)
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if sampling_method not in ("set_subtraction", "rejection"):
        raise ValueError(f"Unknown sampling_method: {sampling_method}")

    if sample_fraction < 1.0:
        print(f"\n  Sampling {sample_fraction*100:.1f}% of train triples for benchmark")
        train_data = train_data.sample(frac=sample_fraction, random_state=seed)

    metadata = {
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "relation_registry": dl.get_relation_registry().to_dict(),
        "filter_policy": {
            "train": "train",
            "val": "train+val",
            "test": "train+val+test",
        },
        "candidate_policy": {
            "target_relations": (
                sorted(target_relations)
                if relation_filter_mode != "none" else None
            ),
            "artist2artist_split_local": True,
        },
        "relation_filter": {
            "mode": relation_filter_mode,
            "relations": sorted(target_relations),
        },
        "splits": {},
    }

    FIELD_NAMES = [
        "query_id", "head", "relation", "tail",
        "head_id", "relation_id", "tail_id",
        "is_positive", "split",
    ]

    for split_name, split_df in [("train", train_data), ("val", val_data), ("test", test_data)]:
        K = negatives_cfg.get(split_name)
        if K is None:
            print(f"\n  Skipping '{split_name}' — not in negatives_per_query")
            continue

        # Inverse facts populate training/truth/candidate structures, but they
        # are not independent public query rows.
        split_df = split_df[~split_df["is_inverse"]].copy()
        if relation_filter_mode == "train_eval":
            split_df = split_df[split_df["relation"].isin(target_relations)]
        elif relation_filter_mode == "eval_only" and split_name != "train":
            split_df = split_df[split_df["relation"].isin(target_relations)]

        n_triples = len(split_df)
        print(f"\n{'─' * 60}")
        print(f"  {split_name.upper()}  ({n_triples:,} triples,  K={K})")
        print(f"{'─' * 60}")

        pool = eval_pools[split_name]

        # Split-aware truth filtering: train filters train truths,
        # val filters train+val truths, test filters train+val+test.
        filter_map = scope_filter_maps[split_name]

        # ── 4a. Collect unique (h,r) pairs and their triples ──────────────
        print(f"  Collecting pairs ...")
        unique_pairs = set()
        pair_rows = defaultdict(list)
        skipped_mapping = 0

        for _, row in tqdm(split_df.iterrows(), total=n_triples,
                           desc=f"  Collecting [{split_name}]",
                           unit="triple", leave=False):
            h_name, r_name, t_name = row["head"], row["relation"], row["tail"]
            try:
                h_id = _id_lookup(entity_to_id, h_name, "head")
                r_id = _id_lookup(relation_to_id, r_name, "relation")
                t_id = _id_lookup(entity_to_id, t_name, "tail")
            except KeyError:
                skipped_mapping += 1
                continue
            unique_pairs.add((h_id, r_id))
            pair_rows[(h_id, r_id)].append((t_id, t_name, h_name, r_name))

        if skipped_mapping:
            print(f"  Skipped {skipped_mapping:,} rows (missing entity/relation ID)")

        n_unique = len(unique_pairs)

        # ── 4b. Parallelise: split pairs into chunks ──────────────────────
        pairs_list = list(unique_pairs)
        n_workers = min(num_workers, n_unique) if n_unique > 0 else 1
        chunks = _chunk_list(pairs_list, n_workers)

        # Build chunk_triples for each chunk (only the keys in that chunk)
        chunk_args = []
        for chunk in chunks:
            chunk_set = set(chunk)
            chunk_triples = {k: v for k, v in pair_rows.items() if k in chunk_set}
            chunk_args.append((chunk, chunk_triples))

        print(f"  Parallelising: {n_unique:,} pairs → {n_workers} workers")

        # ── 4c. Write shared data to disk for worker loading ──────────────
        if sampling_method == "set_subtraction":
            pool_data = {r_id: set(int(x) for x in tails) for r_id, tails in pool.items()}
        else:
            pool_data = {r_id: np.array(sorted(int(x) for x in tails), dtype=np.int64)
                         for r_id, tails in pool.items()}

        tmp_dir = tempfile.mkdtemp(prefix="prep_shared_")
        for fname, obj in [("pool_data.pkl", pool_data),
                           ("filter_map.pkl", filter_map),
                           ("id_to_entity.pkl", id_to_entity)]:
            with open(os.path.join(tmp_dir, fname), "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Shared data → {tmp_dir}")

        # ── 4d. Process chunks in parallel ─────────────────────────────────
        all_rows = []
        total_skipped = 0
        all_neg_counts = []

        try:
            with multiprocessing.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(tmp_dir, K, split_name, seed, FIELD_NAMES, sampling_method),
            ) as pool_mp:
                results = list(tqdm(
                    pool_mp.imap_unordered(_process_chunk, chunk_args),
                    total=len(chunk_args),
                    desc=f"  Processing [{split_name}]",
                    unit="chunk",
                ))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        for rows_chunk, skipped, neg_counts, _chunk_size in results:
            all_rows.extend(rows_chunk)
            total_skipped += skipped
            all_neg_counts.extend(neg_counts)

        # ── 4d. Assign sequential query_ids ────────────────────────────────
        query_counter = 0
        current_qid = -1
        for row_dict in all_rows:
            if row_dict["is_positive"] == 1:
                current_qid = query_counter
                query_counter += 1
            row_dict["query_id"] = current_qid

        n_with_neg = sum(1 for v in all_neg_counts if v > 0)
        n_full = sum(1 for v in all_neg_counts if v >= K)

        print(f"  Unique (h,r) pairs: {n_unique:,}")
        print(f"  Queries: {query_counter:,}  (skipped {total_skipped:,} with no negatives)")

        # ── 4e. Write CSV ──────────────────────────────────────────────────
        csv_path = os.path.join(output_dir, f"{split_name}_queries.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"  → {csv_path}")
        print(f"     {query_counter:,} queries,  {len(all_rows):,} rows")

        if all_neg_counts:
            arr = np.array(all_neg_counts)
            print(f"     Negatives/query:  avg={arr.mean():.1f}  "
                  f"min={arr.min()}  full={(arr == K).mean() * 100:.1f}%")

        metadata["splits"][split_name] = {
            "csv_path": csv_path,
            "num_queries": query_counter,
            "num_rows": len(all_rows),
            "negatives_requested": K,
            "negatives_avg": float(arr.mean()) if all_neg_counts else 0.0,
            "negatives_min": int(arr.min()) if all_neg_counts else 0,
            "negatives_full_pct": float((arr == K).mean() * 100) if all_neg_counts else 0.0,
            "skipped_no_negatives": total_skipped,
            "skipped_missing_id": skipped_mapping,
        }

    # ── 5. Write metadata ─────────────────────────────────────────────────
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nMetadata → {metadata_path}")

    print("\n" + "=" * 80)
    print("PREPARE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
