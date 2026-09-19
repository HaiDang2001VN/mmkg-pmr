"""
PAMR — Path-Aware Multimodal Reasoning

Transformer-based sequence model over interleaved entity / edge paths.
Entity embeddings come from a frozen VL-KGE model; relation embeddings
for edges and the query relation are interleaved as separate tokens.

Token sequence:  [CLS] [q_rel] [h] [e1] [n2] [e2] [n3] ... [t]

Supports four training objectives, selectable via YAML:
  - binary           BCE logit vs is_positive label
  - ranking          MarginRankingLoss (positive > negative per query)
  - link_prediction  Logistic softplus loss (±1 labels, same as KGE)
  - reconstruction   Causal next-token embedding L2 loss (hybrid mask:
                     bidirectional prefix [CLS..t1], causal AR from r2)
"""

import math
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from pmr.dataloader import KnowledgeGraphDataLoader
from pmr import helpers, utils

# ── VL-KGE model loader ────────────────────────────────────────────────────

def load_vlkge_model(model_config_path, checkpoint_path, device="cpu"):
    """
    Load a frozen VL-KGE model from config + checkpoint.

    Returns (vlkge_model, entity_to_id, relation_to_id, num_entities, num_relations)
    """
    # Resolve config
    cfg_path = Path(model_config_path)
    if not cfg_path.is_absolute():
        pkg_root = Path(__file__).parent.parent.parent  # pmr
        for cand in [cfg_path, pkg_root / cfg_path]:
            if cand.exists():
                cfg_path = cand.resolve()
                break
    if not cfg_path.exists():
        raise FileNotFoundError(f"VLKGE model config not found: {model_config_path}")
    cfg = utils.load_yaml_config(cfg_path)
    cfg = utils.resolve_config_paths(cfg, base_dir=cfg_path.parent,
                                     repo_root=Path(__file__).parent.parent.parent.parent)

    # Data loader for entity/relation ID maps
    dl = KnowledgeGraphDataLoader(
        data_path=cfg["data_path"],
        dataset_name=cfg["dataset"],
        exclude_relations=cfg.get("exclude_relations"),
        exclude_relations_eval=cfg.get("exclude_relations_eval"),
        inverse_relation_pairs=cfg.get("inverse_relation_pairs"),
        symmetric_relations=cfg.get("symmetric_relations"),
        use_per_relation_candidates=cfg.get("use_per_relation_candidates", True),
        artist2artist_relations=cfg.get("artist2artist_relations"),
        bidirectional_eval=cfg.get("bidirectional_eval", False),
    )
    entity_to_id, relation_to_id = dl.get_entities_and_relations()

    # Features
    from types import SimpleNamespace
    feat_args = SimpleNamespace(
        use_visual=cfg.get("use_visual", True),
        use_textual=cfg.get("use_textual", True),
        use_relation_features=cfg.get("use_relation_features", False),
        visual_features_path=cfg.get("visual_features_path"),
        textual_features_path=cfg.get("textual_features_path"),
        relation_features_path=cfg.get("relation_features_path"),
        normalize_visual=cfg.get("normalize_visual", False),
        normalize_textual=cfg.get("normalize_textual", False),
        normalize_relation=cfg.get("normalize_relation", False),
    )
    (visual_features, textual_features, relation_features,
     visual_entity_to_index, textual_entity_to_index) = utils.load_features(
         feat_args, entity_to_id, relation_to_id,
         num_relations=dl.get_num_relations())

    # Build model
    model, _, _ = helpers.get_model(
        model_name=cfg["model"],
        num_entities=len(entity_to_id),
        num_relations=dl.get_num_relations(),
        visual_features=visual_features,
        textual_features=textual_features,
        relation_features=relation_features,
        visual_entity_to_index=visual_entity_to_index,
        textual_entity_to_index=textual_entity_to_index,
        embedding_dim=cfg.get("embedding_dim", 768),
        fusion_mode=cfg.get("fusion_mode", "average"),
        use_structural=cfg.get("use_structural", True),
        use_visual=cfg.get("use_visual", True),
        use_textual=cfg.get("use_textual", True),
        freeze_visual=cfg.get("freeze_visual", True),
        freeze_textual=cfg.get("freeze_textual", True),
        visual_proj=cfg.get("visual_proj", False),
        textual_proj=cfg.get("textual_proj", False),
        shared_projection=cfg.get("shared_projection"),
        inductive=cfg.get("inductive", False),
        modality_asymmetry=cfg.get("modality_asymmetry", False),
        normalize_before_fusion=cfg.get("normalize_before_fusion", False),
        p_norm=cfg.get("p_norm", 1),
        normalize_relations=cfg.get("normalize_relations", True),
        margin=cfg.get("margin", 12.0),
        lr=cfg.get("lr", 0.1),
        use_scheduler=False,
        device=device,
    )
    model.to(device)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VLKGE checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # Mark seen entities for inductive
    if cfg.get("inductive", False) and hasattr(model, "mark_seen_entities"):
        train_data, _, _ = dl.split_data()
        seen = set(train_data["head"]).union(set(train_data["tail"]))
        train_ids = torch.tensor(
            [entity_to_id[e] for e in seen if e in entity_to_id], dtype=torch.long)
        model.mark_seen_entities(train_ids)

    # Freeze all VLKGE params
    for p in model.parameters():
        p.requires_grad = False

    num_entities = len(entity_to_id)
    num_relations = dl.get_num_relations()
    return model, entity_to_id, relation_to_id, num_entities, num_relations


# ── Multiprocessing sample-building worker ─────────────────────────────────

def _build_samples_worker(args):
    """Process a chunk of explicit-probe path rows in a worker process.

    Every merged paths-CSV row is one (query_id, row_index) probe carrying
    explicit candidate metadata, so positive labels never need to be re-inferred
    from path endpoints.
    """
    chunk_df, valid_entities, num_relations, max_hops = args
    results = []
    skipped = 0
    for row in chunk_df.itertuples(index=False):
        qid = int(row.query_id)
        row_index = int(row.row_index)

        hops = int(row.hops)
        has_path = hops >= 0
        probe_head_id = int(row.query_head_id)
        if hops == -1:
            # Explicit no-path probe: the candidate has no reachable path in
            # the scoped graph, so coverage keeps every prepared candidate.
            results.append({
                "row_id": (qid, row_index),
                "probe_head_id": probe_head_id,
                "candidate_tail_id": int(row.candidate_tail_id),
                "query_rel_id": int(row.query_relation_id),
                "is_positive": int(row.is_positive),
                "has_path": 0,
                "path_nodes": [],
                "path_edges": [],
                "path_directions": [],
                "node_types": [],
                "hops": 0,
            })
            continue

        if hops > max_hops:
            raise ValueError(
                f"Path for query_id={qid} row_index={row_index} has {hops} "
                f"hops, exceeding max_hops={max_hops}"
            )

        nodes_str = str(row.nodes)
        edges_str = str(row.edge_types)
        directions_str = str(row.edge_directions)
        types_str = str(row.node_types)
        if not nodes_str or not edges_str or not directions_str:
            raise ValueError(
                f"Malformed path fields for query_id={qid} "
                f"row_index={row_index}"
            )

        nodes = [int(x) for x in nodes_str.split(",")]
        edges = [int(x) for x in edges_str.split(",")]
        directions = []
        if directions_str:
            directions = [int(x) for x in directions_str.split(",")]
        ntypes = [int(x) for x in types_str.split(",")]

        if (len(nodes) != hops + 1 or len(edges) != hops
                or len(directions) != hops):
            raise ValueError(
                f"Malformed path for query_id={qid} row_index={row_index}: "
                "expected one node/edge/direction per hop"
            )
        if any(direction not in (0, 1) for direction in directions):
            raise ValueError(
                f"Invalid edge direction for query_id={qid} "
                f"row_index={row_index}; expected 0 or 1"
            )
        if len(ntypes) != len(nodes):
            raise ValueError(
                f"Malformed node types for query_id={qid} "
                f"row_index={row_index}: one type per node is required"
            )
        if any(node_type not in (1, 2, 3) for node_type in ntypes):
            raise ValueError(
                f"Invalid node type for query_id={qid} row_index={row_index}"
            )
        if has_path and int(row.candidate_tail_id) != nodes[-1]:
            raise ValueError(
                f"Path endpoint for query_id={qid} row_index={row_index} "
                f"({nodes[-1]}) disagrees with the explicit candidate tail "
                f"{row.candidate_tail_id}; ID mappings are inconsistent"
            )

        if any(nid not in valid_entities for nid in nodes):
            raise ValueError(
                f"Path for query_id={qid} row_index={row_index} contains an "
                "entity ID outside the canonical registry"
            )
        if any(eid not in range(num_relations) for eid in edges):
            raise ValueError(
                f"Path for query_id={qid} row_index={row_index} contains a "
                "relation ID outside the canonical registry"
            )

        results.append({
            "row_id": (qid, row_index),
            "probe_head_id": probe_head_id,
            "candidate_tail_id": int(row.candidate_tail_id),
            "query_rel_id": int(row.query_relation_id),
            "is_positive": int(row.is_positive),
            "has_path": 1,
            "path_nodes": nodes,
            "path_edges": edges,
            "path_directions": directions,
            "node_types": ntypes,
            "hops": hops,
        })

    return results, 0


# ── Dataset ─────────────────────────────────────────────────────────────────

class PAMRDataset(Dataset):
    """
    Explicit-probe path dataset.

    The path CSV contains one row per prepared probe (query_id, row_index),
    so positive labels, candidate tails, and query relations come straight
    from the CSV and never need re-inference. A probe either yields a path
    sample or an explicit no-path record so candidate coverage always equals
    the prepared protocol.

    Yields (path_nodes, path_edges, path_directions, node_types,
            query_rel_id, is_positive, candidate_tail_id, query_id,
            has_path, hops).
    """

    def __init__(self, paths_csv, queries_csv, entity_to_id, relation_to_id,
                 num_workers=None, max_hops=5, relation_whitelist=None,
                 num_relations=None):
        import time
        t0 = time.perf_counter()

        allowed_relation_ids = set()
        if relation_whitelist:
            unknown = [name for name in relation_whitelist
                       if name not in relation_to_id]
            if unknown:
                raise ValueError(
                    "relation_whitelist contains unknown relation names: "
                    f"{unknown}"
                )
            allowed_relation_ids = {
                relation_to_id[name] for name in relation_whitelist
            }

        # ── Pickle cache (format, hop, and whitelist-aware) ─────────────
        wl_tag = ""
        if relation_whitelist:
            wl_tag = "_wl_" + "_".join(sorted(relation_whitelist))[:120]
        cache_path = paths_csv + wl_tag + f"_h{max_hops}.v3.samples.pkl"
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            data_mtime = max(os.path.getmtime(paths_csv),
                             os.path.getmtime(queries_csv))
            if cache_mtime > data_mtime:
                print(f"  Loading cached samples from {cache_path}...")
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if not isinstance(cached, dict) or "samples" not in cached:
                    raise ValueError(
                        f"Invalid PAMR sample cache format: {cache_path}"
                    )
                self.samples = cached["samples"]
                if self.samples and "candidate_tail_id" not in self.samples[0]:
                    raise ValueError(
                        f"PAMR cache {cache_path} lacks explicit candidate "
                        "tails. Regenerate paths and samples."
                    )
                self.num_input_queries = cached["num_input_queries"]
                self.num_queries_with_positive = cached[
                    "num_queries_with_positive"]
                print(f"    {len(self):,} samples "
                      f"({time.perf_counter() - t0:.1f}s, cached)")
                return

        paths_df = pd.read_csv(paths_csv)
        queries_df = pd.read_csv(queries_csv)
        queries_df = queries_df.reset_index(drop=True)
        queries_df["row_index"] = range(len(queries_df))

        required_path_columns = {
            "query_id", "row_index", "query_head_id", "candidate_tail_id",
            "query_relation_id", "is_positive",
            "hops", "nodes", "node_types", "edge_types",
            "edge_directions",
        }
        missing_path_columns = required_path_columns - set(paths_df.columns)
        if missing_path_columns:
            raise ValueError(
                f"Path file {paths_csv} is missing required columns: "
                f"{sorted(missing_path_columns)}. Regenerate paths with the "
                "explicit-probe preprocessor."
            )
        paths_df = paths_df[sorted(required_path_columns)].copy()
        paths_df = paths_df.sort_values(
            ["query_id", "row_index"], kind="stable"
        ).reset_index(drop=True)

        valid_entities = set(entity_to_id.values())
        num_relations = (num_relations if num_relations is not None
                         else max(relation_to_id.values(), default=-1) + 1)

        # Cross-check every prepared probe: name-to-ID round trip and,
        # when a whitelist is active, drop path rows of probes whose query
        # relation is not in the whitelist.
        for row in queries_df.itertuples(index=False):
            if (entity_to_id.get(row.head) != int(row.head_id)
                    or entity_to_id.get(row.tail) != int(row.tail_id)
                    or relation_to_id.get(row.relation) != int(row.relation_id)):
                raise ValueError(
                    f"Query mapping mismatch at query_id={int(row.query_id)} "
                    f"row_index={int(row.row_index)}. Prepared queries and "
                    "PAMR must use the same canonical entity/relation "
                    "registry."
                )
        if allowed_relation_ids:
            allowed_row_indexes = set(
                queries_df.loc[
                    queries_df["relation_id"].isin(allowed_relation_ids),
                    "row_index"
                ].astype(int)
            )
            paths_df = paths_df[
                paths_df["row_index"].isin(allowed_row_indexes)
            ].reset_index(drop=True)

        self.num_input_queries = len({
            int(qid) for qid in paths_df["query_id"].unique()
        })

        # ── Multiprocessing build samples ──────────────────────────────
        if num_workers is None:
            num_workers = min(cpu_count(), 16)

        np_rows = len(paths_df)
        if num_workers > 1 and np_rows > 10000:
            # Split into chunks, process in parallel
            chunk_size = max(1, np_rows // num_workers)
            chunks = [paths_df.iloc[i:i + chunk_size]
                      for i in range(0, np_rows, chunk_size)]
            args_list = [(c, valid_entities, num_relations, max_hops)
                         for c in chunks]
            print(f"    Building samples with {num_workers} workers "
                  f"({len(chunks)} chunks)...", flush=True)
            with Pool(num_workers) as pool:
                chunk_results = pool.map(_build_samples_worker, args_list)
            self.samples = []
            for results, _skipped in chunk_results:
                self.samples.extend(results)
        else:
            # Single-process fallback (small datasets or num_workers=1)
            self.samples, _ = _build_samples_worker(
                (paths_df, valid_entities, num_relations, max_hops))

        print(f"    {len(self):,} samples "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        self.num_queries_with_positive = len({
            sample["row_id"][0] for sample in self.samples
            if sample["is_positive"]
        })

        # ── Save cache ─────────────────────────────────────────────────
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "samples": self.samples,
                    "num_input_queries": self.num_input_queries,
                    "num_queries_with_positive": self.num_queries_with_positive,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass  # disk full or permission — non-fatal

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["path_nodes"], dtype=torch.long),
            torch.tensor(s["path_edges"], dtype=torch.long),
            torch.tensor(s["path_directions"], dtype=torch.long),
            torch.tensor(s["node_types"], dtype=torch.long),
            s["query_rel_id"],
            s["is_positive"],
            s["row_id"][0],
            s["candidate_tail_id"],
            s["probe_head_id"],
            s["has_path"],
            s["hops"],
        )


def pamr_collate(batch):
    """Collate explicit probe samples with candidate metadata into tensors."""
    path_nodes_list, path_edges_list = [], []
    path_directions_list, node_types_list = [], []
    query_rel_ids, is_positives = [], []
    query_ids, candidate_tail_ids, probe_head_ids = [], [], []
    has_path_list, hops_list = [], []

    for (pn, pe, pd, nt, qr, ip, qi, ct, ph, hp, h) in batch:
        path_nodes_list.append(pn)
        path_edges_list.append(pe)
        path_directions_list.append(pd)
        node_types_list.append(nt)
        query_rel_ids.append(qr)
        is_positives.append(ip)
        query_ids.append(qi)
        candidate_tail_ids.append(ct)
        probe_head_ids.append(ph)
        has_path_list.append(hp)
        hops_list.append(h)

    if max(pn.size(0) for pn in path_nodes_list) == 0:
        # Every probe in the batch is an explicit no-path row: keep a minimal
        # empty-pad layout so the tensors stay shape-consistent.
        max_N = 1
    else:
        max_N = max(pn.size(0) for pn in path_nodes_list)

    # Pad to max_N
    pad_nodes = torch.zeros(len(batch), max_N, dtype=torch.long)
    pad_edges = torch.zeros(len(batch), max_N - 1, dtype=torch.long)
    pad_directions = torch.zeros(len(batch), max_N - 1, dtype=torch.long)
    pad_types = torch.zeros(len(batch), max_N, dtype=torch.long)

    for i, (pn, pe, pd, nt) in enumerate(zip(
            path_nodes_list, path_edges_list, path_directions_list,
            node_types_list)):
        N = pn.size(0)
        pad_nodes[i, :N] = pn
        pad_edges[i, :N - 1] = pe
        pad_directions[i, :N - 1] = pd
        pad_types[i, :N] = nt

    # Interleaved token mask (2*N+1 tokens; True = padding)
    max_len = 2 * max_N + 1
    token_mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    for i in range(len(batch)):
        N = path_nodes_list[i].size(0)
        token_mask[i, :2 * N + 1] = False

    return (
        pad_nodes,
        pad_edges,
        pad_directions,
        pad_types,
        torch.tensor(query_rel_ids, dtype=torch.long),
        torch.tensor(is_positives, dtype=torch.float32),
        torch.tensor(query_ids, dtype=torch.long),
        torch.tensor(candidate_tail_ids, dtype=torch.long),
        torch.tensor(probe_head_ids, dtype=torch.long),
        torch.tensor(has_path_list, dtype=torch.bool),
        torch.tensor(hops_list, dtype=torch.long),
        token_mask,
    )


# ── Paired Dataset ──────────────────────────────────────────────────────────

class PAMRPairedDataset(Dataset):
    """
    Candidate-probe paired sampling. Each (query_id, row_index) probe is an
    explicit positive or negative candidate; every positive probe is paired
    deterministically with one negative probe per epoch, and surplus
    positives stay as standalone samples instead of being dropped.
    """

    def __init__(self, paths_csv, queries_csv, entity_to_id, relation_to_id,
                 max_hops=5, relation_whitelist=None, num_relations=None):
        import time
        t0 = time.perf_counter()
        print("  Building paired-sampling dataset...")

        self.pamr_ds = PAMRDataset(paths_csv, queries_csv, entity_to_id,
                                    relation_to_id, max_hops=max_hops,
                                    relation_whitelist=relation_whitelist,
                                    num_relations=num_relations)
        self.num_input_queries = self.pamr_ds.num_input_queries
        self.num_queries_with_positive = self.pamr_ds.num_queries_with_positive

        # Group by query_id; probes are candidate rows, so grouping deserves
        # deterministic order to keep (query, row) pairings stable.
        self.pos_samples = defaultdict(list)
        neg_samples = defaultdict(list)
        for s in self.pamr_ds.samples:
            qid = s["row_id"][0]
            if s["is_positive"]:
                self.pos_samples[qid].append(s)
            else:
                neg_samples[qid].append(s)

        # Build flat entries over negatives. Cycling through positives makes
        # every positive candidate available when a query has several.
        self.pairs = []
        for qid in sorted(self.pos_samples):
            positives = self.pos_samples[qid]
            negatives = neg_samples.get(qid, [])
            if negatives:
                used_positive_indices = set()
                for neg_idx, neg_sample in enumerate(negatives):
                    positive_idx = neg_idx % len(positives)
                    used_positive_indices.add(positive_idx)
                    positive = positives[positive_idx]
                    self.pairs.append((positive, neg_sample))
                # If there are more positive probes than negatives, retain
                # the unpaired positives as standalone examples rather than
                # dropping them.
                self.pairs.extend(
                    (positive, None)
                    for positive_idx, positive in enumerate(positives)
                    if positive_idx not in used_positive_indices
                )
            else:
                # A positive-only example cannot contribute a pairwise
                # ranking loss, but remains valid for binary/link loss.
                self.pairs.extend((positive, None) for positive in positives)

        n_queries = len(self.pos_samples)
        n_negs = sum(len(samples) for samples in neg_samples.values())
        n_pairs = sum(neg is not None for _, neg in self.pairs)
        n_standalone = len(self.pairs) - n_pairs
        avg = n_negs / max(n_queries, 1)
        print(f"  PAMRPairedDataset: {n_queries:,} queries, "
              f"{n_negs:,} negatives → {n_pairs:,} pairs, "
              f"{n_standalone:,} standalone positives "
              f"(avg {avg:.1f} negs/query, "
              f"{time.perf_counter() - t0:.1f}s)")

        # Free internal PAMRDataset samples (they're now in pos_samples + pairs)
        del self.pamr_ds

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos, neg = self.pairs[idx]

        def as_tuple(sample):
            return (
                torch.tensor(sample["path_nodes"], dtype=torch.long),
                torch.tensor(sample["path_edges"], dtype=torch.long),
                torch.tensor(sample["path_directions"], dtype=torch.long),
                torch.tensor(sample["node_types"], dtype=torch.long),
                sample["query_rel_id"],
                sample["is_positive"],
                sample["row_id"][0],
                sample["candidate_tail_id"],
                sample["probe_head_id"],
                sample["has_path"],
                sample["hops"],
            )

        pos_tuple = as_tuple(pos)
        if neg is None:
            return pos_tuple
        return pos_tuple, as_tuple(neg)


def pamr_paired_collate(batch):
    """
    Interleave positive/negative pairs and retain standalone positives.
    """
    flat = []
    for item in batch:
        if (isinstance(item, tuple) and len(item) == 2
                and isinstance(item[0], tuple)):
            pos, neg = item
            flat.append(pos)
            flat.append(neg)
        else:
            flat.append(item)
    return pamr_collate(flat)


# ── PAMR Model ─────────────────────────────────────────────────────────────

class PAMR(nn.Module):
    """
    Path-Aware Multimodal Reasoning over interleaved paths on top of a frozen
    VL-KGE model.

    Score contract: for every prepared probe (head, query relation, candidate
    tail), PAMR outputs
        pamr_score = vlkge_score(head, qrel, candidate_tail) + path_residual,
    where path_residual comes from the transformer path for probes with an
    extracted path (readout MLP, zero-initialized so the model starts exactly
    at the VLKGE baseline) and from a learned scalar residual for no-path
    probes. The frozen VLKGE module never leaves evaluation mode.
    """

    def __init__(self, vlkge_model, num_node_types=4, max_tokens=13,
                 num_relation_types=None,
                 d_model=None, num_layers=4, num_heads=8,
                 dim_feedforward=None, dropout=0.1,
                 causal=False, readout="cls", use_node_types=True,
                 loss_mode="binary",
                 device=torch.device("cpu")):
        super().__init__()
        self.vlkge = vlkge_model
        self.max_tokens = max_tokens
        self.causal = causal
        if loss_mode == "reconstruction":
            self.causal = True
        self.readout = readout
        self.use_node_types = use_node_types
        self.loss_mode = loss_mode
        self.device = device

        self.vlkge.eval()
        for p in self.vlkge.parameters():
            p.requires_grad = False

        # ── Auto-detect effective dimensions from VLKGE model ────────────
        D = vlkge_model.embedding_dim

        is_rotate = type(vlkge_model).__name__ == 'RotatE'
        has_rel_imag = hasattr(vlkge_model, 'relation_embeddings_imag')
        has_entity_imag = (
            hasattr(vlkge_model, 'get_imaginary_embedding_wikiart')
            or hasattr(vlkge_model, 'get_imaginary_embedding_wn9img')
        )

        # Full entity representation preserves the components the frozen
        # scorer depends on: real (fused modalities) plus the imaginary part
        # (textual-driven for WikiArt ComplEx, rotations for WN9-IMG).
        entity_dim = 2 * D if has_entity_imag else D
        rel_dim = 2 * D if (has_rel_imag or is_rotate) else D

        if d_model is None:
            d_model = max(entity_dim, rel_dim)
        self.d_model = d_model

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        # Validation
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}. "
                f"Pick num_heads ∈ {[h for h in [1,2,4,8,12,16] if d_model % h == 0]}"
            )

        self._entity_dim = entity_dim
        self._rel_dim = rel_dim
        self._has_entity_imag = has_entity_imag
        self._has_rel_imag = has_rel_imag
        self._is_rotate = is_rotate

        # ── Projections (align entity/rel dims to d_model) ─────────────
        self.entity_proj = (
            nn.Linear(entity_dim, d_model) if entity_dim != d_model else nn.Identity()
        )
        self.rel_proj = (
            nn.Linear(rel_dim, d_model) if rel_dim != d_model else nn.Identity()
        )
        # The relation ID identifies the semantic inverse when one exists.
        # This additional bit preserves traversal orientation for relations
        # without an explicit inverse embedding as well.
        self.edge_direction_emb = nn.Embedding(2, d_model)

        # ── Transformer components ──────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_embedding = nn.Embedding(max_tokens, d_model)

        if use_node_types:
            self.node_type_emb = nn.Embedding(
                num_node_types + 1, d_model, padding_idx=0)
        else:
            self.node_type_emb = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Path residual head over pooled transformer output. Its final linear
        # layer is zero-initialized so path probes start exactly at the
        # frozen VLKGE baseline score.
        self.readout_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Learnable scalar residual assigned to probes whose BFS found no
        # path; zero-initialized so no-path probes also start at the VLKGE
        # baseline.
        self.no_path_residual = nn.Parameter(torch.zeros(1))
        # Gate scaling the reconstruction residual in reconstruction mode.
        self.recon_gate = nn.Parameter(torch.zeros(1))

        # ── Reconstruction head ─────────────────────────────────────────
        self.recon_head = nn.Linear(d_model, d_model)
        self._recon_avg_l2 = None
        self._recon_counts = None

        # ── Loss function ───────────────────────────────────────────────
        if loss_mode in ("binary", "binary_candidate"):
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_mode in ("ranking", "grouped_pairwise"):
            self.criterion = nn.MarginRankingLoss(margin=1.0)
        elif loss_mode in ("link_prediction", "reconstruction",
                            "listwise_softmax"):
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        self._reset_parameters()

    def train(self, mode=True):
        """Keep the frozen VLKGE child permanently in evaluation mode."""
        super().train(mode)
        self.vlkge.eval()
        return self

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        if self.node_type_emb is not None:
            nn.init.normal_(self.node_type_emb.weight, std=0.02)
            with torch.no_grad():
                self.node_type_emb.weight[0].zero_()
        nn.init.zeros_(self.edge_direction_emb.weight)
        if self.edge_direction_emb.num_embeddings > 1:
            nn.init.normal_(self.edge_direction_emb.weight[1:], std=0.02)
        for module in self.readout_mlp[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Final residual layer: zero weight and zero bias → exactly the
        # frozen VLKGE score before any training step.
        nn.init.zeros_(self.readout_mlp[-1].weight)
        nn.init.zeros_(self.readout_mlp[-1].bias)
        for proj in [self.entity_proj, self.rel_proj]:
            if isinstance(proj, nn.Linear):
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.recon_head.weight)
        nn.init.zeros_(self.recon_head.bias)
        # Residual parameters start at exactly zero so both path and no-path
        # probes reproduce the frozen VLKGE score before any training step.
        nn.init.zeros_(self.no_path_residual)
        nn.init.zeros_(self.recon_gate)

    def _entity_imag_full(self, entity_ids, real_embs):
        if hasattr(self.vlkge, 'get_imaginary_embedding_wikiart'):
            return self.vlkge.get_imaginary_embedding_wikiart(
                entity_ids, real_embs)
        return self.vlkge.get_imaginary_embedding_wn9img(entity_ids)

    def _build_tokens(self, path_nodes, path_edges, path_directions,
                      node_types, query_rel_id):
        """
        Build interleaved token sequence:
          [CLS] [q_rel] [h] [e1] [n2] [e2] [n3] ... [t]

        Entity tokens concatenate the fused real representation with the
        imaginary part the frozen scorer uses; relation tokens combine
        real + imaginary for ComplEx or cos/sin phases for RotatE.
        """
        B, max_N = path_nodes.shape
        max_len = 2 * max_N + 1
        device = path_nodes.device
        if max_len > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"Path token length {max_len} exceeds max_tokens="
                f"{self.pos_embedding.num_embeddings}"
            )

        # ── Entity tokens: real (fused) + imaginary (if ComplEx/RotatE) ─
        flat_nodes = path_nodes.reshape(-1)
        real_embs = self.vlkge.get_entity_representations(flat_nodes)

        if self._has_entity_imag:
            imag_embs = self._entity_imag_full(flat_nodes, real_embs)
            flat_emb = torch.cat([real_embs, imag_embs], dim=-1)
        else:
            flat_emb = real_embs

        entity_embs = self.entity_proj(flat_emb).reshape(B, max_N, -1)

        # ── Relation tokens: real + imaginary (if available) ────────────
        # Edge relations
        rel_real = self.vlkge.relation_embeddings(path_edges)
        rel_full = self._build_rel_tokens(rel_real, path_edges)
        edge_embs = self.rel_proj(rel_full)
        edge_embs = edge_embs + self.edge_direction_emb(path_directions)

        # Query relation
        qrel_real = self.vlkge.relation_embeddings(query_rel_id)
        qrel_full = self._build_rel_tokens(qrel_real, query_rel_id)
        qrel_emb = self.rel_proj(qrel_full)  # (B, 1, d_model)

        # ── Interleave ──────────────────────────────────────────────────
        tokens = torch.zeros(B, max_len, self.d_model, device=device)

        tokens[:, 0, :] = self.cls_token.squeeze(0).expand(B, -1)
        tokens[:, 1, :] = qrel_emb.squeeze(1)

        for i in range(max_N):
            tokens[:, 2 + 2 * i, :] = entity_embs[:, i, :]
        for i in range(max_N - 1):
            tokens[:, 3 + 2 * i, :] = edge_embs[:, i, :]

        # ── Position embeddings ─────────────────────────────────────────
        pos_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embedding(pos_ids)

        # ── Node type embeddings ────────────────────────────────────────
        if self.use_node_types:
            type_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
            for i in range(max_N):
                type_ids[:, 2 + 2 * i] = node_types[:, i] + 1
            tokens = tokens + self.node_type_emb(type_ids)

        return tokens

    def _build_rel_tokens(self, rel_real, rel_ids):
        """
        Build full relation representation: [real | imag] or [cos(r) | sin(r)].
        """
        if self._has_rel_imag:
            rel_imag = self.vlkge.relation_embeddings_imag(rel_ids)
            return torch.cat([rel_real, rel_imag], dim=-1)
        if self._is_rotate:
            return torch.cat([torch.cos(rel_real), torch.sin(rel_real)], dim=-1)
        return rel_real

    def _make_padding_mask(self, path_lengths, max_N):
        """Build a token mask from explicit path lengths."""
        B = path_lengths.shape[0]
        max_len = 2 * max_N + 1
        device = path_lengths.device

        mask = torch.ones(B, max_len, dtype=torch.bool, device=device)
        for b in range(B):
            N = int(path_lengths[b].item())
            mask[b, :2 * N + 1] = False
        return mask

    def _find_r2_positions(self, token_mask):
        B, max_len = token_mask.shape
        device = token_mask.device
        r2_pos = torch.full((B,), max_len, dtype=torch.long, device=device)
        valid_lengths = (~token_mask).sum(dim=1)
        N_per_sample = (valid_lengths - 1) // 2
        mask_valid = N_per_sample >= 3
        r2_pos[mask_valid] = 5
        return r2_pos

    def _build_hybrid_attention_mask(self, max_len, r2_positions):
        B = r2_positions.shape[0]
        L = max_len
        device = r2_positions.device

        mask = torch.zeros(B, L, L, device=device)

        for b in range(B):
            r2 = int(r2_positions[b].item())
            if r2 >= L:
                continue
            causal_block = torch.triu(
                torch.ones(L - r2, L - r2, device=device), diagonal=1)
            mask[b, r2:, r2:] = causal_block.masked_fill(
                causal_block.bool(), float('-inf'))

        return mask

    def _compute_reconstruction_l2(self, hidden, true_tokens, r2_positions,
                                    token_mask):
        B, L, d_model = hidden.shape
        device = hidden.device

        avg_l2_values = []
        count_values = []

        for b in range(B):
            r2 = int(r2_positions[b].item())
            if r2 >= L - 1:
                avg_l2_values.append(hidden[b, 0, :].sum() * 0.0)
                count_values.append(0)
                continue

            valid_len = int((~token_mask[b]).sum().item())

            total = hidden[b, 0, :].sum() * 0.0
            count = 0
            for i in range(r2, valid_len - 1):
                if token_mask[b, i] or token_mask[b, i + 1]:
                    continue
                pred = self.recon_head(hidden[b, i, :])
                target = true_tokens[b, i + 1, :]
                l2 = (pred - target).pow(2).sum()
                total = total + l2
                count += 1
            if count > 0:
                avg_l2_values.append(total / count)
            else:
                avg_l2_values.append(hidden[b, 0, :].sum() * 0.0)
            count_values.append(count)

        return torch.stack(avg_l2_values), torch.tensor(
            count_values, dtype=torch.float32, device=device)

    def _readout_rep(self, out, token_mask):
        if self.readout == "cls":
            return out[:, 0, :]
        valid = (~token_mask).unsqueeze(-1).float()
        return (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def _aggregate_candidates(self, scores, labels, query_ids,
                              candidate_tail_ids):
        """Candidate-tail aggregation: score(tail) = max over its probes.

        Returns (cand_scores, cand_labels, cand_query_ids) with one entry per
        unique (query_id, candidate_tail_id) pair inside the batch.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(scores.shape[0]):
            key = (int(query_ids[i].item()), int(candidate_tail_ids[i].item()))
            groups[key].append(i)
        cand_scores = []
        cand_labels = []
        cand_query_ids = []
        for (qid, _tail), idxs in sorted(groups.items()):
            gather = torch.tensor(idxs, device=scores.device, dtype=torch.long)
            cand_scores.append(scores[gather].max())
            cand_labels.append(labels[gather].max())
            cand_query_ids.append(qid)
        if not cand_scores:
            zero = scores.sum() * 0.0
            return zero, zero, zero.long()
        return (torch.stack(cand_scores), torch.stack(cand_labels),
                torch.tensor(cand_query_ids, dtype=torch.long,
                             device=scores.device))

    def forward(self, path_nodes, path_edges, path_directions, node_types,
                query_rel_id, token_mask, candidate_tail_ids, probe_head_ids,
                has_path):
        """
        Args:
            path_nodes:         (B, max_N) padded entity IDs
            path_edges:         (B, max_N-1) padded canonical relation IDs
            path_directions:    (B, max_N-1) 0=forward, 1=reverse traversal
            node_types:         (B, max_N) padded type IDs
            query_rel_id:       (B,) query relation ID
            token_mask:         (B, 2*max_N+1) True=padding, from pamr_collate
            candidate_tail_ids: (B,) explicit candidate tail entity ID
            probe_head_ids:     (B,) query head entity ID
            has_path:           (B,) bool; False for explicit no-path probes

        Returns:
            scores: (B,) pamr_score = vlkge_score + path_residual
        """
        self.vlkge.eval()
        B = path_nodes.shape[0]
        with torch.no_grad():
            vlkge_scores = self.vlkge(
                probe_head_ids, query_rel_id, candidate_tail_ids
            ).to(torch.float32)

        # No-path probes keep the frozen score plus a learned scalar residual.
        scores = vlkge_scores + self.no_path_residual.expand(B)

        if not bool(has_path.any()):
            return scores

        hp = has_path
        sub_path_nodes = path_nodes[hp]
        sub_edges = path_edges[hp]
        sub_directions = path_directions[hp]
        sub_types = node_types[hp]
        sub_query_rel = query_rel_id[hp]
        sub_token_mask = token_mask[hp]

        tokens = self._build_tokens(
            sub_path_nodes, sub_edges, sub_directions, sub_types, sub_query_rel)
        attn_mask = None
        if self.causal:
            max_len = tokens.size(1)
            attn_mask = torch.triu(
                torch.ones(max_len, max_len, device=tokens.device,
                           dtype=torch.bool),
                diagonal=1,
            )

        if self.loss_mode == "reconstruction":
            r2_positions = self._find_r2_positions(sub_token_mask)
            hybrid_mask = self._build_hybrid_attention_mask(
                tokens.size(1), r2_positions)
            out = self.transformer(
                tokens,
                src_key_padding_mask=sub_token_mask,
                mask=hybrid_mask,
            )
            avg_l2, recon_counts = self._compute_reconstruction_l2(
                out, tokens, r2_positions, sub_token_mask)
            self._recon_avg_l2 = avg_l2
            self._recon_counts = recon_counts
            path_residual = self.recon_gate * (-avg_l2)
        else:
            out = self.transformer(
                tokens,
                src_key_padding_mask=sub_token_mask,
                mask=attn_mask,
            )
            rep = self._readout_rep(out, sub_token_mask)
            path_residual = self.readout_mlp(rep).squeeze(-1)

        hp_idx = torch.nonzero(hp, as_tuple=False).squeeze(1)
        scores[hp_idx] = scores[hp_idx] + path_residual
        return scores

    def compute_loss(self, scores, labels, query_ids=None,
                     candidate_tail_ids=None):
        """
        Compute loss based on self.loss_mode.

        Probe-level modes (binary, link_prediction, reconstruction) operate on
        the raw probe scores. Candidate-level modes (grouped_pairwise,
        listwise_softmax, binary_candidate, ranking) first aggregate probes by
        (query_id, candidate_tail_id) with a max reduction and then optimize
        the configured objective over candidate tails.
        """
        if self.loss_mode == "binary":
            return self.criterion(scores, labels)

        elif self.loss_mode == "link_prediction":
            lbl = 2 * labels - 1  # 0→-1, 1→+1
            return F.softplus(-lbl * scores).mean()

        elif self.loss_mode in ("ranking", "grouped_pairwise",
                                 "listwise_softmax", "binary_candidate"):
            if query_ids is None or candidate_tail_ids is None:
                raise ValueError(
                    "query_ids and candidate_tail_ids are required for "
                    f"candidate-level loss mode {self.loss_mode}"
                )
            cand_scores, cand_labels, cand_query_ids = \
                self._aggregate_candidates(scores, labels, query_ids,
                                            candidate_tail_ids)

            if self.loss_mode == "binary_candidate":
                return self.criterion(cand_scores, cand_labels)

            # Group candidates by query over the batch.
            from collections import defaultdict
            query_groups = defaultdict(list)
            for i in range(cand_scores.shape[0]):
                query_groups[int(cand_query_ids[i].item())].append(i)

            zero = cand_scores.sum() * 0.0
            if self.loss_mode in ("ranking", "grouped_pairwise"):
                losses = []
                margin = 1.0
                for qid, idxs in query_groups.items():
                    gather = torch.tensor(
                        idxs, device=cand_scores.device, dtype=torch.long)
                    group_scores = cand_scores[gather]
                    group_labels = cand_labels[gather]
                    pos_mask = group_labels == 1
                    neg_mask = group_labels == 0
                    if not pos_mask.any() or not neg_mask.any():
                        continue
                    pos_scores = group_scores[pos_mask].unsqueeze(1)
                    neg_scores = group_scores[neg_mask].unsqueeze(0)
                    hinges = F.relu(margin - (pos_scores - neg_scores))
                    losses.append(hinges.mean())
                if not losses:
                    return zero
                return torch.stack(losses).mean()

            # listwise_softmax: treat each query's candidate set in the batch
            # as a partial distribution and optimize the positive mass.
            losses = []
            for qid, idxs in query_groups.items():
                gather = torch.tensor(
                    idxs, device=cand_scores.device, dtype=torch.long)
                group_scores = cand_scores[gather]
                group_labels = cand_labels[gather]
                pos_positions = torch.nonzero(
                    group_labels == 1, as_tuple=False)
                if pos_positions.numel() == 0:
                    continue
                target = int(pos_positions[0].item())
                log_probs = F.log_softmax(group_scores, dim=0)
                losses.append(-log_probs[target])
            if not losses:
                return zero
            return torch.stack(losses).mean()

        elif self.loss_mode == "reconstruction":
            # Reconstruction stores diagnostics in forward() but ranks probes
            # by their final residual-augmented scores, so it reuses the
            # candidate-level protocol for a coherent objective.
            if query_ids is None or candidate_tail_ids is None:
                raise ValueError(
                    "query_ids and candidate_tail_ids are required for "
                    "candidate-level loss mode reconstruction"
                )
            cand_scores, cand_labels, cand_query_ids = \
                self._aggregate_candidates(scores, labels, query_ids,
                                            candidate_tail_ids)
            from collections import defaultdict
            query_groups = defaultdict(list)
            for i in range(cand_scores.shape[0]):
                query_groups[int(cand_query_ids[i].item())].append(i)
            losses = []
            for _qid, idxs in query_groups.items():
                gather = torch.tensor(
                    idxs, device=cand_scores.device, dtype=torch.long)
                group_scores = cand_scores[gather]
                group_labels = cand_labels[gather]
                pos_mask = group_labels == 1
                neg_mask = group_labels == 0
                if not pos_mask.any() or not neg_mask.any():
                    continue
                pos_scores = group_scores[pos_mask].unsqueeze(1)
                neg_scores = group_scores[neg_mask].unsqueeze(0)
                hinges = F.relu(1.0 - (pos_scores - neg_scores))
                losses.append(hinges.mean())
            if not losses:
                return cand_scores.sum() * 0.0
            return torch.stack(losses).mean()

        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

