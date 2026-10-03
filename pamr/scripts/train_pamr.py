"""
Train PAMR (Path-Aware Multimodal Reasoning) model.

Loads pre-extracted paths and frozen VL-KGE embeddings, trains a
Transformer over interleaved entity/edge path sequences.

Usage:
    python -m pamr.scripts.train_pamr --config configs/pr/smoke_pamr.yaml
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pamr.dataloader import KnowledgeGraphDataLoader
from pamr.models.pr.pamr import (PAMR, PAMRDataset, PAMRPairedDataset,
                                QueryGroupBatchSampler, build_query_group_indices,
                                pamr_collate, pamr_paired_collate)
from pamr.ranking import compute_rank_metrics
from pamr import helpers, utils

def per_relation_path_coverage(datasets, relation_to_id):
    """Per-relation path/no-path probe and query counts per split.

    Consumed from dataset samples so whitelist filtering and split scoping
    are honoured exactly as loaded. Pairs-based train datasets are flattened
    (pos, neg) pairs; standalone positives are included once.
    """
    id_to_relation = {rel_id: name for name, rel_id in relation_to_id.items()}

    def iter_samples(ds):
        if hasattr(ds, "samples"):
            return ds.samples
        if hasattr(ds, "pairs"):
            return [sample for pair in ds.pairs
                    for sample in pair if sample is not None]
        return None

    out = {}
    for split_name, ds in datasets.items():
        source = iter_samples(ds)
        if source is None:
            continue
        rel_stats = {}
        for s in source:
            rel_name = id_to_relation.get(int(s["query_rel_id"]),
                                           "internal_inverse")
            st = rel_stats.setdefault(rel_name, {
                "probes": 0,
                "path_probes": 0,
                "no_path_probes": 0,
                "positive_probes": 0,
                "queries": set(),
                "candidate_pairs": set(),
            })
            st["probes"] += 1
            st["path_probes"] += int(s["has_path"])
            st["no_path_probes"] += int(not s["has_path"])
            st["positive_probes"] += int(bool(s["is_positive"]))
            st["queries"].add(s["row_id"][0])
            st["candidate_pairs"].add(
                (s["row_id"][0], s["candidate_tail_id"]))
        summary = {}
        for rel_name, st in sorted(rel_stats.items()):
            n_candidates = max(len(st["candidate_pairs"]), 1)
            summary[rel_name] = {
                "probes": st["probes"],
                "path_probes": st["path_probes"],
                "no_path_probes": st["no_path_probes"],
                "positive_probes": st["positive_probes"],
                "queries_with_probes": len(st["queries"]),
                "candidate_tails": len(st["candidate_pairs"]),
                "mean_paths_per_candidate": round(
                    st["path_probes"] / n_candidates, 6),
            }
        out[split_name] = summary
    return out


SCRIPT_DIR = Path(__file__).parent.resolve()
PKG_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PKG_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PAMR (Path-Aware Multimodal Reasoning)"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to PAMR YAML config file")
    parser.add_argument("--workers", type=int, default=None,
                        help="DataLoader worker processes (default: 4 on GPU, 0 on CPU)")
    return parser.parse_args()


def resolve_config(config_path_str):
    cfg_path = Path(config_path_str).expanduser()
    if not cfg_path.is_absolute():
        for candidate in [Path.cwd() / cfg_path, REPO_ROOT / cfg_path]:
            if candidate.exists():
                cfg_path = candidate.resolve()
                break
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path_str}")
    config = utils.load_yaml_config(cfg_path)
    config = utils.resolve_config_paths(config, base_dir=cfg_path.parent, repo_root=REPO_ROOT)
    return config, cfg_path.parent
def main():
    # Prevent CUDA memory fragmentation — required when a large frozen model
    # (VLKGE, ~1.4 GB) is loaded before the trainable transformer.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()
    config, config_dir = resolve_config(args.config)

    seed = config.get("seed", 42)
    utils.set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_metrics = config.get("log_metrics", False)

    # ── 1. Load VLKGE config (for entity/relation IDs + feature paths) ──
    vlkge_cfg_path = Path(config["vlkge_config"])
    if not vlkge_cfg_path.is_absolute():
        for candidate in [config_dir / vlkge_cfg_path,
                          REPO_ROOT / vlkge_cfg_path,
                          Path.cwd() / vlkge_cfg_path]:
            if candidate.exists():
                vlkge_cfg_path = candidate.resolve()
                break
    if not vlkge_cfg_path.exists():
        raise FileNotFoundError(
            f"VLKGE config not found: {config['vlkge_config']}")

    vlkge_cfg = utils.load_yaml_config(vlkge_cfg_path)
    vlkge_cfg = utils.resolve_config_paths(
        vlkge_cfg, base_dir=vlkge_cfg_path.parent, repo_root=REPO_ROOT)

    # ── 2. Load DataLoader for entity/relation ID maps ────────────────────
    print("Loading entity/relation mappings...")
    dl = KnowledgeGraphDataLoader(
        data_path=vlkge_cfg["data_path"],
        dataset_name=vlkge_cfg["dataset"],
        exclude_relations=vlkge_cfg.get("exclude_relations"),
        exclude_relations_eval=vlkge_cfg.get("exclude_relations_eval"),
        inverse_relation_pairs=vlkge_cfg.get("inverse_relation_pairs"),
        symmetric_relations=vlkge_cfg.get("symmetric_relations"),
        use_per_relation_candidates=vlkge_cfg.get(
            "use_per_relation_candidates", True),
        artist2artist_relations=vlkge_cfg.get("artist2artist_relations"),
        bidirectional_eval=vlkge_cfg.get("bidirectional_eval", False),
    )
    entity_to_id, relation_to_id = dl.get_entities_and_relations()
    print(f"  Entities:  {len(entity_to_id):,}")
    print(f"  Relations: {dl.get_num_relations():,}")

    # Read excluded relations for eval filtering (honours original config)
    exclude_from_pamr_eval = set(
        vlkge_cfg.get("exclude_relations_eval", []))

    # ── 3. Resolve paths ──────────────────────────────────────────────────
    vlkge_checkpoint = config["vlkge_checkpoint"]
    if not Path(vlkge_checkpoint).is_absolute():
        vlkge_checkpoint = str((REPO_ROOT / vlkge_checkpoint).resolve())
    if not os.path.exists(vlkge_checkpoint):
        raise FileNotFoundError(
            f"VLKGE checkpoint not found: {vlkge_checkpoint}")

    paths_dir = config["paths_dir"]
    if not Path(paths_dir).is_absolute():
        paths_dir = str((REPO_ROOT / paths_dir).resolve())
    if not os.path.isdir(paths_dir):
        raise FileNotFoundError(f"Paths directory not found: {paths_dir}")

    # Queries live at a separate location (default: parent of paths_dir)
    queries_dir = config.get("queries_dir")
    if queries_dir is not None:
        if not Path(queries_dir).is_absolute():
            queries_dir = str((REPO_ROOT / queries_dir).resolve())
    else:
        queries_dir = str(Path(paths_dir).parent)
    if not os.path.isdir(queries_dir):
        raise FileNotFoundError(f"Queries directory not found: {queries_dir}")

    # ── 4. Load VLKGE features and build frozen model ─────────────────────
    print("\nLoading frozen VLKGE model...")
    feat_args = SimpleNamespace(
        use_visual=vlkge_cfg.get("use_visual", True),
        use_textual=vlkge_cfg.get("use_textual", True),
        use_relation_features=vlkge_cfg.get("use_relation_features", False),
        visual_features_path=vlkge_cfg.get("visual_features_path"),
        textual_features_path=vlkge_cfg.get("textual_features_path"),
        relation_features_path=vlkge_cfg.get("relation_features_path"),
        normalize_visual=vlkge_cfg.get("normalize_visual", False),
        normalize_textual=vlkge_cfg.get("normalize_textual", False),
        normalize_relation=vlkge_cfg.get("normalize_relation", False),
    )
    (visual_features, textual_features, relation_features,
     visual_entity_to_index, textual_entity_to_index) = utils.load_features(
         feat_args, entity_to_id, relation_to_id,
         num_relations=dl.get_num_relations())

    # Relation IDs are part of the checkpoint contract. Reject a stale
    # checkpoint instead of silently creating zero-valued relation rows.
    relation_registry = dl.get_relation_registry()
    ckpt = torch.load(vlkge_checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    n_ckpt_rels = state["relation_embeddings.weight"].shape[0]
    n_data_rels = dl.get_num_relations()
    if n_ckpt_rels != n_data_rels:
        raise ValueError(
            "VLKGE checkpoint relation count does not match the current "
            f"DataLoader registry ({n_ckpt_rels} vs {n_data_rels}). "
            "Retrain the VLKGE checkpoint with the canonical registry."
        )
    ckpt_registry = ckpt.get("relation_registry") or {}
    if ckpt_registry.get("registry_hash") != relation_registry.registry_hash:
        raise ValueError(
            "VLKGE checkpoint lacks an identical relation-registry hash. "
            "The canonical registry assigns stable IDs independent of row "
            "order, so an equal relation count does not prove ID alignment. "
            "Retrain the VLKGE checkpoint with the current registry."
        )
    model_rel_count = n_data_rels

    vlkge_model, _, _ = helpers.get_model(
        model_name=vlkge_cfg["model"],
        num_entities=len(entity_to_id),
        num_relations=model_rel_count,
        visual_features=visual_features,
        textual_features=textual_features,
        relation_features=relation_features,
        visual_entity_to_index=visual_entity_to_index,
        textual_entity_to_index=textual_entity_to_index,
        embedding_dim=vlkge_cfg.get("embedding_dim", 768),
        fusion_mode=vlkge_cfg.get("fusion_mode", "average"),
        use_structural=vlkge_cfg.get("use_structural", True),
        use_visual=vlkge_cfg.get("use_visual", True),
        use_textual=vlkge_cfg.get("use_textual", True),
        freeze_visual=vlkge_cfg.get("freeze_visual", True),
        freeze_textual=vlkge_cfg.get("freeze_textual", True),
        visual_proj=vlkge_cfg.get("visual_proj", False),
        textual_proj=vlkge_cfg.get("textual_proj", False),
        shared_projection=vlkge_cfg.get("shared_projection"),
        inductive=vlkge_cfg.get("inductive", False),
        modality_asymmetry=vlkge_cfg.get("modality_asymmetry", False),
        normalize_before_fusion=vlkge_cfg.get("normalize_before_fusion", False),
        p_norm=vlkge_cfg.get("p_norm", 1),
        normalize_relations=vlkge_cfg.get("normalize_relations", True),
        margin=vlkge_cfg.get("margin", 12.0),
        lr=vlkge_cfg.get("lr", 0.1),
        use_scheduler=False,
        device=device,
    )
    vlkge_model.to(device)
    vlkge_model.load_state_dict(state)

    vlkge_model.eval()

    # Mark seen entities for inductive using the trained embedding scope,
    # never facts beyond the checkpoint's declared scope.
    vlkge_scope = config.get("vlkge_scope", "train")
    ckpt_scope = ckpt.get("embedding_scope")
    if ckpt_scope is not None and ckpt_scope != vlkge_scope:
        raise ValueError(
            f"PAMR config expects VLKGE scope '{vlkge_scope}' but the "
            f"checkpoint was trained with scope '{ckpt_scope}'"
        )
    if vlkge_cfg.get("inductive", False) and hasattr(
        vlkge_model, "mark_seen_entities"
    ):
        scope_facts = {
            'train': ('train',),
            'train_val': ('train', 'val'),
            'train_val_test': ('train', 'val', 'test'),
        }[vlkge_scope]
        split_frames = {m: df for m, df in zip(
            ('train', 'val', 'test'), dl.split_data())}
        scope_frame = pd.concat(
            [split_frames[m] for m in scope_facts], ignore_index=True)
        seen = set(scope_frame["head"]).union(set(scope_frame["tail"]))
        train_ids = torch.tensor(
            [entity_to_id[e] for e in seen if e in entity_to_id],
            dtype=torch.long,
        )
        vlkge_model.mark_seen_entities(train_ids)

    fine_tune_backbone = bool(config.get("fine_tune_backbone", False))
    lf_temperature = config.get("lf_temperature", 1.0)

    # Freeze all VLKGE params unless the fine-tune ablation is active. In
    # fine-tune mode the params frozen inside the backbone itself
    # (from_pretrained(freeze=True) visual/textual feature tables) stay
    # frozen, while structural/relation/imaginary parameters keep their
    # default trainable state.
    if not fine_tune_backbone:
        for p in vlkge_model.parameters():
            p.requires_grad = False

    vlkge_param_count = sum(p.numel() for p in vlkge_model.parameters())
    if fine_tune_backbone:
        vlkge_trainable = sum(
            p.numel() for p in vlkge_model.parameters() if p.requires_grad)
        print(f"  VLKGE params: {vlkge_param_count:,} "
              f"(fine-tune mode: {vlkge_trainable:,} trainable)")
    else:
        print(f"  VLKGE params (frozen): {vlkge_param_count:,}")

    # ── 5. Create PAMRDataset per split ───────────────────────────────────
    print(f"\nLoading path data from: {paths_dir}")
    if queries_dir != str(Path(paths_dir).parent):
        print(f"  Queries from:          {queries_dir}")
    datasets = {}
    dataloaders = {}

    num_workers = args.workers
    if num_workers is None:
        num_workers = 4 if device.type == "cuda" else 0
    batch_size = config.get("batch_size", 64)
    # Full val/test enumeration in train-sized batches wastes GPU capacity;
    # evaluation pads shorter paths only, so large eval batches are safe.
    eval_batch_size = int(config.get("eval_batch_size", 1024))
    # Candidate-level losses see complete candidate groups per batch when
    # group batching is enabled (recommended for grouped/listwise modes).
    group_batches = bool(config.get("group_batches", True))

    max_hops = config.get("max_hops", 5)
    lf_no_path_policy = config.get("lf_no_path_policy", "exclude")
    if lf_no_path_policy not in ("exclude", "infinite"):
        raise ValueError(
            "lf_no_path_policy must be 'exclude' or 'infinite', got "
            f"'{lf_no_path_policy}'"
        )

    # ── Relation filter (optional) ─────────────────────────────────────
    rel_filter = config.get("relation_filter") or {}
    rel_mode = rel_filter.get("mode", "none")
    rel_list = set(rel_filter.get("relations", []))
    if rel_mode == "none":
        rel_list = None
    wl_train = rel_list if rel_mode == "train_eval" else None
    wl_eval = rel_list if rel_mode in ("train_eval", "eval_only") else None
    if rel_list:
        print(f"  Relation filter: mode={rel_mode}  "
              f"train_wl={'yes' if wl_train else 'no'}  "
              f"eval_wl={'yes' if wl_eval else 'no'}  "
              f"rels={sorted(rel_list)}")

    for split in ["train", "val", "test"]:
        paths_csv = os.path.join(paths_dir, f"{split}_paths.csv")
        queries_csv = os.path.join(queries_dir, f"{split}_queries.csv")

        if not os.path.exists(paths_csv) or not os.path.exists(queries_csv):
            print(f"  SKIP {split}: missing paths or queries CSV")
            continue

        wl = wl_train if split == "train" else wl_eval
        print(f"  Loading {split} split...")
        if split == "train":
            ds = PAMRPairedDataset(paths_csv, queries_csv, entity_to_id,
                                   relation_to_id, max_hops=max_hops,
                                   relation_whitelist=wl,
                                   num_relations=dl.get_num_relations(),
                                   registry_hash=relation_registry.registry_hash)
        else:
            ds = PAMRDataset(paths_csv, queries_csv, entity_to_id,
                             relation_to_id, max_hops=max_hops,
                             relation_whitelist=wl,
                             num_relations=dl.get_num_relations(),
                             registry_hash=relation_registry.registry_hash)
        print(f"    {len(ds):,} samples")
        datasets[split] = ds

        loader_batch_size = batch_size if split == "train" else eval_batch_size
        shuffle = (split == "train")
        collate_fn = pamr_paired_collate if split == "train" else pamr_collate
        loader_kwargs = dict(
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if split == "train" and group_batches:
            group_indices = build_query_group_indices(ds)
            batch_sampler = QueryGroupBatchSampler(group_indices,
                                                   loader_batch_size)
            dl_obj = DataLoader(
                ds,
                batch_sampler=batch_sampler,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            print(f"    Group batching: {len(group_indices):,} query groups "
                  f"→ {len(batch_sampler):,} batches "
                  f"(batch_size={loader_batch_size})")
        else:
            dl_obj = DataLoader(
                ds,
                batch_size=loader_batch_size,
                shuffle=shuffle,
                drop_last=False,
                **loader_kwargs,
            )
        dataloaders[split] = dl_obj

    if "train" not in dataloaders:
        print("\nERROR: No training data found.")
        sys.exit(1)
    n_train_batches = len(dataloaders["train"])
    print(f"  Train batches: {n_train_batches:,}")

    # Build query_id → relation_name map for excluded-relation filtering.
    # Built from dataset samples (not raw CSV) to guarantee query_id alignment
    # after any relation_whitelist filtering has been applied. Samples carry
    # canonical relation IDs; inverting the public registry mapping recovers
    # the raw relation name used for eval-scope exclusions.
    pamr_rel_map = None
    if exclude_from_pamr_eval:
        id_to_relation = {
            rel_id: name for name, rel_id in relation_to_id.items()
        }
        pamr_rel_map = {}
        for split_name in ["val", "test"]:
            if split_name in datasets:
                split_map = {}
                for s in datasets[split_name].samples:
                    rel_name = id_to_relation.get(int(s["query_rel_id"]), "")
                    split_map[s["row_id"][0]] = rel_name
                pamr_rel_map[split_name] = split_map
        counts = ", ".join(f"{s}={len(m):,}" for s, m in pamr_rel_map.items())
        print(f"  Built PAMR rel-map from datasets: {counts} "
              f"for exclude filter ({sorted(exclude_from_pamr_eval)})")

    # ── 6. Build PAMR model ───────────────────────────────────────────────
    max_tokens = config.get("max_tokens", 2 * (max_hops + 1) + 1)
    num_node_types = 4  # 3 types (1,2,3) + 1 offset for padding_idx=0

    print(f"\nBuilding PAMR model...")
    model = PAMR(
        vlkge_model=vlkge_model,
        num_node_types=num_node_types,
        max_tokens=max_tokens,
        num_relation_types=dl.get_num_relations(),
        d_model=config.get("d_model"),           # None → auto-detect from VLKGE
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        dim_feedforward=config.get("dim_feedforward"),  # None → auto 4*d_model
        dropout=config.get("dropout", 0.1),
        causal=config.get("causal", False),
        readout=config.get("readout", "cls"),
        use_node_types=config.get("use_node_types", True),
        loss_mode=config.get("loss_mode", "binary"),
        fine_tune_backbone=config.get("fine_tune_backbone", False),
        lf_temperature=config.get("lf_temperature", 1.0),
        device=device,
    )
    model.to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  VLKGE dims: entity={model._entity_dim}, rel={model._rel_dim}"
          f"  →  d_model={model.d_model}  (auto{' ' if 'd_model' not in config else ' (from config)'})")
    print(f"  PAMR params: {trainable_params:,} trainable / {total_params:,} total")

    # ── 7. Optimizer ──────────────────────────────────────────────────────
    lr = config.get("lr", 0.001)
    epochs = config.get("epochs", 50)
    patience = config.get("patience", 10)
    eval_every = config.get("eval_every", 1)
    eval_max_batches = config.get("eval_max_batches", None)
    max_train_batches = config.get("max_train_batches", None)
    save_checkpoint_flag = config.get("save_checkpoint", True)
    min_eval_coverage = float(config.get("min_eval_coverage", 0.0))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=config.get("weight_decay", 0.01),
    )

    print(f"\n{'=' * 80}")
    print("PAMR TRAINING")
    print(f"{'=' * 80}")
    print(f"  Config:      {args.config}")
    print(f"  Device:      {device}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}  (eval: {eval_batch_size})")
    print(f"  LR:          {lr}")
    print(f"  Patience:    {patience}")
    print(f"  Eval every:  {eval_every}")
    print(f"  Loss mode:   {config.get('loss_mode', 'binary')}")
    print(f"  Log metrics: {log_metrics}")
    if max_train_batches:
        print(f"  Max batches: {max_train_batches}")
    print(f"{'=' * 80}")

    # Defragment GPU memory after loading VLKGE + PAMR models, before cuBLAS
    # handle creation during the first transformer forward.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("  (CUDA cache cleared)")

    # ── 8. Training loop ──────────────────────────────────────────────────
    if log_metrics:
        torch.cuda.reset_peak_memory_stats()
        train_wall_start = time.perf_counter()
        train_cpu_start = time.process_time()
        epoch_metrics = []

    best_val_mrr = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        batch_times = [] if log_metrics else None
        epoch_wall_start = time.perf_counter() if log_metrics else None

        train_loader = dataloaders["train"]
        total_batches = len(train_loader)
        report_every = max(1, total_batches // 100)

        for i, batch in enumerate(train_loader):
            if max_train_batches and i >= max_train_batches:
                break
            if log_metrics:
                b_start = time.perf_counter()

            (path_nodes, path_edges, path_directions, node_types,
             query_rel_ids, is_positives, query_ids, candidate_tail_ids,
             probe_head_ids, _batch_hops, batch_has_path, token_mask) = batch

            path_nodes = path_nodes.to(device)
            path_edges = path_edges.to(device)
            path_directions = path_directions.to(device)
            node_types = node_types.to(device)
            query_rel_ids = query_rel_ids.to(device)
            is_positives = is_positives.to(device)
            query_ids = query_ids.to(device)
            candidate_tail_ids = candidate_tail_ids.to(device)
            probe_head_ids = probe_head_ids.to(device)
            batch_has_path = batch_has_path.to(device)
            token_mask = token_mask.to(device)

            optimizer.zero_grad()
            scores = model(path_nodes, path_edges, path_directions, node_types,
                           query_rel_ids, token_mask, candidate_tail_ids,
                           probe_head_ids, batch_has_path)
            loss = model.compute_loss(scores, is_positives, query_ids,
                                       candidate_tail_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if log_metrics:
                batch_times.append(time.perf_counter() - b_start)

            # Progress logging every 1%
            if i % report_every == 0:
                pct = i * 100 // total_batches
                print(f"  Epoch {epoch}/{epochs} {pct:3d}% "
                      f"({i}/{total_batches}) "
                      f"loss={loss.item():.4f}", flush=True)

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        epoch_wall = (time.perf_counter() - epoch_wall_start
                      if log_metrics else 0)

        # ── Validation ─────────────────────────────────────────────────
        val_metrics = {}
        if eval_every > 0 and epoch % eval_every == 0 and "val" in dataloaders:
            model.eval()
            val_scores = []
            val_labels = []
            val_qids = []
            val_hops = []
            val_cands = []
            val_has_path = []
            val_loss_sum = 0.0
            val_batches = 0

            val_loader = dataloaders["val"]
            effective_val = min(eval_max_batches, len(val_loader)) if eval_max_batches else len(val_loader)
            val_report = max(1, effective_val // 100)

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if eval_max_batches and i >= eval_max_batches:
                        break
                    if i % val_report == 0:
                        pct = i * 100 // effective_val
                        print(f"    Val {pct:3d}% ({i}/{effective_val})",
                              flush=True)

                    (path_nodes, path_edges, path_directions, node_types,
                     query_rel_ids, is_positives, query_ids,
                     candidate_tail_ids, probe_head_ids,
                     batch_hops, batch_has_path, token_mask) = batch

                    path_nodes = path_nodes.to(device)
                    path_edges = path_edges.to(device)
                    path_directions = path_directions.to(device)
                    node_types = node_types.to(device)
                    query_rel_ids = query_rel_ids.to(device)
                    is_positives = is_positives.to(device)
                    query_ids = query_ids.to(device)
                    candidate_tail_ids = candidate_tail_ids.to(device)
                    probe_head_ids = probe_head_ids.to(device)
                    batch_has_path = batch_has_path.to(device)
                    token_mask = token_mask.to(device)

                    scores = model(path_nodes, path_edges, path_directions,
                                   node_types, query_rel_ids, token_mask,
                                   candidate_tail_ids, probe_head_ids,
                                   batch_has_path)
                    loss = model.compute_loss(
                        scores, is_positives, query_ids, candidate_tail_ids)
                    val_loss_sum += loss.item()
                    val_batches += 1

                    val_scores.extend(scores.cpu().numpy().tolist())
                    val_labels.extend(
                        is_positives.cpu().numpy().tolist())
                    val_qids.extend(
                        query_ids.cpu().numpy().tolist())
                    val_hops.extend(
                        batch_hops.cpu().numpy().tolist())
                    val_cands.extend(
                        candidate_tail_ids.cpu().numpy().tolist())
                    val_has_path.extend(
                        batch_has_path.cpu().numpy().tolist())

            avg_val_loss = (val_loss_sum / val_batches
                            if val_batches > 0 else 0.0)
            val_metrics = compute_rank_metrics(
                val_scores, val_labels, val_qids,
                top_k=[1, 3, 5, 10],
                exclude_relations=exclude_from_pamr_eval,
                split="val", all_hops=val_hops,
                expected_queries=getattr(
                    datasets["val"], "num_input_queries", None),
                all_cand_ids=val_cands,
                rel_for_qid=pamr_rel_map,
                all_has_path=val_has_path,
                lf_no_path_policy=lf_no_path_policy,
            )
            val_metrics["loss"] = round(avg_val_loss, 4)

            hits_str = " ".join(
                f"H@{k}={v:.4f}"
                for k, v in sorted(val_metrics["Hits"].items())
                if k in (1, 3, 10)
            )
            hits_ns_str = " ".join(
                f"nsH@{k}={v:.4f}"
                for k, v in sorted(val_metrics["Hits_ns"].items())
                if k in (1, 3, 10)
            )
            has_lf = "MRR_lf" in val_metrics
            lf_str = ""
            ns_lf_str = ""
            if has_lf:
                lf_str = " ".join(
                    f"lfH@{k}={v:.4f}"
                    for k, v in sorted(val_metrics["Hits_lf"].items())
                    if k in (1, 3, 10)
                )
                ns_lf_str = " ".join(
                    f"lfnsH@{k}={v:.4f}"
                    for k, v in sorted(val_metrics["Hits_lfns"].items())
                    if k in (1, 3, 10)
                )
            val_mrr_lf = f"  val_MRRlf={val_metrics['MRR_lf']:.4f}  val_MRRlfns={val_metrics['MRR_lfns']:.4f}" if has_lf else ""
            coverage_str = (f"  val_coverage={val_metrics['coverage']:.4f}"
                            if val_metrics.get("coverage") is not None else "")
            print(f"\n  Epoch {epoch}: "
                  f"train_loss={avg_loss:.4f}  "
                  f"val_loss={avg_val_loss:.4f}  "
                  f"val_MRR={val_metrics['MRR']:.4f}  "
                  f"val_MRRns={val_metrics['MRR_ns']:.4f}"
                  f"{val_mrr_lf}{coverage_str}\n"
                  f"    {hits_str}\n"
                  f"    {hits_ns_str}")
            if lf_str:
                print(f"    {lf_str}\n    {ns_lf_str}")

            # Per-relation breakdown
            per_rel = val_metrics.get("per_relation", {})
            if per_rel:
                print(f"  Per-Relation ({len(per_rel)}):")
                for rel, m in sorted(per_rel.items()):
                    nq = m["count"]
                    h1 = m["Hits"][1]
                    h10 = m["Hits"][10]
                    h1_ns = m.get("Hits_ns", {}).get(1, 0)
                    h10_ns = m.get("Hits_ns", {}).get(10, 0)
                    lf_tag = ""
                    if "MRR_lf" in m:
                        lf_tag = f" (lf={m['MRR_lf']:.4f})"
                    print(f"    {rel:<40s} n={nq:>6d}  "
                          f"MRR={m['MRR']:.4f} (ns={m.get('MRR_ns',0):.4f}{lf_tag})  "
                          f"H@1={h1:.4f} (ns={h1_ns:.4f})  H@10={h10:.4f} (ns={h10_ns:.4f})")

            coverage = val_metrics.get("coverage")
            coverage_ok = coverage is None or coverage >= min_eval_coverage
            if not coverage_ok:
                print("    Coverage notice: val coverage "
                      f"{coverage:.4f} below min_eval_coverage="
                      f"{min_eval_coverage}; best-checkpoint selection "
                      "skipped for this epoch")
            if val_metrics["MRR"] > best_val_mrr and coverage_ok:
                best_val_mrr = val_metrics["MRR"]
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            print(f"\n  Epoch {epoch}: train_loss={avg_loss:.4f}")

        if log_metrics:
            gpu_alloc = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                         if device.type == "cuda" else 0.0)
            gpu_reserved = (torch.cuda.max_memory_reserved() / (1024 ** 2)
                            if device.type == "cuda" else 0.0)
            avg_batch_ms = (sum(batch_times) / len(batch_times) * 1000
                            if batch_times else 0.0)

            entry = {
                "epoch": epoch,
                "train_loss": round(float(avg_loss), 6),
                "val_mrr": round(float(val_metrics.get("MRR", 0)), 6),
                "val_mrr_ns": round(float(val_metrics.get("MRR_ns", 0)), 6),
                "val_metrics": {
                    k: v for k, v in val_metrics.items()
                    if k not in ("Hits", "Hits_ns", "Hits_lf", "Hits_lfns")
                },
                "val_hits": {
                    str(k): round(v, 6)
                    for k, v in val_metrics.get("Hits", {}).items()
                },
                "val_hits_ns": {
                    str(k): round(v, 6)
                    for k, v in val_metrics.get("Hits_ns", {}).items()
                },
                "wall_time_s": round(epoch_wall, 2),
                "gpu_allocated_mb": round(gpu_alloc, 2),
                "gpu_reserved_mb": round(gpu_reserved, 2),
                "avg_batch_time_ms": round(avg_batch_ms, 2),
            }
            if "MRR_lf" in val_metrics:
                entry["val_mrr_lf"] = round(float(val_metrics.get("MRR_lf", 0)), 6)
                entry["val_mrr_lfns"] = round(float(val_metrics.get("MRR_lfns", 0)), 6)
                entry["val_hits_lf"] = {
                    str(k): round(v, 6)
                    for k, v in val_metrics.get("Hits_lf", {}).items()
                }
                entry["val_hits_lfns"] = {
                    str(k): round(v, 6)
                    for k, v in val_metrics.get("Hits_lfns", {}).items()
                }
            epoch_metrics.append(entry)

        # Early stopping
        if patience > 0 and patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(patience={patience})")
            break

    # ── 9. Restore best model ────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best model from epoch {best_epoch} "
              f"(val MRR={best_val_mrr:.4f})")
    else:
        # No validation was run, use current state
        best_val_mrr = 0.0
        best_epoch = epochs

    # ── 10. Test evaluation ──────────────────────────────────────────────
    test_metrics = {}
    if "test" in dataloaders:
        model.eval()
        test_scores = []
        test_labels = []
        test_qids = []
        test_hops = []
        test_cands = []
        test_has_path = []

        print(f"\n{'─' * 60}")
        print("  TEST EVALUATION")
        print(f"{'─' * 60}")

        if log_metrics:
            torch.cuda.reset_peak_memory_stats()
            test_wall_start = time.perf_counter()
            test_cpu_start = time.process_time()

        with torch.no_grad():
            effective_test = min(eval_max_batches, len(dataloaders["test"])) if eval_max_batches else len(
                dataloaders["test"])
            test_report = max(1, effective_test // 100)
            for i, batch in enumerate(dataloaders["test"]):
                if eval_max_batches and i >= eval_max_batches:
                    break

                if i % test_report == 0:
                    pct = i * 100 // effective_test
                    print(f"  Test {pct:3d}% ({i}/{effective_test})",
                          flush=True)

                (path_nodes, path_edges, path_directions, node_types,
                 query_rel_ids, is_positives, query_ids,
                 candidate_tail_ids, probe_head_ids,
                 batch_hops, batch_has_path, token_mask) = batch

                path_nodes = path_nodes.to(device)
                path_edges = path_edges.to(device)
                path_directions = path_directions.to(device)
                node_types = node_types.to(device)
                query_rel_ids = query_rel_ids.to(device)
                is_positives = is_positives.to(device)
                query_ids = query_ids.to(device)
                candidate_tail_ids = candidate_tail_ids.to(device)
                probe_head_ids = probe_head_ids.to(device)
                batch_has_path = batch_has_path.to(device)
                token_mask = token_mask.to(device)

                scores = model(path_nodes, path_edges, path_directions,
                               node_types, query_rel_ids, token_mask,
                               candidate_tail_ids, probe_head_ids,
                               batch_has_path)

                test_scores.extend(scores.cpu().numpy().tolist())
                test_labels.extend(
                    is_positives.cpu().numpy().tolist())
                test_qids.extend(
                    query_ids.cpu().numpy().tolist())
                test_hops.extend(
                    batch_hops.cpu().numpy().tolist())
                test_cands.extend(
                    candidate_tail_ids.cpu().numpy().tolist())
                test_has_path.extend(
                    batch_has_path.cpu().numpy().tolist())

        test_metrics = compute_rank_metrics(
            test_scores, test_labels, test_qids,
            top_k=[1, 3, 5, 10],
            exclude_relations=exclude_from_pamr_eval,
            split="test", all_hops=test_hops,
            expected_queries=getattr(
                datasets["test"], "num_input_queries", None),
            all_cand_ids=test_cands,
            rel_for_qid=pamr_rel_map,
            all_has_path=test_has_path,
            lf_no_path_policy=lf_no_path_policy,
        )

        print(f"\n  Queries:  {test_metrics['num_queries']}")
        if test_metrics.get("coverage") is not None:
            print(f"  Coverage: {test_metrics['coverage']:.4f} "
                  f"({test_metrics['candidate_queries']}/"
                  f"{test_metrics['expected_queries']})")
        print(f"  MRR:      {test_metrics['MRR']:.4f}  (ns={test_metrics['MRR_ns']:.4f})")
        has_lf = "MRR_lf" in test_metrics
        if has_lf:
            print(f"  MRR_lf:   {test_metrics['MRR_lf']:.4f}  (lfns={test_metrics['MRR_lfns']:.4f})")
        for k in sorted(test_metrics["Hits"]):
            ns_v = test_metrics["Hits_ns"].get(k, 0)
            lf_s = ""
            if has_lf:
                lf_s = f"  lf={test_metrics['Hits_lf'].get(k,0):.4f}  lfns={test_metrics['Hits_lfns'].get(k,0):.4f}"
            print(f"  Hits@{k}:   {test_metrics['Hits'][k]:.4f}  (ns={ns_v:.4f}){lf_s}")

        # Per-relation breakdown (test)
        test_per_rel = test_metrics.get("per_relation", {})
        if test_per_rel:
            print(f"\n  Per-Relation ({len(test_per_rel)}):")
            for rel, m in sorted(test_per_rel.items()):
                nq = m["count"]
                h1 = m["Hits"][1]
                h10 = m["Hits"][10]
                h1_ns = m.get("Hits_ns", {}).get(1, 0)
                h10_ns = m.get("Hits_ns", {}).get(10, 0)
                lf_tag = ""
                if "MRR_lf" in m:
                    lf_tag = f" (lf={m['MRR_lf']:.4f})"
                print(f"    {rel:<40s} n={nq:>6d}  "
                      f"MRR={m['MRR']:.4f} (ns={m.get('MRR_ns',0):.4f}{lf_tag})  "
                      f"H@1={h1:.4f} (ns={h1_ns:.4f})  H@10={h10:.4f} (ns={h10_ns:.4f})")

        # Score CSV export (probe-level; aggregation to candidate tails
        # happens in compute_rank_metrics, so keep candidate identity too for
        # downstream evaluators that want tail-level ranking).
        scores_csv_path = os.path.join(paths_dir, "test_scores.csv")
        pd.DataFrame({
            "query_id": test_qids,
            "candidate_tail_id": test_cands,
            "score": test_scores,
            "is_positive": test_labels,
        }).to_csv(scores_csv_path, index=False)
        print(f"\n  Scores CSV → {scores_csv_path}")

        if log_metrics:
            test_wall = time.perf_counter() - test_wall_start
            test_cpu = time.process_time() - test_cpu_start
            test_gpu_alloc = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                              if device.type == "cuda" else 0.0)
            test_gpu_reserved = (torch.cuda.max_memory_reserved() / (1024 ** 2)
                                 if device.type == "cuda" else 0.0)
            test_metrics["wall_time_seconds"] = round(test_wall, 3)
            test_metrics["cpu_time_seconds"] = round(test_cpu, 3)
            test_metrics["gpu_allocated_mb"] = round(test_gpu_alloc, 2)
            test_metrics["gpu_reserved_mb"] = round(test_gpu_reserved, 2)

    # ── 11. Save checkpoint ──────────────────────────────────────────────
    if save_checkpoint_flag and best_state is not None:
        ckpt_dir = os.path.join(paths_dir, "pamr_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "pamr_best.pt")
        torch.save({
            "model_state_dict": best_state,
            "epoch": best_epoch,
            "val_mrr": best_val_mrr,
            "config": {str(k): str(v) if isinstance(v, Path) else v
                       for k, v in config.items()},
        }, ckpt_path)
        print(f"\n  Checkpoint → {ckpt_path}")

    # ── 12. Write metrics JSON ───────────────────────────────────────────
    if log_metrics:
        total_wall = time.perf_counter() - train_wall_start
        total_cpu = time.process_time() - train_cpu_start
        peak_gpu_alloc = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                          if device.type == "cuda" else 0.0)
        peak_gpu_reserved = (torch.cuda.max_memory_reserved() / (1024 ** 2)
                             if device.type == "cuda" else 0.0)

        metrics_out = {
            "config_file": args.config,
            "vlkge_checkpoint": vlkge_checkpoint,
            "fine_tune_backbone": fine_tune_backbone,
            "lf_temperature": lf_temperature,
            "device": str(device),
            "param_count": trainable_params,
            "param_count_total": total_params,
            "vlkge_param_count_frozen": vlkge_param_count,
            "wall_time_seconds": round(total_wall, 2),
            "cpu_time_seconds": round(total_cpu, 2),
            "peak_gpu_allocated_mb": round(peak_gpu_alloc, 2),
            "peak_gpu_reserved_mb": round(peak_gpu_reserved, 2),
            "per_epoch": epoch_metrics,
            "best_epoch": best_epoch,
            "best_val_mrr": round(best_val_mrr, 6),
            "path_coverage": per_relation_path_coverage(
                datasets, relation_to_id),
            "test": {
                k: v for k, v in test_metrics.items()
            },
        }

        metrics_path = os.path.join(paths_dir, "pamr_training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_out, f, indent=2, default=str)
        print(f"\n  Metrics → {metrics_path}")

    print(f"\n{'=' * 80}")
    print("PAMR TRAINING COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
