"""Throwaway probe: batch-size / sampler / eval-throughput benchmark for the
KGE backbone trainer on the campaign GPU. Protocol-neutral measurements
only — batch_size is a TRAINING-hyperparameter decision for the user; this
probe quantifies the wall-clock and GPU-utilization tradeoffs.

Sections:
  1. sampler wall: CUDA-input tensors (current pre-fix pattern) vs CPU-input
     tensors (post-fix pattern) at bs=512 — quantifies the per-triple
     .item() sync overhead.
  2. full train step (sampler + fwd + loss + bwd + Adagrad step) at
     bs ∈ {512, 2048, 8192}: wall/batch, sampler share, GPU util, peak mem.
  3. eval-side scoring throughput at loader bs ∈ {512, 4096} over a fixed
     slice of val triples (batching is protocol-neutral for evaluation).
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent.parent


def util_poller(stop, samples):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            samples.append(int(out.splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.5)


def bench(fn, n, label, stop_util, util_samples, log):
    util_samples.clear()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    util = (sum(util_samples) / len(util_samples) if util_samples else -1)
    peak = torch.cuda.max_memory_allocated() / 2**20
    line = {"label": label, "wall_per_batch_ms": round(wall / n * 1000, 2),
            "gpu_util_pct": round(util, 1), "peak_alloc_mb": round(peak)}
    print("ACCEPTANCE probe", json.dumps(line), flush=True)
    log.append(line)
    return wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pamr/configs/wikiart_mkg_v2/transe_clip.yaml")
    ap.add_argument("--batches", type=int, default=60,
                    help="batches per bench point")
    ap.add_argument("--eval-triples", type=int, default=8192)
    ap.add_argument("--batch-sizes", default="512,2048,8192")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(SRC))
    from pamr import helpers, utils
    from pamr.dataloader import KnowledgeGraphDataLoader, KGDataset

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = SRC / cfg_path
    cfg = utils.load_yaml_config(cfg_path)
    cfg = utils.resolve_config_paths(cfg, base_dir=cfg_path.parent,
                                     repo_root=SRC)

    t0 = time.perf_counter()
    dl = KnowledgeGraphDataLoader(
        data_path=cfg["data_path"], dataset_name=cfg["dataset"],
        exclude_relations=cfg.get("exclude_relations"),
        exclude_relations_eval=cfg.get("exclude_relations_eval"),
        inverse_relation_pairs=cfg.get("inverse_relation_pairs"),
        symmetric_relations=cfg.get("symmetric_relations"),
        use_per_relation_candidates=cfg.get("use_per_relation_candidates", True),
        artist2artist_relations=cfg.get("artist2artist_relations"),
        bidirectional_eval=False)
    entity_to_id, relation_to_id = dl.get_entities_and_relations()
    train_data, val_data, test_data = dl.split_data()
    filter_map = dl.compute_filter_map()
    # scope-train pools (same semantics as train_embedding.py)
    pool_frame = train_data.copy()
    pool_frame["rid"] = pool_frame["relation_id"].astype(int)
    pool_frame["tid"] = pool_frame["tail"].map(entity_to_id)
    pools = {int(r): sorted(int(x) for x in t)
             for r, t in pool_frame.groupby("rid")["tid"].unique().items()}
    print(f"ACCEPTANCE data-ready in {time.perf_counter()-t0:.0f}s "
          f"(triples={len(train_data):,} pools={len(pools)})", flush=True)

    feat_args = SimpleNamespace(
        use_visual=cfg.get("use_visual", True),
        use_textual=cfg.get("use_textual", True),
        use_relation_features=False,
        visual_features_path=cfg.get("visual_features_path"),
        textual_features_path=cfg.get("textual_features_path"),
        relation_features_path=None,
        normalize_visual=False, normalize_textual=False,
        normalize_relation=False)
    (visual_features, textual_features, relation_features,
     vis_idx, txt_idx) = utils.load_features(
        feat_args, entity_to_id, relation_to_id,
        num_relations=dl.get_num_relations())
    device = torch.device("cuda")
    model, _, _ = helpers.get_model(
        model_name=cfg["model"], num_entities=len(entity_to_id),
        num_relations=dl.get_num_relations(),
        visual_features=visual_features, textual_features=textual_features,
        relation_features=relation_features,
        visual_entity_to_index=vis_idx, textual_entity_to_index=txt_idx,
        embedding_dim=cfg.get("embedding_dim", 768),
        fusion_mode=cfg.get("fusion_mode", "average"),
        use_structural=True, use_visual=True, use_textual=True,
        freeze_visual=True, freeze_textual=True,
        inductive=cfg.get("inductive", True),
        modality_asymmetry=cfg.get("modality_asymmetry", True),
        p_norm=cfg.get("p_norm", 1),
        normalize_relations=cfg.get("normalize_relations", True),
        margin=cfg.get("margin", 12.0), lr=cfg.get("lr", 0.1),
        use_scheduler=False, device=device)
    model.mark_seen_entities(torch.tensor(
        sorted(set(train_data['head'].map(entity_to_id).dropna().astype(int))
               .union(set(train_data['tail'].map(entity_to_id)
                                 .dropna().astype(int)))),
        dtype=torch.long))
    optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg.get("lr", 0.1))
    loss_fn = helpers.compute_logistic_loss
    print("ACCEPTANCE model-ready", flush=True)

    stop = threading.Event()
    util_samples = []
    poll = threading.Thread(target=util_poller, args=(stop, util_samples))
    poll.start()
    log = []

    # ── 1. sampler wall: cuda-inputs vs cpu-inputs (bs 512) ──
    ds = KGDataset(train_data, entity_to_id, relation_to_id)
    loader = DataLoader(ds, batch_size=512, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    it = iter(loader)
    batches = [next(it) for _ in range(args.batches)]
    torch.cuda.synchronize()

    def sampler_cuda():
        i = batches[0]
        g = torch.Generator().manual_seed(7)
        helpers.negative_sampling_per_relation_fast(
            i["head_id"].cuda(), i["relation_id"].cuda(), i["tail_id"].cuda(),
            1, filter_map, pools, g)

    def sampler_cpu():
        i = batches[0]
        g = torch.Generator().manual_seed(7)
        helpers.negative_sampling_per_relation_fast(
            i["head_id"], i["relation_id"], i["tail_id"],
            1, filter_map, pools, g)

    bench(sampler_cuda, args.batches, "sampler@bs512/cuda-inputs", stop,
          util_samples, log)
    bench(sampler_cpu, args.batches, "sampler@bs512/cpu-inputs", stop,
          util_samples, log)

    # ── 2. full train step at each batch size ──
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        loader = DataLoader(ds, batch_size=bs, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        it = iter(loader)
        state = {"n": 0}

        def step():
            i = next(it)
            state["n"] += 1
            heads = i["head_id"].to(device)
            rels = i["relation_id"].to(device)
            tails = i["tail_id"].to(device)
            pos = model(heads, rels, tails)
            g = torch.Generator().manual_seed(11)
            (nh, nr, nt, keep) = helpers.negative_sampling_per_relation_fast(
                heads.cpu(), rels.cpu(), tails.cpu(), 1, filter_map, pools, g)
            neg = model(nh.to(device), nr.to(device), nt.to(device))
            loss = loss_fn(pos, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        try:
            bench(step, args.batches, f"trainstep@bs{bs}", stop,
                  util_samples, log)
        except StopIteration:
            pass

    # ── 3. eval-side scoring throughput (protocol-neutral) ──
    val_ds = KGDataset(val_data.head(args.eval_triples), entity_to_id,
                       relation_to_id)
    for bs in [512, 4096]:
        vloader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        state = {"n": 0}

        def score():
            i = next(viter)
            state["n"] += len(i["head_id"])
            with torch.no_grad():
                model(i["head_id"].to(device), i["relation_id"].to(device),
                      i["tail_id"].to(device))

        viter = iter(vloader)
        try:
            bench(score, max(1, args.eval_triples // bs),
                  f"evalscore@bs{bs}", stop, util_samples, log)
        except StopIteration:
            pass

    stop.set()
    poll.join()
    print("\nACCEPTANCE probe-summary", json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
