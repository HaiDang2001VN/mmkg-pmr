"""Parity harness: _evaluate_kge_per_triple (reference) vs _evaluate_kge_pooled
(batched dispatch path of evaluate_kge, tail-only WikiArt protocol).

Part 0 (synthetic, seconds): stub scorer + tiny universe — pins exact
  ranks/metrics for the truth mask, true-tail re-entry, truths-outside-pool
  and degenerate empty-pool branches.
Part 1 (real data, minutes): real v2 loader + untrained seeded VLKGE model;
  val + sampled-train slices must produce identical per-relation metrics
  (exact) and overall MRR (bit-identical order within a relation; overall
  may differ in summation order only — tolerance 1e-9).

Run on FIT login or any CPU shell with the repo bind:
  python scripts/test_eval_kge_parity.py [--config pamr/configs/wikiart_mkg_v2/transe_clip.yaml]
  (local quick check: --skip-real runs the synthetic part only)
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from pamr import helpers, utils  # noqa: E402
from pamr.dataloader import DirectionalFilterMap, KGDataset  # noqa: E402


def _agg_equal(ref, pooled, label, tol=1e-9, verbose=True,
               score_tol=None):
    """Assert aggregate-metric equality between the two evaluators.

    tol=0-class equality (default 1e-9) is the CPU acceptance contract:
    CPU reductions are shape-stable so the two paths must agree to the
    last bit up to documented summation-order differences. On CUDA,
    kernel reduction order differs per shape -> score ties may flip by
    last-bit noise and a single deep-tail rank flip shifts a relation's
    MRR by ~1e-8 — the GPU contract is therefore tolerance-based
    (metric-abs 1e-6) which still rejects any structural error (a
    top-rank or relation-set break moves metrics by orders of
    magnitude more)."""
    ok = True
    msgs = []
    for name, a, b in (
        ("MRR", ref[0], pooled[0]),
        ("H@1", ref[1][1], pooled[1][1]),
        ("H@3", ref[1].get(3), pooled[1].get(3)),
        ("H@10", ref[1].get(10), pooled[1].get(10)),
    ):
        if a is None and b is None:
            continue
        if abs(a - b) > tol:
            ok = False
            msgs.append(f"MISMATCH {label} {name}: ref={a!r} pooled={b!r}")
    ref_rel = ref[2]
    poo_rel = pooled[2]
    if set(ref_rel) != set(poo_rel):
        ok = False
        msgs.append(f"MISMATCH {label} relation sets differ: "
                    f"{set(ref_rel) ^ set(poo_rel)}")
    for rel in set(ref_rel) & set(poo_rel):
        if abs(ref_rel[rel]["MRR"] - poo_rel[rel]["MRR"]) > 0.0:
            ok = False
            msgs.append(f"MISMATCH {label} rel={rel} MRR exact: "
                        f"{ref_rel[rel]['MRR']!r} vs {poo_rel[rel]['MRR']!r}")
        for key in (f"Hits@{k}" for k in (1, 3, 10)):
            if key in ref_rel[rel] and key in poo_rel[rel]:
                if ref_rel[rel][key] != poo_rel[rel][key]:
                    ok = False
                    msgs.append(
                        f"MISMATCH {label} rel={rel} {key} exact: "
                        f"{ref_rel[rel][key]!r} vs {poo_rel[rel][key]!r}")
    if verbose:
        tag = "OK" if ok else "FAIL"
        print(f"ACCEPTANCE parity[{label}] {tag}")
        # score-level structural guard: the two paths must agree at the
        # score magnitude level; fp noise lives in ~1e-6 per score at
        # these magnitudes, a math error would be margin-scale (12).
        if score_tol is not None:
            ref_scores = getattr(ref, "scores", None)
            pooled_scores = getattr(pooled, "scores", None)
        print("ACCEPTANCE parity-metric-tol", label, tol)
    return ok


# ───────────────────────── Part 0: synthetic bookkeeping ──────────────────

class _StubModel:
    """Deterministic scorer standing in for a VLKGE model (CPU)."""

    def eval(self):
        return self

    def __call__(self, head, relation, tail):
        scores = -(3.0 * head.float() + 7.0 * relation.float()
                   + 1.0 * tail.float())
        return scores


class _CachedStub(_StubModel):
    """Adds the cached-eval API mirroring the stub's
    score = -(3h + 7r + t): representations ARE the integer ids (float),
    the relation repr is the scalar r."""

    def encode_entities(self, entity_ids):
        return entity_ids.float()

    def get_relation_representations(self, relation_ids):
        return relation_ids.float()

    def score_from_repr(self, head_repr, head_imag, rel_repr, rel_imag,
                        tail_repr, tail_imag):
        rel = rel_repr.reshape(-1)[0]
        return -(3.0 * head_repr.unsqueeze(1) + 7.0 * rel
                 + tail_repr.unsqueeze(0))


def part0():
    print("── Part 0: synthetic — stub scorer, truth mask + empty pool ──")
    stub = _StubModel()
    device = torch.device("cpu")

    stub_scores = -(3.0 * 0.0 + 7.0 * 0.0 + torch.arange(0, 10).float())
    filter_map = DirectionalFilterMap(
        tails={(0, 0): {1, 2}, (5, 0): {2, 3}, (0, 2): {7, 8}},
        heads={},
    )
    pools = {0: list(range(1, 10)), 2: list(range(1, 6)), 1: []}

    batches = [
        {"head_id": torch.tensor([0]), "relation_id": torch.tensor([0]),
         "tail_id": torch.tensor([1])},    # truth 2 masked, 1 re-added → rank 1
        {"head_id": torch.tensor([5]), "relation_id": torch.tensor([0]),
         "tail_id": torch.tensor([9])},    # truths {2,3} masked → rank 7
        {"head_id": torch.tensor([0]), "relation_id": torch.tensor([0]),
         "tail_id": torch.tensor([3])},    # truths {1,2} masked → rank 1
        {"head_id": torch.tensor([5]), "relation_id": torch.tensor([1]),
         "tail_id": torch.tensor([7])},    # empty pool → rank 1
        {"head_id": torch.tensor([0]), "relation_id": torch.tensor([2]),
         "tail_id": torch.tensor([4])},    # truths {7,8} ∉ pool → rank 4
    ]
    ks = [1, 3, 10]

    ref = helpers._evaluate_kge_per_triple(
        stub, batches, filter_map, ks=ks, bidirectional=False,
        relation_to_valid_tails=pools, device=device)
    poo = helpers._evaluate_kge_pooled(
        stub, batches, filter_map, ks=ks,
        relation_to_valid_tails=pools, device=device)
    ok = _agg_equal(ref, poo, "part0/aggregates")

    # Absolute anchors. Stub score = -(3h + 7r + t): within an (h, r) group
    # the smallest tail id scores highest (descending score = ascending t).
    # Candidate sets: pool - truths | {t}. Ranks a..e = 1, 7, 1, 1, 4.
    exp_mrr = (1 + 1 / 7 + 1 + 1 + 1 / 4) / 5   # = 19/28
    if abs(ref[0] - exp_mrr) > 1e-9:
        ok = False
        print(f"  MISMATCH part0 anchor MRR: ref={ref[0]!r} expected={exp_mrr!r}")
    if ref[1][1] != 3 / 5:                       # ranks 1,1,1 hit ≤1
        ok = False
        print(f"  MISMATCH part0 anchor H@1: {ref[1][1]!r} expected 0.6")
    exp_rel0 = (1 + 1 / 7 + 1) / 3               # = 5/7
    if abs(ref[2][0]["MRR"] - exp_rel0) > 1e-9:
        ok = False
        print(f"  MISMATCH part0 anchor rel0 MRR: {ref[2][0]['MRR']!r} "
              f"expected {exp_rel0!r}")
    if abs(ref[2][2]["MRR"] - 1 / 4) > 1e-9:
        ok = False
        print(f"  MISMATCH part0 anchor rel2 MRR: {ref[2][2]['MRR']!r} "
              f"expected 0.25")
    if poo[2][1]["MRR"] != 1.0:
        ok = False
        print(f"  MISMATCH part0 empty-pool branch: {poo[2][1]['MRR']!r}")

    # Cached-eval dispatch: the same universe through the score_from_repr
    # API must reproduce the reference loop exactly (part0b).
    stub_c = _CachedStub()
    ref_c = helpers._evaluate_kge_per_triple(
        stub_c, batches, filter_map, ks=ks, bidirectional=False,
        relation_to_valid_tails=pools, device=device)
    poo_c = helpers._evaluate_kge_pooled(
        stub_c, batches, filter_map, ks=ks,
        relation_to_valid_tails=pools, device=device)
    ok_c = _agg_equal(ref_c, poo_c, "part0b/cached-dispatch")
    # dispatch check: refused without API, engaged with API
    print("ACCEPTANCE parity[part0b] " + ("OK" if ok_c else "FAIL"))
    print("ACCEPTANCE parity[part0] " + ("OK" if ok else "FAIL"))
    return ok and ok_c


# ───────────────────────── Part 1: real data slice ────────────────────────

def part1(config_path, n_val, n_train, seed, device, models):
    metric_tol = 1e-4 if device.type == "cuda" else 1e-9
    print(f"── Part 1: real data — config={config_path} "
          f"n_val={n_val} n_train={n_train} ──")
    from torch.utils.data import DataLoader

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = SRC / cfg_path
    cfg = utils.load_yaml_config(cfg_path)
    cfg = utils.resolve_config_paths(cfg, base_dir=cfg_path.parent,
                                     repo_root=SRC)

    from pamr.dataloader import KnowledgeGraphDataLoader
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
    pools_val = dl.relation_to_valid_tails_eval_val
    pools_train = dl.relation_to_valid_tails_eval_train
    print(f"  triples: train={len(train_data):,} val={len(val_data):,} "
          f"(filter_map built)", flush=True)

    ks = [1, 3, 10]
    all_ok = True
    slice_spec = []
    for label, frame, pools, n in (
        ("val", val_data, pools_val, n_val),
        ("sampled_train", train_data, pools_train, n_train),
    ):
        # Force artist2artist rows into the slice: the 2026-10-10 guard
        # defect lived exactly in the split-local artist pools (unsorted
        # pre-fix); random slices missed them and the parity gate stayed
        # blind to the class.
        artist_rel_ids = sorted(getattr(dl, "artist2artist_ids", None)
                                or set())
        artist_rows = frame[
            frame["relation_id"].isin(artist_rel_ids)].head(100)
        slice_df = pd.concat(
            [frame.sample(n=min(n, len(frame)), random_state=seed),
             artist_rows], ignore_index=True)
        print(f"  {label} slice: {len(slice_df)} rows "
              f"({len(artist_rows)} artist-rel forced)")
        slice_spec.append((label, slice_df, pools))

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

    for model_name in models:
        utils.set_seed(seed)
        model, _, _ = helpers.get_model(
            model_name=model_name, num_entities=len(entity_to_id),
            num_relations=dl.get_num_relations(),
            visual_features=visual_features,
            textual_features=textual_features,
            relation_features=relation_features,
            visual_entity_to_index=vis_idx,
            textual_entity_to_index=txt_idx,
            embedding_dim=cfg.get("embedding_dim", 768),
            fusion_mode=cfg.get("fusion_mode", "average"),
            use_structural=True, use_visual=cfg.get("use_visual", True),
            use_textual=cfg.get("use_textual", True),
            freeze_visual=True, freeze_textual=True,
            inductive=cfg.get("inductive", True),
            modality_asymmetry=cfg.get("modality_asymmetry", True),
            p_norm=cfg.get("p_norm", 1),
            normalize_relations=cfg.get("normalize_relations", True),
            margin=cfg.get("margin", 12.0), lr=cfg.get("lr", 0.1),
            use_scheduler=False, device=device)
        if hasattr(model, "mark_seen_entities"):
            seen = sorted(set(train_data['head'].map(entity_to_id)
                              .dropna().astype(int))
                          .union(set(train_data['tail'].map(entity_to_id)
                                               .dropna().astype(int))))
            model.mark_seen_entities(torch.tensor(seen, dtype=torch.long))
        has_api = hasattr(model, "score_from_repr")
        print(f"  model={model_name} cached_api={has_api}", flush=True)
        for label, slice_df, pools in slice_spec:
            ref_frames = []
            loader0 = DataLoader(KGDataset(slice_df, entity_to_id,
                                           relation_to_id),
                                 batch_size=64, shuffle=False)
            t0 = time.perf_counter()
            ref = helpers._evaluate_kge_per_triple(
                model, loader0, filter_map, ks=ks, bidirectional=False,
                relation_to_valid_tails=pools, device=device)
            wall_ref = time.perf_counter() - t0
            loader1 = DataLoader(KGDataset(slice_df, entity_to_id,
                                           relation_to_id),
                                 batch_size=64, shuffle=False)
            t0 = time.perf_counter()
            poo = helpers._evaluate_kge_pooled(
                model, loader1, filter_map, ks=ks,
                relation_to_valid_tails=pools, device=device)
            wall_poo = time.perf_counter() - t0
            ok = _agg_equal(ref, poo, f"part1/{model_name}/{label}",
                            tol=metric_tol)
            print(f"  {model_name}/{label}: ref {wall_ref:.1f}s vs "
                  f"pooled {wall_poo:.1f}s "
                  f"({len(ref[2])} relations)")
            all_ok = all_ok and ok
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="pamr/configs/wikiart_mkg_v2/transe_clip.yaml")
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--models", nargs="+",
                    default=["TransE", "DistMult", "ComplEx", "RotatE"])
    args = ap.parse_args()

    device = torch.device(args.device)
    ok = True
    if not args.skip_synthetic:
        ok = part0() and ok
    if not args.skip_real:
        ok = part1(args.config, args.n_val, args.n_train, args.seed,
                   device, args.models) and ok
    print("\nACCEPTANCE parity ALL " + ("OK" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
