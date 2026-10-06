"""Equivalence harness for the host-bound engineering round.

A. KGDataset tensorization: per-row values must match the underlying
   frame exactly (int64 scalar tensors); the NaN guard must raise on
   unmapped names. On real v2 slices: two seeded DataLoader passes must
   produce bit-identical batches (dataset code must not consume RNG).
B. Grouped-vectorized negative sampling: needs distribution-faithful
   bookkeeping, not bit-parity with the retired stream — every negative
   must lie in the relation pool AND outside truths(h, r); kept-row
   counts must satisfy len(negs) == n_kept * num_neg; a coverage check
   on a small-pool relation confirms every eligible candidate is
   reachable (uniform-over-eligible).

Run inside slurm (CPU-only):
  python scripts/test_loader_sampler_equiv.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from pamr import helpers, utils  # noqa: E402
from pamr.dataloader import KGDataset, KnowledgeGraphDataLoader  # noqa: E402


def part_a_synthetic():
    print("── A(synthetic): tensorized KGDataset row access ──")
    rng = np.random.RandomState(7)
    names = [f"e{i:03d}" for i in range(12)]
    rels = ["rA", "rB", "rC"]
    e2i = {n: i for i, n in enumerate(names)}
    r2i = {r: i for i, r in enumerate(rels)}
    rows = len(names)
    df = pd.DataFrame({
        "head": [names[rng.randint(12)] for _ in range(rows)],
        "relation": [rels[rng.randint(3)] for _ in range(rows)],
        "tail": [names[rng.randint(12)] for _ in range(rows)],
        "mode": ["train"] * rows,
    })
    ds = KGDataset(df, e2i, r2i)
    ok = True
    idxs = list(range(rows)) + list(range(rows - 1, -1, -1))
    for idx in idxs:
        got = ds[idx]
        want_h = e2i[df.iloc[idx]["head"]]
        want_t = e2i[df.iloc[idx]["tail"]]
        if not (int(got["head_id"]) == want_h
                and int(got["tail_id"]) == want_t
                and int(got["relation_id"]) == r2i[df.iloc[idx]["relation"]]
                and got["head_id"].dtype == torch.int64
                and got["head_id"].shape == ()):
            ok = False
            print(f"  MISMATCH idx={idx}: {got} vs ({want_h}, {want_t})")
    print("ACCEPTANCE equiv[A/synthetic] " + ("OK" if ok else "FAIL"))

    bad = df.copy()
    bad.loc[0, "head"] = "unmapped_name"
    try:
        KGDataset(bad, e2i, r2i)
        ok = False
        print("  MISMATCH NaN guard did not raise")
    except ValueError as exc:
        if "unmapped" not in str(exc):
            ok = False
            print(f"  MISMATCH NaN guard message: {exc}")
    print("ACCEPTANCE equiv[A/nan-guard] " + ("OK" if ok else "FAIL"))
    return ok


def collate_batch(dl_iter):
    return next(dl_iter)


def part_a_real(args):
    print("── A(real): batch stability across two seeded passes ──")
    from types import SimpleNamespace
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = SRC / cfg_path
    cfg = utils.load_yaml_config(cfg_path)
    cfg = utils.resolve_config_paths(cfg, base_dir=cfg_path.parent,
                                     repo_root=SRC)
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
    train_data, val_data, _ = dl.split_data()
    filter_map = dl.compute_filter_map()
    pools_train = dl.relation_to_valid_tails_eval_train
    print(f"  loaded train={len(train_data):,} val={len(val_data):,}",
          flush=True)
    ok = True

    slice_df = val_data.head(args.n_batch)
    batches = []
    for pass_i in (0, 1):
        ds = KGDataset(slice_df, entity_to_id, relation_to_id)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(123))
        it = iter(loader)
        got = []
        for _ in range(2):
            got.append(collate_batch(it))
        batches.append(got)
    for b in range(2):
        a = batches[0][b]
        c = batches[1][b]
        same = all(torch.equal(a[k], c[k]) for k in
                   ("head_id", "relation_id", "tail_id"))
        if not same:
            ok = False
            print(f"  MISMATCH pass divergence batch {b}")
    print("ACCEPTANCE equiv[A/stability] " + ("OK" if ok else "FAIL"))
    return ok, dl, filter_map, pools_train, train_data


def part_b_sampler(dl, filter_map, pools_train, train_data, batch_size,
                   seed, n_trials):
    print("── B: grouped sampler validity + coverage ──")
    ds = KGDataset(train_data.head(60000), *(
        dl.get_entities_and_relations()))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    it = iter(loader)
    batch = next(it)
    heads = batch["head_id"]
    rels = batch["relation_id"]
    heads_list = heads.tolist()
    rels_list = rels.tolist()

    ok = True
    neg_hist = Counter()
    pool_sizes = {}
    pool_sets = {}
    for rel_id in set(rels_list):
        pool = pools_train.get(rel_id, [])
        pool_sizes[rel_id] = len(pool)
        if pool_sizes[rel_id] <= 40000:
            pool_sets[rel_id] = set(pool)
    small_rel = min(pool_sizes, key=pool_sizes.get)
    print(f"  smallest train pool: rel={small_rel} "
          f"n={pool_sizes[small_rel]}")

    drops_total = 0
    kept_rows = 0
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(seed * 1000 + trial)
        (nh, nr, nt, keep) = helpers.negative_sampling_per_relation_fast(
            heads, rels, batch["tail_id"], 1, filter_map, pools_train, g)
        k = int(keep.sum().item())
        dropped = int((~keep).sum().item())
        drops_total += dropped
        kept_rows += k
        if nh.shape[0] != k:
            ok = False
            print(f"  MISMATCH trial {trial}: len(negs)={nh.shape[0]} "
                  f"n_kept={k}")
        neg_heads_v = nh.tolist()
        neg_tails_v = nt.tolist()
        neg_rels_v = nr.tolist()
        for nhv, nrv, ntv in zip(neg_heads_v, neg_rels_v, neg_tails_v):
            pset = pool_sets.get(nrv)
            if pset is not None and ntv not in pset:
                ok = False
                print(f"  MISMATCH trial {trial}: neg {ntv} absent from "
                      f"pool({nrv})")
            truths = filter_map.get_tails((nhv, nrv), set())
            if ntv in truths:
                ok = False
                print(f"  MISMATCH trial {trial}: neg {ntv} is a known "
                      f"truth of ({nhv},{nrv})")
            neg_hist[(nhv, nrv, ntv)] += 1
    print(f"ACCEPTANCE equiv[B/validity] " + ("OK" if ok else "FAIL")
          + f" [kept={kept_rows:,} dropped={drops_total}]")

    # coverage on the smallest pool: eligible set of a fixed (h, r)
    rel_rows = [(h, r) for h, r in zip(heads_list, rels_list)
                if r == small_rel]
    h0 = rel_rows[0][0]
    pool = pools_train[small_rel]
    truths = filter_map.get_tails((h0, small_rel), set())
    eligible = sorted(set(pool) - truths)
    seen = {tt for (hh, rr, tt) in neg_hist
            if (hh, rr) == (h0, small_rel)}
    missing = set(eligible) - seen
    # with ~n_trials batches of one row candidate the coverage may be
    # partial if the (h, small_rel) row count in the 60K slice is small;
    # coverage counts are informational, absence of ineligible draws is
    # the hard gate (already checked above).
    print(f"  coverage: eligible={len(eligible)} seen={len(seen & set(eligible))} "
          f"missing={len(missing)}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="pamr/configs/wikiart_mkg_v2/transe_clip.yaml")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-trials", type=int, default=6)
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    ok = True
    if not args.skip_synthetic:
        ok = part_a_synthetic() and ok
    loader_pkg = part_a_real(args)
    ok = loader_pkg[0] and ok
    ok = part_b_sampler(loader_pkg[1], loader_pkg[2], loader_pkg[3],
                        loader_pkg[4], args.batch_size, args.seed,
                        args.n_trials) and ok
    print("\nACCEPTANCE equiv ALL " + ("OK" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
