"""Census: how many val/test facts have a true tail OUTSIDE their eval
candidate pool? (The pooled evaluator's guard premise "pools are
supersets of loader facts" was empirically falsified by rel 12 / tail
220397 during wave 78597; the legacy evaluator silently tolerated those
rows via candidates = pool - truths + {t}.)

Scans the full restored val/test splits against the exact eval pools
train_embedding.py feeds evaluate_kge (relation_to_valid_tails_eval_*),
counts outliers per relation, dumps representatives, and dissects the
known failing row (relation 12, tail 220397): which pool-policy branch
(artist2artist split-local vs all-splits) applies and where its tail is
relative to val_pool / all_pool.

Run inside a CPU slurm job (login node serves login only):
  python scripts/census_eval_pools.py \
    --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from pamr.dataloader import KnowledgeGraphDataLoader  # noqa: E402
from pamr import utils  # noqa: E402


def census(frame, pools, artist_ids, relation_to_id, entity_to_id, label):
    counts = {}
    first_rows = []
    heads = frame["head"].map(entity_to_id).tolist()
    rels = frame["relation_id"].astype(int).tolist()
    tails = frame["tail"].map(entity_to_id).tolist()
    t0 = time.time()
    for i, (h, r, t) in enumerate(zip(heads, rels, tails)):
        pool = pools.get(r, [])
        if not pool:
            counts.setdefault(r, {}).setdefault("empty_pool", 0)
            counts[r]["empty_pool"] = counts[r].get("empty_pool", 0) + 1
            continue
        if t in pool:
            continue
        st = counts.setdefault(r, {"n": 0, "examples": []})
        st["n"] += 1
        if len(st["examples"]) < 5:
            st["examples"].append({"row": i, "head": h, "tail": t})
        if len(first_rows) < 40:
            first_rows.append((label, i, h, r, t))
    rel_names = {i: n for n, i in relation_to_id.items()}
    total = sum(st["n"] for st in counts.values() if isinstance(st, dict)
                and "n" in st) + sum(
        st.get("empty_pool", 0) for st in counts.values()
        if isinstance(st, dict))
    print(f"ACCEPTANCE census[{label}] outlier_facts={total:,} "
          f"relations_affected={len(counts)} scan={time.time()-t0:.1f}s",
          flush=True)
    for r in sorted(counts):
        st = counts[r]
        name = rel_names.get(r, f"internal_id_{r}")
        if "n" in st:
            print(f"  rel_id={r} ({name}): {st['n']:,} outliers "
                  f"examples={st['examples'][:2]}")
        if st.get("empty_pool"):
            print(f"  rel_id={r} ({name}): {st['empty_pool']:,} "
                  "rows with EMPTY pool")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="pamr/configs/wikiart_mkg_v2/transe_clip.yaml")
    args = ap.parse_args()

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
    entities, relations = dl.get_entities_and_relations()
    artist_ids = dl.artist2artist_ids
    rel_names = {i: n for n, i in relations.items()}
    print(f"ACCEPTANCE artist2artist_ids={sorted(artist_ids)} -> "
          f"{[rel_names.get(r) for r in sorted(artist_ids)]}")
    print(f"ACCEPTANCE raw relation 12 info: is_artist2artist="
          f"{12 in artist_ids} name={rel_names.get(12, 'INTERNAL_INVERSE')}")

    _, val_data, test_data = dl.split_data()
    pools_val = dl.relation_to_valid_tails_eval_val
    pools_test = dl.relation_to_valid_tails_eval_test
    pool_all = dl.relation_to_valid_tails_all

    # anatomy of the known failing row's universe
    r = 12
    vp = set(pools_val.get(r, []))
    ap_ = set(pool_all.get(r, []))
    print(f"ACCEPTANCE rel12 |val_pool|={len(vp):,} |all_pool|={len(ap_):,} "
          f"identical={vp == ap_} "
          f"tails_only_in_all={len(ap_ - vp):,}")

    out = {}
    out["val"] = census(val_data, pools_val, artist_ids, relations,
                        entities, "val")
    out["test"] = census(test_data, pools_test, artist_ids, relations,
                         entities, "test")
    print("ACCEPTANCE census ALL "
          + json.dumps({k: sum(v.get("n", 0) for v in d.values()
                               if isinstance(v, dict) and "n" in v)
                        for k, d in out.items()}))


if __name__ == "__main__":
    main()
