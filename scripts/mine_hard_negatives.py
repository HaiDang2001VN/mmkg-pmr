"""Mine hard negatives for the journal-v2 training arm.

For every train triple in the target relations, sample M candidates from the
relation's evaluation pool (split-local for artist-to-artist relations),
exclude known train truths and the positive tail, score the candidates with
the frozen backbone, and keep the K highest-scoring ones. The resulting train
queries file follows the prepare.py schema so preprocess.py can extract paths
for the same probes.

Usage:
    python -m scripts.mine_hard_negatives \
        --config pamr/configs/eval/wikiart_mkg_v2_v2_random.yaml \
        --out-dir ../checkpoints/preproc/wikiart_mkg_v2_v2/hard/queries \
        --m 100 --k 10 --seed 42
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pamr import helpers, utils
from pamr.dataloader import KnowledgeGraphDataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELD_NAMES = ["query_id", "head", "relation", "tail", "head_id",
               "relation_id", "tail_id", "is_positive", "split"]


def _rooted(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _load_cfg(path, base_dir):
    cfg = utils.load_yaml_config(path)
    return utils.resolve_config_paths(cfg, base_dir=base_dir, repo_root=REPO_ROOT)


def _build_backbone(vcfg, checkpoint_path, device):
    dl = KnowledgeGraphDataLoader(
        data_path=vcfg["data_path"],
        dataset_name=vcfg["dataset"],
        exclude_relations=vcfg.get("exclude_relations"),
        exclude_relations_eval=vcfg.get("exclude_relations_eval"),
        inverse_relation_pairs=vcfg.get("inverse_relation_pairs"),
        symmetric_relations=vcfg.get("symmetric_relations"),
        use_per_relation_candidates=vcfg.get("use_per_relation_candidates", True),
        artist2artist_relations=vcfg.get("artist2artist_relations"),
        bidirectional_eval=vcfg.get("bidirectional_eval", False),
    )
    entity_to_id, relation_to_id = dl.get_entities_and_relations()
    feat_args = SimpleNamespace(
        use_visual=vcfg.get("use_visual", True),
        use_textual=vcfg.get("use_textual", True),
        use_relation_features=vcfg.get("use_relation_features", False),
        visual_features_path=vcfg.get("visual_features_path"),
        textual_features_path=vcfg.get("textual_features_path"),
        relation_features_path=vcfg.get("relation_features_path"),
        normalize_visual=vcfg.get("normalize_visual", False),
        normalize_textual=vcfg.get("normalize_textual", False),
        normalize_relation=vcfg.get("normalize_relation", False),
    )
    (visual_features, textual_features, relation_features,
     visual_index, textual_index) = utils.load_features(
        feat_args, entity_to_id, relation_to_id,
        num_relations=dl.get_num_relations())
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    n_rels = state["relation_embeddings.weight"].shape[0]
    registry = dl.get_relation_registry()
    if n_rels != dl.get_num_relations():
        raise ValueError("checkpoint relation count mismatch")
    if (ckpt.get("relation_registry") or {}).get("registry_hash") != registry.registry_hash:
        raise ValueError("checkpoint registry hash mismatch")
    model, _, _ = helpers.get_model(
        model_name=vcfg["model"],
        num_entities=len(entity_to_id),
        num_relations=dl.get_num_relations(),
        visual_features=visual_features,
        textual_features=textual_features,
        relation_features=relation_features,
        visual_entity_to_index=visual_index,
        textual_entity_to_index=textual_index,
        embedding_dim=vcfg.get("embedding_dim", 768),
        fusion_mode=vcfg.get("fusion_mode", "average"),
        use_structural=vcfg.get("use_structural", True),
        use_visual=vcfg.get("use_visual", True),
        use_textual=vcfg.get("use_textual", True),
        freeze_visual=vcfg.get("freeze_visual", True),
        freeze_textual=vcfg.get("freeze_textual", True),
        visual_proj=vcfg.get("visual_proj", False),
        textual_proj=vcfg.get("textual_proj", False),
        shared_projection=vcfg.get("shared_projection"),
        inductive=vcfg.get("inductive", False),
        modality_asymmetry=vcfg.get("modality_asymmetry", False),
        normalize_before_fusion=vcfg.get("normalize_before_fusion", False),
        p_norm=vcfg.get("p_norm", 1),
        normalize_relations=vcfg.get("normalize_relations", True),
        margin=vcfg.get("margin", 12.0),
        lr=vcfg.get("lr", 0.1),
        use_scheduler=False,
        device=device,
    )
    model.load_state_dict(state)
    model.eval()
    if hasattr(model, "mark_seen_entities"):
        train = dl.split_data()[0]
        seen = set(train["head"]).union(set(train["tail"]))
        model.mark_seen_entities(torch.tensor(
            [entity_to_id[e] for e in seen if e in entity_to_id],
            dtype=torch.long))
    return dl, entity_to_id, relation_to_id, model


def _score(model, h_id, r_id, candidates, device, batch=4096):
    scores = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch):
            chunk = torch.as_tensor(candidates[start:start + batch],
                                    dtype=torch.long, device=device)
            h = torch.full_like(chunk, int(h_id))
            r = torch.full_like(chunk, int(r_id))
            scores.append(model(h, r, chunk).cpu().numpy())
    return np.concatenate(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="pamr/configs/eval/wikiart_mkg_v2_v2_random.yaml")
    ap.add_argument("--vlkge-config",
                    default="pamr/configs/wikiart_mkg_v2/transe_clip.yaml")
    ap.add_argument("--checkpoint",
                    default="../checkpoints/vlkge/wikiart_mkg_v2/"
                            "transe_wikiart_mkg_v2_train.pt")
    ap.add_argument("--out-dir",
                    default="../checkpoints/preproc/wikiart_mkg_v2_v2/"
                            "hard/queries")
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pcfg_path = _rooted(args.config)
    pcfg = _load_cfg(pcfg_path, pcfg_path.parent)
    vcfg_path = _rooted(args.vlkge_config)
    vcfg = _load_cfg(vcfg_path, vcfg_path.parent)
    dl, entity_to_id, relation_to_id, model = _build_backbone(
        vcfg, _rooted(args.checkpoint), device)
    id_to_entity = {v: k for k, v in entity_to_id.items()}

    target = set((pcfg.get("relation_filter") or {}).get("relations", []))
    if not target:
        raise ValueError("prepare config lacks relation_filter.relations")
    train = dl.split_data()[0]
    train = train[(~train["is_inverse"].astype(bool))
                  & (train["relation"].isin(target))].reset_index(drop=True)
    pools = dl.relation_to_valid_tails_eval_train
    rel_ids = sorted(relation_to_id[r] for r in target
                     if r in relation_to_id)
    truth_map = dl.compute_filter_maps_by_scope(relation_ids=rel_ids)["train"]

    rng = np.random.default_rng(args.seed)
    rows = []
    skipped = 0
    qid = 0
    hard_above_pos = 0
    for row in train.itertuples(index=False):
        h_id = entity_to_id[row.head]
        r_id = relation_to_id[row.relation]
        t_id = entity_to_id[row.tail]
        pool = np.asarray(pools.get(r_id, []), dtype=np.int64)
        truths = truth_map.get_tails((h_id, r_id), set())
        excluded = set(int(x) for x in truths)
        excluded.add(int(t_id))
        if pool.size == 0:
            skipped += 1
            continue
        eligible = pool[~np.isin(pool, np.fromiter(excluded, dtype=np.int64,
                                                   count=len(excluded)))]
        if eligible.size < args.k:
            skipped += 1
            continue
        n = min(args.m, eligible.size)
        cand = rng.choice(eligible, size=n, replace=False)
        cand_scores = _score(model, h_id, r_id, cand, device)
        order = np.argsort(-cand_scores)[:args.k]
        hard = cand[order]
        pos_score = float(_score(model, h_id, r_id,
                                 np.asarray([t_id]), device)[0])
        hard_above_pos += int((cand_scores[order] > pos_score).sum())
        rows.append(dict(zip(FIELD_NAMES, [
            qid, row.head, row.relation, row.tail, h_id, r_id, t_id, 1,
            "train"])))
        for c in hard:
            rows.append(dict(zip(FIELD_NAMES, [
                qid, row.head, row.relation, id_to_entity[int(c)],
                h_id, r_id, int(c), 0, "train"])))
        qid += 1

    out_dir = _rooted(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "train_queries.csv"
    pd.DataFrame(rows, columns=FIELD_NAMES).to_csv(out_csv, index=False)
    stats = {
        "queries": qid,
        "rows": len(rows),
        "skipped": skipped,
        "m": args.m,
        "k": args.k,
        "hard_negatives_above_positive": hard_above_pos,
        "seed": args.seed,
        "out_csv": str(out_csv),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("ACCEPTANCE hard-mining " + json.dumps(stats))


if __name__ == "__main__":
    main()
