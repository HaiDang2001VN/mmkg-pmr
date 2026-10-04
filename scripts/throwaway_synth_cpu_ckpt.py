"""Throwaway: synthesize a contract-valid TransE checkpoint for CPU machinery
smokes. Builds the model through pamr.helpers.get_model with the score.py
orchestration (same config, same feature loading) so every key and shape
(including visual/textual tables, raw_margin, seen_in_train) is exact, then
randomizes the trainable weights and attaches the registry manifest +
mapping hashes + train scope metadata."""
import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SRC = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True,
                    help="relation_manifest.json from the run dir")
    ap.add_argument("--entity-mapping", required=True)
    ap.add_argument("--config", required=True,
                    help="backbone config yaml (e.g. transe_clip.yaml)")
    args = ap.parse_args()

    sys.path.insert(0, str(SRC))
    from pamr import helpers, utils     # noqa: E402
    from pamr.dataloader import KnowledgeGraphDataLoader   # noqa: E402
    from pamr.relation_registry import mapping_hash        # noqa: E402

    with open(args.manifest) as f:
        manifest = json.load(f)
    relation_to_id = {name: int(rid)
                      for name, rid in manifest["relation_to_id"].items()}
    num_relations = int(manifest["num_relations"])
    for rid in range(len(relation_to_id), num_relations):
        relation_to_id[f"internal_inverse_{rid}"] = rid

    import pandas as pd                       # noqa: E402
    ent = pd.read_csv(args.entity_mapping)
    entity_to_id = dict(zip(ent["entity"], ent["id"].astype(int)))

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = SRC / cfg_path
    model_cfg = utils.load_yaml_config(cfg_path)
    model_cfg = utils.resolve_config_paths(
        model_cfg, base_dir=cfg_path.parent, repo_root=SRC)

    dl = KnowledgeGraphDataLoader(
        data_path=model_cfg["data_path"],
        dataset_name=model_cfg["dataset"],
        exclude_relations=model_cfg.get("exclude_relations"),
        exclude_relations_eval=model_cfg.get("exclude_relations_eval"),
        inverse_relation_pairs=model_cfg.get("inverse_relation_pairs"),
        symmetric_relations=model_cfg.get("symmetric_relations"),
        use_per_relation_candidates=model_cfg.get(
            "use_per_relation_candidates", True),
        artist2artist_relations=model_cfg.get("artist2artist_relations"),
        bidirectional_eval=model_cfg.get("bidirectional_eval", False),
    )

    feat_args = SimpleNamespace(
        use_visual=model_cfg.get("use_visual", True),
        use_textual=model_cfg.get("use_textual", True),
        use_relation_features=model_cfg.get("use_relation_features", False),
        visual_features_path=model_cfg.get("visual_features_path"),
        textual_features_path=model_cfg.get("textual_features_path"),
        relation_features_path=model_cfg.get("relation_features_path"),
        normalize_visual=model_cfg.get("normalize_visual", False),
        normalize_textual=model_cfg.get("normalize_textual", False),
        normalize_relation=model_cfg.get("normalize_relation", False),
    )
    (visual_features, textual_features, relation_features,
     visual_entity_to_index, textual_entity_to_index) = utils.load_features(
        feat_args, entity_to_id, relation_to_id,
        num_relations=dl.get_num_relations())

    device = torch.device("cpu")
    model, _, _ = helpers.get_model(
        model_name=model_cfg["model"],
        num_entities=len(entity_to_id),
        num_relations=dl.get_num_relations(),
        visual_features=visual_features,
        textual_features=textual_features,
        relation_features=relation_features,
        visual_entity_to_index=visual_entity_to_index,
        textual_entity_to_index=textual_entity_to_index,
        embedding_dim=model_cfg.get("embedding_dim", 768),
        fusion_mode=model_cfg.get("fusion_mode", "average"),
        use_structural=model_cfg.get("use_structural", True),
        use_visual=model_cfg.get("use_visual", True),
        use_textual=model_cfg.get("use_textual", True),
        freeze_visual=model_cfg.get("freeze_visual", True),
        freeze_textual=model_cfg.get("freeze_textual", True),
        visual_proj=model_cfg.get("visual_proj", False),
        textual_proj=model_cfg.get("textual_proj", False),
        shared_projection=model_cfg.get("shared_projection"),
        inductive=model_cfg.get("inductive", False),
        modality_asymmetry=model_cfg.get("modality_asymmetry", False),
        normalize_before_fusion=model_cfg.get("normalize_before_fusion", False),
        p_norm=model_cfg.get("p_norm", 1),
        normalize_relations=model_cfg.get("normalize_relations", True),
        margin=model_cfg.get("margin", 12.0),
        lr=model_cfg.get("lr", 0.1),
        use_scheduler=False,
        device=device,
    )

    # Keep frozen feature tables + buffers from the real construction;
    # scramble the trainable weights so the smoke is not a trivial pass.
    state = copy.deepcopy(model.state_dict())
    g = torch.Generator().manual_seed(0)
    for key in list(state):
        if "visual_embeddings" in key or "textual_embeddings" in key:
            continue
        if "seen_in_train" in key or "raw_margin" in key:
            continue
        w = state[key]
        if torch.is_floating_point(w):
            state[key] = (torch.randn(w.shape, generator=g) * 0.01).to(w.dtype)
    seen_ids = torch.arange(len(entity_to_id))
    state["seen_in_train"] = torch.ones(len(entity_to_id), dtype=torch.bool)

    ckpt = {
        "model_state_dict": state,
        "embedding_scope": "train",
        "relation_registry": manifest,
        "entity_mapping_hash": mapping_hash(entity_to_id),
        "relation_mapping_hash": mapping_hash(relation_to_id),
        "dataset_name": model_cfg["dataset"],
        "model_name": model_cfg["model"],
        "epoch": -1,
        "synthetic": True,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"synth checkpoint -> {args.out} "
          f"(keys={len(state)}, relations={dl.get_num_relations()}, "
          f"entities={len(entity_to_id):,})")


if __name__ == "__main__":
    main()
