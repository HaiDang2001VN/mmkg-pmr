"""Throwaway: synthesize a contract-valid TransE checkpoint for CPU machinery
smokes (random weights; registry manifest + mapping hashes + train scope
carried so train_pamr.py / vlkge score.py load it without rejections)."""
import argparse
import importlib.util
import json
import os

import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mappings", required=True,
                    help="entity_mapping.csv from the run dir")
    ap.add_argument("--relations", required=True,
                    help="relation_mapping.csv from the run dir")
    ap.add_argument("--manifest", required=True,
                    help="relation_manifest.json from the run dir")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location(
        "pamr_registry", os.path.join(os.path.dirname(__file__),
                                      "pamr", "relation_registry.py"))
    reg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reg)

    ent = pd.read_csv(args.mappings)
    rel = pd.read_csv(args.relations)
    entity_to_id = dict(zip(ent["entity"], ent["id"].astype(int)))
    relation_to_id = dict(zip(rel["relation"], rel["id"].astype(int)))

    with open(args.manifest) as f:
        manifest = json.load(f)

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    dim = 768
    print(f"synth: entities={num_entities:,} relations={num_relations} dim={dim}")

    state = {
        "entity_embeddings.weight": torch.randn(num_entities, dim) * 0.01,
        "relation_embeddings.weight": torch.randn(num_relations, dim) * 0.01,
        "seen_in_train": torch.zeros(num_entities, dtype=torch.bool),
    }
    ckpt = {
        "model_state_dict": state,
        "embedding_scope": "train",
        "relation_registry": manifest,
        "entity_mapping_hash": reg.mapping_hash(entity_to_id),
        "relation_mapping_hash": reg.mapping_hash(relation_to_id),
        "dataset_name": "wikiart_mkg_v2",
        "model_name": "TransE",
        "epoch": -1,
        "synthetic": True,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"synth checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
