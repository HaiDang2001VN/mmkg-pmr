"""Diagnose the PAMR encoder: eval-mode output vs a manual reference.

Builds the frozen backbone + PAMR from the campaign config, loads a PAMR
checkpoint, and compares three encoder paths on a real batch:

  1. the default `nn.TransformerEncoder` forward in eval mode
  2. a manual reference loop (explicit SDPA + post-LN) that never uses the
     PyTorch multi-head-attention fast path
  3. the default forward in train mode with dropout disabled

It also rebuilds PAMR with fresh (random) weights and compares the same
paths, and reports the path-residual distribution of the loaded model.

Usage:
    python -m scripts.diagnose_pamr_encoder \
        --config pamr/configs/pr/pamr.yaml \
        --checkpoint ../checkpoints/preproc/wikiart_mkg_v2/full/pamr/pamr_checkpoints/pamr_best.pt
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pamr import helpers, utils
from pamr.dataloader import KnowledgeGraphDataLoader
from pamr.models.pr.pamr import PAMR, PAMRDataset, pamr_collate

REPO_ROOT = Path(__file__).resolve().parent.parent
WHITELIST = ["isInfluencedBy", "isInfluencedOn", "isPupilOf", "isTeacherOf"]


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
    scope = ckpt.get("embedding_scope") or "train"
    scope_modes = {"train": ("train",), "train_val": ("train", "val"),
                   "train_val_test": ("train", "val", "test")}[scope]
    frames = dict(zip(("train", "val", "test"), dl.split_data()))
    frame = pd.concat([frames[m] for m in scope_modes], ignore_index=True)
    seen = set(frame["head"]).union(set(frame["tail"]))
    if hasattr(model, "mark_seen_entities"):
        ids = torch.tensor([entity_to_id[e] for e in seen if e in entity_to_id],
                           dtype=torch.long)
        model.mark_seen_entities(ids)
    return dl, entity_to_id, model


def _build_pamr(cfg, backbone, num_relations, device):
    return PAMR(
        vlkge_model=backbone,
        num_node_types=4,
        max_tokens=cfg.get("max_tokens", 2 * (cfg.get("max_hops", 5) + 1) + 1),
        num_relation_types=num_relations,
        d_model=cfg.get("d_model"),
        num_layers=cfg.get("num_layers", 4),
        num_heads=cfg.get("num_heads", 8),
        dim_feedforward=cfg.get("dim_feedforward"),
        dropout=cfg.get("dropout", 0.1),
        causal=cfg.get("causal", False),
        readout=cfg.get("readout", "cls"),
        use_node_types=cfg.get("use_node_types", True),
        loss_mode=cfg.get("loss_mode", "binary"),
        fine_tune_backbone=cfg.get("fine_tune_backbone", False),
        lf_temperature=cfg.get("lf_temperature", 1.0),
        device=device,
    )


def _manual_encoder(encoder, src, key_padding_mask):
    x = src
    for layer in encoder.layers:
        mha = layer.self_attn
        heads = mha.num_heads
        emb = mha.embed_dim
        head_dim = emb // heads
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(t):
            return t.view(t.shape[0], t.shape[1], heads, head_dim).permute(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        am = None
        if key_padding_mask is not None:
            am = torch.zeros(key_padding_mask.shape[0], 1, 1,
                             key_padding_mask.shape[1], device=x.device,
                             dtype=x.dtype)
            am = am.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        attn = attn.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], emb)
        attn = mha.out_proj(attn)
        x = layer.norm1(x + layer.dropout1(attn))
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(ff))
    return x


def _out_stats(out, mask):
    valid = ~mask
    counts = valid.sum(dim=1)
    first = out[0]
    n_first = int(counts[0].item())
    pad_zero = bool(torch.allclose(first[n_first:],
                                   torch.zeros_like(first[n_first:]))) \
        if n_first < out.shape[1] else True
    return {
        "valid_counts": counts.tolist(),
        "norms": [round(float(v), 4) for v in out.norm(dim=(1, 2))],
        "pos_std_sample0": round(float(first.std(dim=0).mean()), 6),
        "pad_zero_sample0": pad_zero,
        "cls_pairwise_l2_0_1": round(float((out[0, 0] - out[1, 0]).norm()), 8),
        "out_pairwise_l2_0_1": round(float((out[0] - out[1]).norm()), 4),
        "first_coord_sample0": [round(float(v), 6) for v in first[:6, 0]],
    }


def _inspect(tag, mod, tokens, mask, report):
    mod.eval()
    out_default = mod.transformer(tokens, src_key_padding_mask=mask)
    out_manual = _manual_encoder(mod.transformer, tokens, mask)
    report[tag] = {
        "default": _out_stats(out_default, mask),
        "manual": _out_stats(out_manual, mask),
        "default_vs_manual_max_abs": round(
            float((out_default - out_manual).abs().max()), 8),
    }
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        out_nofast = mod.transformer(tokens, src_key_padding_mask=mask)
        report[tag]["fastpath_off_max_abs_vs_manual"] = round(
            float((out_nofast - out_manual).abs().max()), 8)
    except Exception as exc:  # noqa: BLE001
        report[tag]["fastpath_off_error"] = str(exc)
    finally:
        try:
            torch.backends.mha.set_fastpath_enabled(True)
        except Exception:  # noqa: BLE001
            pass

    mod.train()
    for m in mod.transformer.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    out_train = mod.transformer(tokens, src_key_padding_mask=mask)
    report[tag]["train_mode"] = _out_stats(out_train, mask)
    report[tag]["train_vs_manual_max_abs"] = round(
        float((out_train - out_manual).abs().max()), 8)
    mod.eval()
    return out_manual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pamr/configs/pr/pamr.yaml")
    ap.add_argument("--checkpoint",
                    default="../checkpoints/preproc/wikiart_mkg_v2/full/pamr/"
                            "pamr_checkpoints/pamr_best.pt")
    ap.add_argument("--paths",
                    default="../checkpoints/preproc/wikiart_mkg_v2/full/pamr/"
                            "test_paths.csv")
    ap.add_argument("--queries",
                    default="../checkpoints/preproc/wikiart_mkg_v2/"
                            "test_queries.csv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg_path = _rooted(args.config)
    cfg = _load_cfg(cfg_path, cfg_path.parent)
    vcfg_path = (cfg_path.parent / cfg["vlkge_config"]).resolve()
    vcfg = _load_cfg(vcfg_path, vcfg_path.parent)
    checkpoint_path = _rooted(cfg["vlkge_checkpoint"])

    dl, entity_to_id, backbone = _build_backbone(vcfg, checkpoint_path, device)
    num_relations = dl.get_num_relations()
    relation_to_id = dl.get_entities_and_relations()[1]
    registry_hash = dl.get_relation_registry().registry_hash

    model = _build_pamr(cfg, backbone, num_relations, device)
    state = torch.load(_rooted(args.checkpoint), map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    fresh = _build_pamr(cfg, backbone, num_relations, device)
    fresh.eval()

    ds = PAMRDataset(str(_rooted(args.paths)), str(_rooted(args.queries)),
                     entity_to_id, relation_to_id,
                     max_hops=cfg.get("max_hops", 5),
                     relation_whitelist=WHITELIST,
                     num_relations=num_relations,
                     registry_hash=registry_hash)
    indices = list(range(min(args.batch, len(ds))))
    batch = pamr_collate([ds[i] for i in indices])
    (pn, pe, pdir, nt, qr, ip, qi, ct, ph, hops, hp, mask) = batch
    hp = hp.bool()
    if hp.sum() == 0:
        print("ACCEPTANCE diagnose-pamr-encoder failed=no-path-samples")
        sys.exit(1)
    tokens = model._build_tokens(pn[hp], pe[hp], pdir[hp], nt[hp], qr[hp])
    sub_mask = mask[hp]

    report = {"device": str(device), "batch": len(indices),
              "path_rows": int(hp.sum())}
    out_manual = _inspect("trained", model, tokens, sub_mask, report)
    _inspect("fresh", fresh, tokens, sub_mask, report)

    with torch.no_grad():
        vlkge_scores = model.vlkge(ph[hp], qr[hp], ct[hp])
        rep = model._readout_rep(out_manual, sub_mask)
        path_res = model.readout_mlp(rep).squeeze(-1)
        if model.no_path_residual_emb is not None:
            path_res = path_res + model.no_path_residual_emb(qr[hp]).squeeze(-1)
        report["loaded_model"] = {
            "path_residual_mean": round(float(path_res.mean()), 6),
            "path_residual_std": round(float(path_res.std()), 8),
            "path_residual_min": round(float(path_res.min()), 6),
            "path_residual_max": round(float(path_res.max()), 6),
            "vlkge_score_std": round(float(vlkge_scores.std()), 6),
        }

    print("ACCEPTANCE diagnose-pamr-encoder " +
          json.dumps(report, default=str))


if __name__ == "__main__":
    main()
