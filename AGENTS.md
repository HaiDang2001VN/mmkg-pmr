# AGENTS.md — mmkg-pmr

This is the VL-KGE (Vision-Language Knowledge Graph Embedding) project training on WikiArt-MKG-v2.

## Quick start (Vast.ai)

```bash
# Connect to instance
bash scripts/vast-connect.sh <instance-id>

# SSH in
ssh vastai

# Pull data for training
rclone copy gdrive:Study/HCMUS/Grad/Doctor/Thesis/Code/pmr/data/wikiart_mkg_v2/ /workspace/data/wikiart_mkg_v2/ -P
rclone copy gdrive:Study/HCMUS/Grad/Doctor/Thesis/Code/pmr/mmkg-pmr/ /workspace/mmkg-pmr/ -P

# Install deps
cd /workspace/mmkg-pmr && pip install -r requirements.txt -q

# Train ComplEx on WikiArt-MKG-v2
cd vlkge/scripts
python3 train.py --config ../configs/wikiart_mkg_v2/complex_clip.yaml \
  --data_path /workspace/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
  --visual_features_path /workspace/data/wikiart_mkg_v2/features/wikiart_mkg_v2_vf_clip.pkl \
  --textual_features_path /workspace/data/wikiart_mkg_v2/features/wikiart_mkg_v2_tf_clip.pkl \
  --save_path /workspace/checkpoints/complex_wikiart_v2.pt
```

## Architecture

- `vlkge/data/` — dataset CSV files and features
- `vlkge/models/` — KGE models (ComplEx, DistMult, RotatE, TransE) extending VLKGEBase
- `vlkge/scripts/` — training and evaluation scripts
- `vlkge/configs/` — YAML configs per dataset per model
- `vlkge/dataloader.py` — KnowledgeGraphDataLoader with per-relation candidate pools
- `vlkge/helpers.py` — training loop, negative sampling, evaluation
- `vlkge/utils.py` — feature loading, checkpointing, YAML utilities

## Key flags

| Flag | Complex + WikiArt-v2 default |
|------|------------------------------|
| `--model` | ComplEx |
| `--inductive` | true |
| `--modality_asymmetry` | true |
| `--bidirectional_eval` | false |
| `--fusion_mode` | average |
| `--num_neg_samples` | 1 |
| `--epochs` | 20 |
| `--batch_size` | 512 |
| `--lr` | 0.1 |

## Lint / Typecheck

There are no lint or typecheck commands configured for this project.
