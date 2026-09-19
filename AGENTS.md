# AGENTS.md — mmkg-pmr

This is the VL-KGE (Vision-Language Knowledge Graph Embedding) project, now expanded
with PAMR (Path-Aware Multimodal Reasoning).  Trained on WikiArt-MKG-v2.

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

- `pmr/data/` — dataset CSV files and features
- `pmr/models/embedding/` — KGE models (VLKGEBase, TransE, ComplEx, DistMult, RotatE)
- `pmr/models/vlkge/` — VLKGE scoring (`score.py`)
- `pmr/models/pr/` — path-relation module (`preprocess.cpp`, `preprocess.py`, `pamr.py`)
- `pmr/scripts/` — `train_embedding.py`, `train_pamr.py`, `prepare.py`, `evaluate.py`
- `pmr/configs/` — YAML configs: `eval/`, `pr/`, `wikiart_mkg_v2/`, `wikiart_mkg_v1/`, `wn9_img/`
- `pmr/dataloader.py` — KnowledgeGraphDataLoader with per-relation candidate pools
- `pmr/helpers.py` — training loop, negative sampling, evaluation
- `pmr/utils.py` — feature loading, checkpointing, YAML path resolution

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

## Vast.ai workspace layout

The workspace mirrors the local checkpoints structure.  All paths in YAML configs use
relative references (resolved via `resolve_config_paths` relative to repo root
`mmkg-pmr/` or the config file's own directory).

```
/workspace/
├── data/wikiart_mkg_v2/              ← rclone from GDrive (triples CSV + features)
├── mmkg-pmr/                          ← rclone from GDrive (code, pmr package)
├── checkpoints/
│   ├── preproc/wikiart_mkg_v2/
│   │   ├── graph.csv                  ← shared graph (used by preprocess.cpp)
│   │   ├── full/                      ← PROD: 100 % data
│   │   │   ├── train_queries.csv, val_queries.csv, test_queries.csv
│   │   │   ├── metadata.json          ← from prepare.py
│   │   │   ├── vlkge/<datetime>/      ← score + evaluate runs
│   │   │   │   ├── *_scores.csv
│   │   │   │   ├── *_score_metrics.json
│   │   │   │   ├── metrics.json       ← evaluate.py output
│   │   │   │   ├── score_prod.log
│   │   │   │   └── evaluate_prod.log
│   │   │   └── pamr/                  ← preprocessed paths (prod)
│   │   │       ├── train_paths.csv
│   │   │       ├── val_paths.csv
│   │   │       └── test_paths.csv
│   │   └── smoke/                     ← SMOKE: small subset for fast testing
│   │       ├── train_queries.csv, val_queries.csv, test_queries.csv
│   │       ├── vlkge/<datetime>/      ← VLKGE smoke score metrics + eval log
│   │       └── pamr/                  ← PAMR smoke paths + training runs
│   │           ├── *_paths.csv
│   │           └── <datetime>/        ← training metrics + logs
│   ├── vlkge/wikiart_mkg_v2/
│   │   └── complex_wikiart_mkg_v2.pt  ← trained model checkpoint (2.6 GB)
│   └── evaluate/
│       └── vlkge_prod.log
└── logs/                              ← live training / eval logs (tee'd)
```

## Config → checkpoints path mapping

| Config | Key | Resolves to |
|--------|-----|-------------|
| `eval/smoke_score.yaml` | `output_dir` | `../checkpoints/preproc/wikiart_mkg_v2/smoke/vlkge/` |
| `eval/wikiart_mkg_v2_score.yaml` → `prepare_config` | `output_dir` | `../checkpoints/preproc/wikiart_mkg_v2/full/` |
| `pr/smoke_pamr.yaml` | `paths_dir` | `../checkpoints/preproc/wikiart_mkg_v2/smoke/pamr/` |
| `pr/smoke_pamr.yaml` | `queries_dir` | `../checkpoints/preproc/wikiart_mkg_v2/smoke/` |
| `pr/pamr.yaml` | `paths_dir` | `../checkpoints/preproc/wikiart_mkg_v2/full/pamr/` |
| `pr/pamr.yaml` | `queries_dir` | `../checkpoints/preproc/wikiart_mkg_v2/full/` |

Paths are relative to the config file's parent directory.  `resolve_config_paths` tries
candidates in order: config dir → repo root (`mmkg-pmr/`) → cwd.  Fallback uses
`(REPO_ROOT / path).resolve()`.

## Known issues / gotchas

- **Relation checkpoint contract**: WikiArt configs now use a real
  `add_inverse_relations` mapping.  A checkpoint trained before this correction
  may contain 22 relations while current data contains 23.  `score.py` and
  `train_pamr.py` reject this mismatch.  Retrain the VLKGE checkpoint before
  scoring or training PAMR.
- **node_type_emb OOB**: PAMR's `node_type_emb` needs `num_node_types = 4` (types
  1/2/3 plus padding offset), not 3.  Fixed in `pamr.py` and `train_pamr.py`.
- **CUDA 13 driver + PyTorch 2.5**: `nn.MultiheadAttention` CUBLAS ops fail on
  Vast.ai images with CUDA 13 driver.  The PAMR forward has a mean-pool fallback.
  Use a PyTorch 2.6+ Docker image for the real transformer.
- **Dataset validation**: `PAMRDataset` validates query IDs against entity and
  relation names and requires direction-aware path files.  Path caches use a
  versioned filename so old ID-misaligned paths are not silently reused.
- **Path ID mapping**: `preprocess.py` obtains entity/relation IDs from the same
  `KnowledgeGraphDataLoader` configuration used by `prepare.py`.  It writes
  `reverse_edge_type` in the graph and `edge_directions` in path files.

## Lint / Typecheck

There are no lint or typecheck commands configured for this project.
