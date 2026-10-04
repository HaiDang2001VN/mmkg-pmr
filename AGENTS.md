# AGENTS.md — pamr (PAMR + VL-KGE, WikiArt-MKG-v2)

This repo (`src/`, git root) is the VL-KGE project (multimodal knowledge graph
embedding with an explicit negative-sampled evaluation protocol) extended with
**PAMR** (Path-Aware Multimodal Reasoning): frozen VL-KGE embeddings enter a
transformer as tokens and PAMR learns a path-based residual per candidate tail
(`ŷ = g + Δ`, Δ zero-initialized). The package was renamed `pmr` → `pamr`
(2026-10); no `pmr.` references matter anymore.

Defect register + experiment program: `memory/bugs.md` (READ the §5.4
rediscovery workflow after compaction). Paper status: `memory/wip.md`.

## Layout

- `pamr/dataloader.py` — `KnowledgeGraphDataLoader` (per-relation candidate pools, artist2artist split-local policy, vectorized inverse materialization)
- `pamr/helpers.py` — training loop, negative sampling (Bernoulli + fail-fast contracts), `evaluate_kge`
- `pamr/ranking.py` — THE single ranker (strict/non-strict, multi-positive, candidate aggregation); used by `evaluate.py` and `train_pamr.py`
- `pamr/relation_registry.py` — 40-ID registry, involution invariant, registry hash + manifest
- `pamr/models/embedding/` — `VLKGEBase` + TransE / ComplEx / DistMult / RotatE
- `pamr/models/vlkge/score.py` — scores prepared query CSVs (`query_id,score,is_positive`)
- `pamr/models/pr/preprocess.cpp` (+ `preprocess.py` wrapper) — B1 path extractor worker v2; split-scoped graphs, own-edge masking, completeness audit
- `pamr/models/pr/pamr.py` — PAMR transformer (score contract, freeze/fine-tune gate, listwise temperature)
- `pamr/scripts/` — `train_embedding.py` (KGE), `prepare.py` (queries), `train_pamr.py`, `evaluate.py`, `check_registry.py`, `verify_paths.py` (leakage checks A/B/C)
- `pamr/configs/` — `wikiart_mkg_v2/` (backbones + `transe_struct` for PAR), `eval/` (neg-sampling + score chains), `pr/` (pamr prod + ablations + smoke)

## Environment (compute)

- **FIT cluster** via `ssh FIT`. Cluster rules read order on FIT:
  `~/AGENTS.md` → `/media02/hlhdang/README.md` → `~/admin_note.txt`.
  Full verified handoff (SLURM limits, lifecycle, sbatch patterns, program):
  `memory/bugs.md` §4–§5.3.
- **Universal env**: `/media02/hlhdang/uvenvs/main-py312` (CPython 3.12.14,
  torch 2.11.0+cu130 trio). **NO new venvs** — bind the project with
  `uv lock` + `UV_PROJECT_ENVIRONMENT=/media02/hlhdang/uvenvs/main-py312
  uv sync --locked`; keep added packages minimal and note them in
  `uvenvs/registry.json`.
- uv binary pinned at `tools/uv/uv`; bootstrap shells with
  `source /media02/hlhdang/tools/uvenv.sh`.
- Compute nodes are OFFLINE: everything (env, data, code) staged on the login
  node before `sbatch`.
- **This workstation (Mac)**: analysis, editing, verification only — no
  training, no GPU, no artifact pipeline runs.

## Commands (regen chain order)

```bash
python -m pamr.scripts.check_registry --config configs/wikiart_mkg_v2/transe_clip.yaml --expect-raw 22 --expect-total 40
python -m pamr.scripts.prepare        --config configs/eval/wikiart_mkg_v2_neg_sampling.yaml
python -m pamr.scripts.train_pamr_preprocess?  # see below
python -m pamr.models.pr.preprocess   --config configs/pr/wikiart_mkg_v2_paths.yaml   # path extraction
python -m pamr.scripts.train_embedding --config configs/wikiart_mkg_v2/transe_clip.yaml (--save_path ...)
python -m pamr.scripts.train_pamr     --config configs/pr/pamr.yaml  [ablation: pamr_{finetune,hop3,depth12,cls,lf_temp,seed1,par}.yaml]
python -m pamr.scripts.verify_paths   ...   # checks A/B/C on regenerated artifacts
python -m pamr.models.vlkge.score / pamr.scripts.evaluate --config configs/eval/wikiart_mkg_v2_score*.yaml
```
(Exact flag sets live in each config; smoke variants mirror prod under
`configs/{eval,pr}/smoke_*.yaml`.)

## Protocol facts (locked)

- Relationship IDs: 40-ID registry (22 raw + 18 internal inverse IDs, no new public names; `hasCreatedArtwork` deprecated). Checkpoints/scorers reject relation-count and registry-hash mismatches.
- Backbone: retrained **train-scope TransE** (`transe_wikiart_mkg_v2_train.pt`); path-graph scopes = train (val eval) and train+val (test eval).
- Evaluation: 10 negative candidates per query, 2 seeds (42 + 1) for PAMR; the 4 artist-to-artist target relations under `relation_filter: train_eval`; per-relation split-local candidate pools.
- Path extraction: own query edges masked (relation + registry inverse, both orientations) in EVERY split; one uniform target-traversal flag; prod BFS H=5/F=100/δ=0.5 (H<3 yields zero evidence).
- Frozen VL-KGE stays eval-mode inside PAMR; candidate tails aggregate by `max`; no-path probes keep the direct score + learned per-relation residual.
- Old `assets/{result,average,count}.csv` values are per user decision ignored — fresh regenerated runs are authoritative.

## Path conventions

- Config paths are workspace-relative (`../data/…`, `../checkpoints/…`), resolved as config dir → repo root → cwd.
- Artifact layout lands in `checkpoints/preproc/wikiart_mkg_v2/{full,smoke}/` (queries, paths, scores, metrics); KGE checkpoints go to `checkpoints/vlkge/wikiart_mkg_v2/`.

## Known issues / gotchas

- **torch ≥ 2.6 floor**: CUDA 13 drivers break `nn.MultiheadAttention` CUBLAS on torch 2.5 (mean-pool fallback exists, dormant). FIT universal env runs 2.11.0+cu130.
- **Sample caches v4**: `*.v4.samples.pkl` carries a manifest (registry hash, max_hops, whitelist hash) — mismatched caches rebuild, never silently reused.
- **ID validation**: `PAMRDataset` validates query IDs against entity/relation names; `node_type` tokens are 1/2/3 plus padding offset 0 (embedding size 4).
- **Legacy checkpoints are history only** (`checkpoints/vlkge/wikiart_mkg_v2/*.pt`): trained before the registry/leakage corrections — contract-invalid.
- **Known-swap CSVs**: `result.csv` H1/H3 TransE↔PAMR per-relation columns were swapped — do not trust old per-relation readings; the B-queue regenerates.

## Lint / Typecheck

There are no lint or typecheck commands configured for this project.
Local verification policy: `ast`/`py_compile` gates + targeted CPU micro-tests only.
