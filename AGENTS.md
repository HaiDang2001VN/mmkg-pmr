# AGENTS.md — pamr (PAMR + VL-KGE, WikiArt-MKG-v2)

This repo (`src/`, git root) is the VL-KGE project (multimodal knowledge graph
embedding with an explicit negative-sampled evaluation protocol) extended with
**PAMR** (Path-Aware Multimodal Reasoning): frozen VL-KGE embeddings enter a
transformer as tokens and PAMR learns a path-based residual per candidate tail
(`ŷ = g + Δ`, Δ zero-initialized). The package was renamed `pmr` → `pamr`
(2026-10); no `pmr.` references matter anymore.

Defect register + experiment program: `memory/bugs.md` (READ the §6.4
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
  `memory/bugs.md` §4–§6.
- **Environment = uv WORKSPACE** (final era, 2026-10-03+): workspace root
  `/media02/hlhdang/projects/pyproject.toml` (virtual; members ejepwm,
  DreamZero-Project-IML, akg-sv, **pamr/src**); THE lock is
  `projects/uv.lock` (union lock, ~509 pkgs); ONE shared venv
  `/media02/hlhdang/uvenvs/t251cu121-py312` (CPython 3.12.14,
  torch 2.5.1+cu121 / tv 0.20.1 / ta 2.5.1, numpy 2.5.3, pandas 3.0.6,
  sklearn 1.9.1). **NO new venvs/conda/pip-venv — ever**; re-lock via
  `cd /media02/hlhdang/projects && uv lock && uv sync --locked --no-dev
  --all-packages` (uv binary pinned at `tools/uv/uv`); only the torch
  triad + py3.12 are pinned — resolver deadlocks extend the root
  `[tool.uv] override-dependencies` list, never hand-editing pins.
- Runtime: `source /media02/hlhdang/tools/uvenv_ws_env.sh`, then run via
  the direct venv python `uvenvs/t251cu121-py312/bin/python -m …`. NEVER
  bare `uv run --project` (spawns a forbidden repo-local `.venv`).
- Compute etiquette: ALL compute (incl. CPU-only micro-tests and eval
  chains) runs as slurm jobs; the login node serves login/launch only.
- Compute nodes are OFFLINE: everything (env, data, code) staged on the login
  node before `sbatch`.
- **This workstation (Mac)**: analysis, editing, verification only — no
  training, no GPU, no artifact pipeline runs.

## Commands (regen chain order)

```bash
# Every command runs inside a slurm job (CPU-only jobs for pure-CPU steps,
# GPU jobs for training/scoring). Environment per job:
#   source /media02/hlhdang/tools/uvenv_ws_env.sh
#   PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
$PY -m pamr.scripts.check_registry --config configs/wikiart_mkg_v2/transe_clip.yaml --expect-raw 22 --expect-total 40
$PY -m pamr.scripts.prepare        --config configs/eval/wikiart_mkg_v2_neg_sampling.yaml
$PY -m pamr.models.pr.preprocess   --config configs/pr/wikiart_mkg_v2_paths.yaml   # path extraction
$PY -m pamr.scripts.train_embedding --config configs/wikiart_mkg_v2/transe_clip.yaml (--save_path ...)
$PY -m pamr.scripts.train_pamr     --config configs/pr/pamr.yaml  [ablation: pamr_{finetune,hop3,depth12,cls,lf_temp,seed1,par}.yaml]
$PY -m pamr.scripts.verify_paths   ...   # checks A/B/C on regenerated artifacts
$PY -m pamr.models.vlkge.score / pamr.scripts.evaluate --config configs/eval/wikiart_mkg_v2_score*.yaml
```
(Exact flag sets live in each config; smoke variants mirror prod under
`configs/{eval,pr}/smoke_*.yaml`. Turnkey chains live in `slurm/*.sbatch` —
prefer submitting those over hand-rolled command sequences.)

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

- **torch 2.5.1+cu121 is the pinned family** (workspace lock). The era of
  cu130 wheels + driver-535 nodes broke `nn.MultiheadAttention` CUBLAS —
  resolved by the workspace's cu121 triad; the mean-pool fallback in
  `pamr.py` stays dormant as contingency. Driver→CUDA gate map
  (535→cu121 only, 550→cu124, 560→cu126, 570→cu128, 580→cu130): a family
  change (new driver/wheels) requires an on-node smi+matmul proof job
  first — see `memory/bugs.md` §2/§4.
- **Sample caches v4**: `*.v4.samples.pkl` carries a manifest (registry hash, max_hops, whitelist hash) — mismatched caches rebuild, never silently reused.
- **ID validation**: `PAMRDataset` validates query IDs against entity/relation names; `node_type` tokens are 1/2/3 plus padding offset 0 (embedding size 4).
- **Legacy checkpoints are history only** (`checkpoints/vlkge/wikiart_mkg_v2/*.pt`): trained before the registry/leakage corrections — contract-invalid.
- **Known-swap CSVs**: `result.csv` H1/H3 TransE↔PAMR per-relation columns were swapped — do not trust old per-relation readings; the B-queue regenerates.

## Lint / Typecheck

There are no lint or typecheck commands configured for this project.
Local verification policy: `ast`/`py_compile` gates + targeted CPU micro-tests only.
