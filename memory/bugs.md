# memory/bugs.md — PAMR Defect Register

Why PAMR does not outperform VL-KGE despite consuming frozen VL-KGE embeddings as input. Triage view: fully resolved issues, partially resolved issues, and gaps that have not been touched.

Context files: `pamr/models/pr/pamr.py`, `pamr/scripts/train_pamr.py`, `pamr/models/pr/preprocess.py`, `pamr/models/pr/preprocess.cpp`, `pamr/models/embedding/base.py`, `pamr/dataloader.py`, `pamr/scripts/{prepare,train_embedding,evaluate}.py`, `pamr/models/vlkge/score.py`, `pamr/configs/wikiart_mkg_v2/*`, `configs/pr/pamr.yaml`, artifact evidence in `checkpoints/preproc/wikiart_mkg_v2/full/pamr/`.

---

## 1. Resolved — in previous turns

Code and configuration work is complete; only artifact regeneration/retraining (Section 3) remains.

### Relation vocabulary and IDs
- **Canonical 40-ID registry** — implemented in `pmr/relation_registry.py`. Raw relations get stable lexical IDs; every other asymmetric relation gets an internal inverse integer ID without inventing public names; categories (`undirected`, `directed_existing_inverse`, `directed_internal_inverse`) and an involution invariant are asserted at construction; a registry hash + manifest is persisted and consumed by `prepare.py`, `preprocess.py`, and checkpoints. *(Resolved in a previous turn.)*
- **P0 — Entity/relation ID-space mismatch between BFS graph and PAMR.** Was: `preprocess.py:179-183` built a sorted ID space while `KnowledgeGraphDataLoader` used first-occurrence order; `test_paths.csv` qid0 contained node `1062` typed `2` (artist), impossible in DataLoader space; range-set validation passed silently, so every interior node and edge token was the embedding of the wrong entity/relation, degenerating PAMR into a noisy pointwise scorer (observed MRR 0.379). Now: `preprocess.py` rebuilds the exact DataLoader configuration used by `prepare.py`, validates name↔ID round-trips of every query row, and writes `entity_mapping.csv`, `relation_mapping.csv`, `relation_manifest.json` beside split-scoped graphs. *(Resolved in a previous turn; artifact regeneration pending — Section 3.)*
- **P5 — `add_inverse_relations` config corruption.** Was: `complex_clip.yaml:33` held a YAML string-pair instead of a dict, silently yielding 22 relations patched ad hoc by zero-row extension. Now: all WikiArt configs use `inverse_relation_pairs` (`isInfluencedBy`↔`isInfluencedOn`, `isPupilOf`↔`isTeacherOf`) and `symmetric_relations`; `add_inverse_relations` raises if used; zero-row surgery in `train_pamr.py`/`vlkge/score.py` replaced by strict relation-count + registry-hash rejection; `exclude_relations_eval` no longer lists deprecated names. *(Resolved in a previous turn.)*

### Path pipeline
- **P2 — Undirected BFS with direction-agnostic relation tokens.** Was: reverse traversal fed the base relation ID with no direction info; parallel edges picked an arbitrary `unordered_map` element for `edge_type`; `node_type[u]` was overwritten per triple line. Now: edges carry a canonical oriented relation ID from the registry inverse map, reverse traversal has correct VL-KGE inverse semantics, adjacency is sorted with deterministic tie-breaks, and node types are pre-derived in a full-graph pass before serialization. The direction bit remains only as physical-traversal metadata. *(Resolved in a previous turn.)*
- **P7 — Silent skips from `-1` edge types.** Was: backward-BFS endpoint lookup could fail ⇒ `edge_type=-1` ⇒ rows folded silently into `total_skipped`. Now: edge types are collected during BFS reconstruction via ParentLink records (no ambiguous endpoint lookup) and the merge errors out when worker output lacks the explicit field layout. *(Resolved in a previous turn.)*

### PAMR data and model contract
- **P1 — Positive-path destruction in `PAMRPairedDataset`.** Was: `pos_samples[qid] = s` kept only the last positive, queries without negatives lost their positives, and pairing was frozen per epoch. Now: positives are kept in lists, deterministically cycled across negatives, and surplus or negative-free positives are retained as standalone samples with gradient-safe loss. *(Resolved in a previous turn.)*
- **P2b — Multi-path train/eval mismatch (probe-level).** Was: positive multiplicity collapsed in training while all positives persisted at eval; the first positive row counted against the rank. Now: paths CSV carries explicit `row_index`, `candidate_tail_id`, `query_relation_id`, `is_positive`; labels are never re-inferred; proposed per candidate tail (`max`) via `_aggregate_candidates`/`compute_rank_metrics(all_cand_ids=...)`. Full candidate-set batching per query is still batch-limited (Section 2). *(Resolved in a previous turn; sampler gap open.)*
- **P3 — `path_nodes != 0` used as the padding test though entity 0 is a real entity.** Was: `_make_padding_mask`/`_find_r2_positions`/`_compute_reconstruction_l2` counted non-zero IDs as validity, truncating paths touching entity 0. Now: all valid-length logic derives from the explicit token mask. *(Resolved in a previous turn.)*
- **P4 — Frozen embeddings attenuated by the fusion bug.** Was: average fusion divided by the count of enabled modalities, with inductive structural masking counting as present, halving textual-only embeddings (`(0 + textual)/2`) and collapsing feature-less entities to zero vectors. Now: the denominator counts present modality slots only (`seen_in_train` respected; weighted renormalization matched). *(Resolved in a previous turn.)*
- **PAMR score contract** — `pamr_score = vlkge_score(head, qrel, candidate_tail) + path_residual`: readout-MLP final layer, `no_path_residual`, and `recon_gate` all zero-initialized; frozen VL-KGE child stays permanently in eval mode via a `PAMR.train()` override; entity tokens carry the full real+imag ComplEx/RotatE representations; `probe_head_ids` explicit so no-path probes never fabricate embeddings from entity ID 0. *(Resolved in a previous turn.)*
- **Candidate-tail ranking/objectives** — three objectives are selectable and operational (`grouped_pairwise`, `listwise_softmax`, `binary_candidate`); `min_eval_coverage` gates best-checkpoint selection; scores CSV carries candidate identity. *(Resolved in a previous turn.)*
- **Minor: spurious `node_type_emb` padding bias** — padding row and direction-embedding index 0 zero-initialized. *(Resolved in a previous turn.)*
- **Minor: O(n²) CPU ranking loss** — replaced by grouped tensor ops in candidate-level modes (perf note). *(Resolved in a previous turn.)*

### VL-KGE training and data
- **Bernoulli corruption ignored** — Was: `relation_probs`/`use_bernoulli` passed in but corruption was a fixed 50/50 batch mask. Now: per-triple relation-specific corruption probability; legacy behavior preserved when `use_bernoulli=False`. *(Resolved in a previous turn.)*
- **Positive tail reused as a negative** — Was: `negative_sampling_per_relation_fast` substituted the positive tail when a pool was missing or exhausted, silently corrupting the loss. Now: fail-fast contract (`RuntimeError`/`ValueError`). *(Resolved in a previous turn.)*
- **Filter-map key collision** — Was: `(h,r)->tails` and `(t,r)->heads` shared one dict. Now: `DirectionalFilterMap` with `get_tails`/`get_heads`, wired through training, `evaluate_kge`, and `predict_heads`. *(Resolved in a previous turn.)*
- **CLI inverse syntax mismatch** — Was: `--add_inverse_relations "a:b"` strings passed to a dict-only loader. Now: `--inverse_relation_pairs` converts `"a:b"` entries to a mapping; `--symmetric_relations` added; YAML mappings take precedence through `extra_config`. *(Resolved in a previous turn.)*
- **Embedding scopes** — `train` / `train_val` / `train_val_test` (diagnostic oracle) implemented in `train_embedding.py` with scope-local loaders, quick-eval frames, scope-local negative pools, and checkpoint metadata (`embedding_scope`, registry manifest, entity/relation mapping hashes). *(Resolved in a previous turn.)*
- **Split-aware truth filtering** — `compute_filter_maps_by_scope` feeds `prepare.py`: train filters train truths, val filters train+val, test filters train+val+test; inverse facts materialize inside their source split. *(Resolved in a previous turn.)*
- **Exact negative sampling** — set-subtraction requires K eligible candidates; rejection sampling completes deterministically from the eligible set; incomplete queries are skipped and counted, never silently underfilled. Relation metadata, split/candidate policies, and registry hash are persisted in `metadata.json`. *(Resolved in a previous turn.)*
- **Path-graph split leakage** — Was: BFS graph loaded the raw all-split CSV. Now: per-split scoped graphs (`graph_{split}.csv` + `graph_*_meta.json`), defaults train for val and train+val for test; transductive diagnostics require an explicit `path_graph` section. *(Resolved in a previous turn; runtime verification deferred — Section 3.)*

### Inspection wave 2026-09-19 — deep no-run audit findings (F-numbered)
- **F1 — `NameError` in `_aggregate_candidates`.** Was: the signature was `(scores, query_ids, candidate_tail_ids)` while the body read `labels[gather]`, so every candidate-level loss mode (`grouped_pairwise` prod default, `binary_candidate`, `listwise_softmax`, `reconstruction`) crashed on the first `compute_loss` with `NameError: name 'labels' is not defined`. Now: probe labels are threaded through both `compute_loss` call sites; verified by a torch micro-test on grouped max-aggregation. *(Resolved in this wave.)*
- **F2 — `pamr_collate` built but never returned `probe_head_ids`.** Was: the return tuple carried 11 tensors against a 12-name unpack in `train_pamr.py` (train/val/test sites) → `ValueError: not enough values to unpack` at the first batch in any loss mode. Now: 12-field contract with `probe_head_ids` at index 8, `hops` at 9, `has_path` at 10, `token_mask` last, verified against the exact unpack. *(Resolved in this wave.)*
- **F18 — mixed path/no-path padding crash** (found by the collate micro-test). Was: `pad_edges[i, :N-1]` with `N=0` sliced `[:-1]` — all-but-last column — instead of empty, so any batch mixing path and no-path probes with `max_N >= 2` raised a shape RuntimeError. Now: `edge_len = max(N-1, 0)` clamp. *(Resolved in this wave.)*
- **F3 — merge-step positional unpack scramble** (`preprocess.py`). Was: `row_meta` tuples were `(query_id, head_id, tail_id, relation_id, is_positive)` but the unpack bound `candidate_tail_id ← head_id`, `query_relation_id ← tail_id`, `is_positive ← relation_id`, `query_head_id ← is_positive` — silently wrong columns in every merged paths row (path probes crashed downstream, no-path probes carried head=entity 0/1 and label=relation id). Now: keyed probe records (dicts) — no positional unpack anywhere in the merge. *(Resolved in this wave.)*
- **F4 — stale missing-path worker row layout** (`preprocess.cpp:321`). Was: unreachable probes emitted `qid;-1;;;;;row_index` (old contract), so the merge parsed `row_index=-1` → RuntimeError, aborting path extraction at the first unreachable probe. Now: emits `qid;row_index;-1;;;;`. *(Resolved in this wave.)*
- **F5 — self-probe rows crashed sample building.** Was: the `head == tail` C++ branch emitted `hops=0` with two nodes (`"h,h"`), violating the `len(nodes)==hops+1` sample contract whenever a sampled candidate equaled the query head. Now: self-probes emit the standard no-path record (per interview decision: no self-edge exists, so no-path is semantically correct and keeps the direct VLKGE score + learned residual); `self` is a subset of unreachable and no longer double-counted in the run summary. *(Resolved in this wave.)*
- **F10a — worker failures swallowed at merge.** Was: failed C++ threads only printed a WARNING; their probes silently vanished from the paths CSV (shrinking candidate sets, possibly dropping positives while negatives survived), and nothing verified output completeness. Now: failed threads abort the split before merge, and the pure `audit_probe_completeness` helper enforces every prepared probe exactly once (micro-tested for complete/missing/duplicate cases). *(Resolved in this wave.)*
- **Evaluator centralization (closes Section 2 gap 2).** Was: `evaluate.py` computed `rank_ns = sum(scores >= pos_score) + 1`, double-counting the selected positive per tie, with a second, divergent evaluator in `train_pamr.py`. Now: `pmr/ranking.py` is the single shared ranker (strict/non-strict corrected semantics, best-positive selection, candidate-tail aggregation, per-relation Accuracy); `evaluate.py` is rewired onto it and `train_pamr.py` passes `rel_for_qid` explicitly (module-global removed). Micro-tested: tie cases, multi-positive queries, candidate aggregation, exclusion, no-positive skips. *(Resolved in this wave.)*

### Inspection wave — runtime bundle (Phase 2, resolved in this wave)
- **Eval batch size no longer tied to the train batch size** — Was: `train_pamr.py` used one `batch_size` (prod: 8) for all three loaders, so full val/test enumeration ran in 8-probe batches. Now: `eval_batch_size` (default 1024; `pamr.yaml` explicit) drives val/test loaders. *(Resolved in this wave.)*
- **Dead all-splits filter map removed** (`prepare.py`) — Was: `compute_filter_map()` built two defaultdicts over the ~16.6M materialized rows and was never consumed (scope maps are built per split). Now: removed. *(Resolved in this wave.)*
- **Vectorized inverse materialization** (`dataloader.py`) — Was: an 8.3M-row `itertuples` Python loop, re-executed in every process that constructs a `KnowledgeGraphDataLoader` (5+ times per experiment chain). Now: vectorized rename/concat + dedup; micro-tested for public inverse names, dedup precedence of raw rows, and internal-inverse `None` names. *(Resolved in this wave.)*
- **v4 sample-cache manifest** — Was: `v3` caches validated only via file mtimes and used a 120-char truncated whitelist tag (collision-prone); no registry/policy identity at load time. Now: cache suffix `v4`, whitelist tag is a content hash, and the cache payload carries a manifest (`format`, `registry_hash`, `num_relations`, `max_hops`, `whitelist_hash`) validated at load — mismatched caches rebuild instead of silently reusing. *(Resolved in this wave; closes the Section 2 cache-manifest gap.)*
- **Per-relation path/no-path coverage columns** — Was: counts existed only at dataset level. Now: `per_relation_path_coverage` (probes, path/no-path/positive probes, queries, candidate tails, mean paths per candidate, per relation, per split) is written into `pamr_training_metrics.json` under `path_coverage`; helper micro-tested. *(Resolved in this wave; closes the Section 2 per-relation-reporting gap.)*
- **Registry acceptance script** — New `pmr/scripts/check_registry.py` verifies raw/total counts, inverse involution, recomputed-vs-persisted hash, and public-vocabulary integrity against a live config; runs on the compute machine with the real CSV. *(Resolved in this wave; closes the Section 2 acceptance-script gap.)*

### Inspection wave — train/eval evidence + BFS (Phase 4, resolved in this wave)
- **F7 — query-edge masking implemented (closes the B1 operational half).** Was: train positives sat in their own path graph, so BFS returned the trivial direct edge (hops=1) while val/test positives never have that edge — a residual-head shortcut that transfers poorly. Now (per interview): the probe's own edge — the query relation and its registry inverse between head and tail, in both orientations — is masked in **every** split via `path_graph.mask_query_edges: true` (default), making val/test behaviour unchanged (scopes already exclude own facts) while train evidence matches eval; the target-traversal rule is one uniform flag (`allow_target_relation_traversal: true` = non-own target edges traversable everywhere; `false` denies all target edges in every split). The Python wrapper passes the policy and relation IDs to the C++ worker, and both meta manifests record it. *(Resolved in this wave.)*
- **F14 — forward-BFS reuse + drain fix.** Was: each of a query's ~11 probes recomputed the identical forward BFS and the loop stopped when either frontier drained (reachable tails could be mislabeled `hops=-1`). Now: forward BFS from the head runs once per query and is reused by all its probes; backward BFS runs per probe with meet detection against the forward tree and an explicit depth bound; probes whose own edge is present (in practice the positive) fall back to a fully masked per-probe bidirectional BFS. Verified end-to-end on a synthetic graph (mask on/off, deny, self, reverse orientation). *(Resolved in this wave.)*
- **F13 — thread RAM documented** — each worker builds its own full-split adjacency; the paths config now states the ≈ num_threads × graph-size budget. *(Documented in this wave; per-machine tuning remains with the user.)*

### Inspection wave — model refinement (Phase 5, resolved in this wave)
- **F8 — per-relation no-path residual.** Was: one global learnable scalar shifted every no-path candidate equally across all relations. Now: zero-initialized per-relation `nn.Embedding(num_relations, 1)` (fallback to the shared scalar when no relation count is available); the frozen VLKGE baseline contract and the Decision Record are intact (residual is exactly 0 before training). Micro-tested on a stub model: zero-init parity, per-relation offsets, gradient flow. *(Resolved in this wave.)*
- **F9 — objective scale (decision recorded, no code change).** The margin-1 hinge on absolute fused scores remains for the first retrain (single-variable A/B against frozen VLKGE); scale-aware margins and contrastive/temperature variants stay with the planned B3 ablation. *(Decision recorded in this wave.)*
- **F17 — accepted hot loops** — per-column token assembly, reconstruction L2 loop, and per-triple `pool_set` rebuilds remain; they are bounded (max_N=6, smoke-only, rejection fallback) and were deliberately left untouched. *(Accepted in this wave.)*
- **Dead code removed** — `load_vlkge_model` (superseded by `train_pamr.py`'s inline loader) and the unused `_make_padding_mask` are gone. *(Resolved in this wave.)*

### Worker v2 hardening — full re-read audit (resolved in this wave)
A complete re-read of `preprocess.cpp` (861 lines) and `preprocess.py` after the v2 rewrite found no memory leaks (all RAII, clean destruction on every path) but four defects/robustness gaps, now fixed and verified:
- **C1 — empty-graph/OOB-head crash in the shared forward BFS.** Was: `bfs_levels` for the group head ran before the per-probe OOB guard, so an edge-less graph (`g.n == 0`) or a head beyond the node range indexed empty arena vectors (UB crash; v1 tolerated this as hash-map misses). Now: the shared search is skipped when `head_probe.head_id >= g.n`; every such probe is OOB-guarded and emitted as no-path. Regression fixture: edge-less graph → rc=0, all probes `-1`.
- **C2 — negative/oversized fields crashed the CSR build.** Was: `degree[f[0]]++` with a negative node id wrote out of bounds, and `pack_rev` with a relation id ≥ 2^30 corrupted the packed direction bit. Now: `graph_row_valid` (non-negative node ids, packed relation ids < 2^30, full-int types after widening `types` to int32) is applied identically in all three loading passes; skipped rows are counted and summarized as one `[Warn]` line. Regression fixture: negative-id and oversized-relation rows are warn-skipped and the valid search is intact.
- **C3 — LineReader fabricated a premature EOF on a >chunk line.** Was: a "line" filling the whole 16 MB chunk with no newline made `read()` return 0 and set `eof_` while the file continued (silent data loss for pathological input). Now: the buffer doubles when full instead of reading zero bytes.
- **C4 — negative `max_hops` crashed the fanout-table allocation** (`vector` size_t conversion). Now clamped to 0 (every probe becomes no-path, rc=0).
- Python cosmetics: module docstring updated to the v2 process model, unused `numpy` import removed, the merge's legacy `h == 0` self-count branch annotated (v2 emits self-probes as `hops=-1`).
- Re-verified after hardening: 7/7 v1-vs-v2 equivalence cases MATCH, calibration rc=0, out-of-range warns appear, negative-hops fixture rc=0, and v2 remains byte-deterministic across runs.

### Phase 0 (2026-10-03) — provenance resolution + leakage-verification runner
Executed toward the locked experiment plan; local-only work (no training, no C++).
- **Experiment plan locked (interview round 1 + backbone round):** compute = FIT cluster (Slurm, A100 40GB, ≤2 concurrent jobs × 2 GPUs, env being prepared by another agent — GPU phases wait); regen matrix = FULL (TransE, ComplEx, DistMult, RotatE + structural-only backbone + PAMR + PAR); backbone scope = train everywhere (no matched-scope pairing); protocol = 10 negatives × 2 seeds (42 + 1, n=2 minimal std; baselines single-seed); ablations = core 5 (frozen-vs-tuned, K=3, depth 12, cls readout, B3 contrastive/temperature); camera-ready = correct the numbers if accepted; backbone = **TransE-faithful**.
- **Provenance resolved from synced artifacts (resolves the shuffled-value forensics):** pamr_training_metrics.json records `vlkge_checkpoint = /workspace/checkpoints/vlkge/wikiart_mkg_v2/complex_wikiart_mkg_v2.pt` and all 22 PAMR run logs echo `entity=768, rel=1536 → d_model=1536` (ComplEx-family; TransE relations would be 768) — scenario A's "reported runs used frozen TransE" recollection was WRONG about the artifacts. Per the user's decision, the regeneration is TransE-faithful: results are a NEW configuration relative to the published numbers; the ComplEx evidence stays here as the provenance of the published numbers.
- **Old-test-eval degeneracy confirmed:** the June campaign's post-training test eval prints MRR=1.0000 on 235 queries — the old all-splits test path graph contained test edges (trivial direct edge). The locked scope+masking design removes exactly this.
- **Split facts pinned:** count.csv (74+76+26+26=202) matches the **val** split — the paper's Table 1 evaluates val (202 whitelist queries, 2,215 valid probes); the test set has 238–239 queries (2,549 probes). The earlier "239-vs-202 discrepancy" is a split-shape difference, not an error.
- **`pamr/scripts/verify_paths.py` (new)** — the three B1 structural assertions from artifacts only: (A) scope hygiene (no excluded fact's own edge — relation or registry inverse, both orientations — inside the split-scoped graph; exclusion map streamed from the raw triples CSV with a frozenset pair index); (B) own-edge non-usage in emitted paths of ANY split + `len(nodes)==hops+1` / `len(edge_types)==hops` sample contract + per-split probe completeness (each prepared probe exactly once); (C) candidate identity — PAMR↔VL-KGE scored CSVs cover identical rows (same query sets, per-query row counts, per-query positive multiset) and PAMR candidate tails match prepared rows (gold counts cross-checked against the prepared queries CSV when supplied). rc=0 clean / 1 violations. Micro-tested: 5 synthetic worlds (clean, own-edge-detected, scope-leak-detected, completeness-detected, identity-aligned/detected).
- **Rename sweep (`pmr` → `pamr` leftovers):** 47 fatal `pmr.` import refs repaired across 14 modules (pamr/__init__.py, dataloader.py, helpers.py, models/*, scripts/* — `import pamr` round-trips); pyproject entry points + package-data rewired; the wikiart regen chain configs moved to the workspace-relative convention (`../data/…`, `../checkpoints/…`, resolving via resolve_config_paths candidate 2 = repo root); the second freeze loop in `train_pamr.py:298` gated on the flag; wn9_img/wikiart_mkg_v1 dataset configs retain their rename leftovers — out of scope for this campaign, noted here.
- **Second freeze loop repair** — `train_pamr.py` had its own `p.requires_grad = False` loop outside the model which would have silently defeated `fine_tune_backbone`; both sites now gate on the flag (model `__init__` + script), and trainable-vs-frozen params are printed and logged (`fine_tune_backbone`, `lf_temperature` keys in `pamr_training_metrics.json`).
- Torch micro-tests via the graphml env: default freeze (no grad to backbone), fine-tune split (feature table frozen, structural/relation trainable, backward reaches vlkge), and listwise temperature (temp=1.0 equals raw softmax; temp=0.1 sharpens and differs). Full package py_compile OK.

### Phase 0 audit + defect round (2026-10-03) — READ FIRST after compaction

The Phase 0 output was audited; two critical defects were found and now fixed. Verification policy for this round: ast/compile + logical determination only (no local runs — the workstation was starved).

**Fixed — `pamr/scripts/train_pamr.py` missing `def main():` (CRITICAL).** The entire training body (1-indexed 117–989) was unreachable dead code inside `resolve_config()` after its `return`. Fix: inserted `def main():` right after the return; the body was already at correct 4-space function indentation (no dedent needed — the first transform wrongly dedented to 0-indent, caught by ast.parse before the corrupted write was accepted; the file was restored via `git checkout` and the insert applied). Verified: resolve_config 106–117, main 118–990, `if __name__ == "__main__"` calls `main()` (992–993), `PYTORCH_CUDA_ALLOC_CONF` reachable as main's first statements, package-wide sweep finds ZERO unreachable statements after returns, py_compile OK.

**Fixed — `pamr/scripts/verify_paths.py` check_meta registry-hash false mismatch (CRITICAL).** The per-key cross-check loop previously compared graph meta's top-level `relation_registry_hash` against REAL paths meta, where the hash is NESTED (`relation_registry.manifest.registry_hash` — `to_dict()` adds it inside) — every real artifact split would falsely FAIL. Now: `graph_hash` = graph meta top-level; `paths_hash` = `(paths_meta["relation_registry"] or {}).get("registry_hash")`; the loop covers only path_graph_scope / mask_query_edges / allow_target_relation_traversal. Verified by AST: loop excludes the key; nested comparison present.

**Hardened — check A gains registry-inverse ids.** `build_exclusion_index` loads `inverse_relation_id` (str→int) from the artifact's own paths-meta manifest and adds each excluded fact's registry-inverse id to the pair's forbidden set (try/except per entry; degrades to raw-id-only if the manifest is missing). This matches the docstring's promise; no false-negative today (the graph file never materializes inverses — the C++ does at load), but the audit is now bulletproof against future pipeline changes.

**Policy — `requirements.txt` relaxed to version floors** (per user: no strict matching on all libraries; main ones keep floors): `torch>=2.6`, `pandas>=2.2`, `numpy>=1.26`, `transformers>=4.48`. The exact campaign environment is selected on FIT after its preparation completes (see the FIT handoff below).

**Repointed — all smoke chains to the TransE train-scope base (user instruction).** `configs/pr/smoke_pamr.yaml` + `smoke_pamr_recon.yaml` now carry `vlkge_config: ../wikiart_mkg_v2/transe_clip.yaml` + `vlkge_checkpoint: …/transe_wikiart_mkg_v2_train.pt` (with stale d_model/heads comments fixed to 768 / 96 per head). `configs/eval/smoke_score.yaml` (self-contained) additionally swapped the inline arch `model: ComplEx` → `model: TransE` + TransE-specific keys (`p_norm: 1`, `normalize_relations: true`, `margin: 12.0`) — otherwise a ComplEx arch would load a TransE state dict and crash on key mismatch. Ablation configs already inherited TransE (derived after the repoint). YAML-validated.

**Hop flattening confirmed (user): prod and h3 flatten the SAME way.** Both sbatch chains flatten the preprocess run dir (`<base>/<ts>/` → `<base>/pamr/`) identically; the committed configs' paths_dir conventions already agree.

**Test fixture mirrored to real shapes** (temp dir): paths-meta writer now carries the nested manifest (raw names + inverse ids + registry_hash) instead of the vacuous top-level duplicate, and graph meta carries the top-level hash + target_relation_ids.

**Untouched by scope (noted):** legacy `configs/eval/vlkge_score.yaml` (absolute `/workspace` path + complex checkpoint), `pathlen_*_eval.yaml` chain, wn9_img/wikiart_mkg_v1 configs (pmr/ rename leftovers), user's `" 2.yaml"` duplicates.

**VC state (stale as written; corrected 2026-10-04):** the fix round was committed at `d4c2d23` with the registers, and the §5 reference mapping landed at `ba7c335` — for the authoritative snapshot see §19 and `git log`.

---

## 2. Partially resolved — code present, known gaps

- **P6 — Metric and protocol asymmetry vs VLKGE.** Was: strict-vs-non-strict tie semantics diverged, evaluation conditioned on BFS reachability only (202 evaluated val queries, Hits@10≈0.97 blocks), and `mode: train_eval` accidentally filtered PAMR training instead of eval-only as commented. Since then: the relation filter mode semantics are explicit and enforced, PAMR reports candidate coverage and ranks candidate tails, tie handling is centralized in `pamr/ranking.py` (Section 1), and the small-block conditioning is documented via coverage. Remaining gap:
  1. score VL-KGE on exactly the same split-local target candidate set as PAMR — CLOSED by decision (2026-10-03, interview: 10-neg × 2 seeds; full-pool ranking declined) for the identity half: `pamr/scripts/verify_paths.py` check C now proves both score CSVs cover identical rows + candidate tails (runs on regenerated artifacts in Phase B/C). The full-filtered-pool scorer stays unimplemented by decision.

Closed by this wave (moved to Section 1): the cache manifest gap (v4 caches now embed and validate the registry/split-policy hashes), batch-limited candidate grouping (the bucketed query-group sampler guarantees complete candidate groups per batch), per-relation coverage reporting (`path_coverage` in `pamr_training_metrics.json`), and the registry acceptance script (`pamr/scripts/check_registry.py`).

---

## 3. Not touched

- **Artifact regeneration (Phase 6)** — nothing has been re-run: old query CSVs, all-split path CSVs, `*.v2.samples.pkl` caches, preprocessed `graph.csv`, and both old VL-KGE (22 relations) and PAMR checkpoints remain on disk and are invalid under the new contract. Mandatory on compute:
  1. run `pmr/scripts/check_registry.py --config configs/wikiart_mkg_v2/complex_clip.yaml --expect-raw 22 --expect-total 40`;
  2. compile `preprocess.cpp` (worker v2: one process with `num_threads` internal threads over one shared CSR adjacency, fast in-place CSV parsing, flat epoch-stamped BFS arenas, exact-duplicate edge dedupe; part-file output contract unchanged, partition is contiguous group ranges);
  3. `prepare.py` → new query CSVs (target relations only, exact-K negatives). Memory note: scope truth maps are built with `relation_ids` restricted to the target relations under `train_eval` (full-graph maps over 16.6M materialized triples OOM-killed a 12.7 GB Colab runtime at ~7.4 min, rc=-9 with a buffered-empty log), and the DataLoader now releases its raw frame after construction;
  4. `preprocess.py` → split-scoped `graph_{train,val,test}.csv`, per-probe masked train paths, enriched `*_paths.csv` + `*_paths_meta.json`;
  5. retrain VL-KGE scopes (`train`, `train_val`, optional `train_val_test` oracle) with the 40-ID registry;
  6. retrain PAMR per selected scope with `*_v4.samples.pkl` caches (group batching on, `min_eval_coverage` 0.95, `patience` 5, per-relation no-path residual).
- **Runtime structural checks** — runner EXISTS (`pamr/scripts/verify_paths.py`, Phase 0): "no validation/test edge inside main val/test path graphs", "own edges masked in every split's paths", and identical candidate tails between VL-KGE and PAMR score CSVs are all checked from artifacts. Execution is deferred only because it depends on regenerated artifacts (Phase B/C on FIT).
- **bfs v3 sweep (27-point grid)** — the benchmark notebook now carries the v3 batch: `SWEEP` = hops {2,3,5} × fanout {10,50,100} × decay {0.2,0.5,1.0} (27 configs), rows tagged `worker_version="v3"` (run-batch identity; the worker binary is unchanged from v2 — v1 was never measured, v2 = the 12-point run). The export cell merges & appends: existing `assets/bfs.csv` (12 v2 rows, all complete) plus the 27 v3 rows, deduplicated per (worker_version, hops, fanout, decay) keeping the newest — one cumulative CSV. Cells 7-8 gain validity guards that reuse the prepared val queries and the val fixture set when a previous run in the same Colab runtime is still valid (metadata + manifest checks), falling through to a full re-prep otherwise. Measured v2 baselines the v3 grid resolves: `load_wall_s` is a constant 3.5 s (single shared CSR, peak RSS ≈ 205-227 MB total) and reachability is dominated by `max_fanout` (fanout 10 finds ~4% of the 2,222 val candidates at hops=3; fanout 100 finds ~99.4%), so the new points resolve the decay axis and the hops=2 short-evidence regime.
- **Paper-metric refresh** — blocked on the regenerated, leakage-safe comparison.
- **B3 contrastive/temperature objective** — deferred by decision (F9); the scale-aware-margin variant also lives there.

---

## Decision Record

- No relations are assumed symmetric by default.
- Existing inverse pairs are `isInfluencedBy` ↔ `isInfluencedOn` and
  `isPupilOf` ↔ `isTeacherOf`.
- Every other raw relation receives an internal inverse integer ID without
  a new public relation name.
- The legacy public `hasCreatedArtwork` relation is deprecated.
- Raw relation IDs use stable lexical ordering. Internal inverse IDs are
  appended in the corresponding raw-relation order.
- The resulting registry has 22 raw IDs plus 18 internal inverse IDs, for 40
  relation IDs.
- Inverse facts are materialized within the same split as their source fact.
- PAMR query filtering uses `train_eval` for the four artist-to-artist target
  relations, while intermediate path edges may use every registered relation.
- Main path scopes are train for validation and train plus validation for test.
- Current-split VLKGE embeddings are diagnostic only. Current-split facts are
  not used for PAMR paths, PAMR training, or path negative filtering.
- Candidate tails are split-local for the four target relations.
- VLKGE and PAMR rank identical candidate tails.
- PAMR aggregates multiple paths per candidate tail with `max`.
- PAMR starts from the frozen VLKGE score and learns a zero-initialized path
  residual. No-path candidates retain the direct VLKGE score.
- PAMR loss selection is configurable among grouped pairwise, listwise
  softmax, and binary candidate objectives.

Inspection-wave decisions (2026-09-19 interview):

- Self-probes (candidate tail equal to the query head) are explicit
  no-path probes: no self-edge exists, so the candidate keeps the direct
  frozen VLKGE score plus the learned no-path residual.
- Path merges abort on failed workers and on incomplete or duplicated
  probe outputs; probe completeness is audited per split.
- Ranking semantics live in `pmr/ranking.py` and are shared by
  `evaluate.py` and `train_pamr.py` (single strict/non-strict/multi-positive
  definition).
- Local verification is restricted to py_compile, a C++ syntax check,
  numpy/pure-function micro-tests, and torch micro-tests via the
  `graphml` conda env; no training, data generation, or GPU work happens
  locally. Artifact regeneration, compilation for production, and
  retraining remain on the compute machine.
- Code state is committed to git per phase (checkpoint `5a3af76`,
  Phase 1 `9f19ca2`).
- Per-probe masking (F7): own query relations (query relation + registry
  inverse between head and tail) are masked in every split; the
  target-traversal rule is one uniform flag across splits (default true
  = non-own target edges traversable; `false` denies all target edges).
  Uniformity is what actually matches train and eval evidence rules; a
  per-split difference would itself be a distributional skew.
- Length-first composite: no-path candidates are excluded from lf ranking
  by default (`lf_no_path_policy: exclude`), with the lf population and
  skip counts reported; `infinite` ranks them after all path candidates.
- Candidate-level objectives run with the bucketed query-group sampler
  (`group_batches: true`) so candidate sets are complete per batch.
- Production gates: `min_eval_coverage: 0.95` and `patience: 5`.
- Production objective for the first retrain stays `grouped_pairwise`;
  contrastive/temperature and scale-aware margins are B3 ablations.
- No-path residual granularity is per query relation (zero-initialized);
  the Decision Record's zero-init baseline contract is unchanged.

---

## 4. FIT cluster handoff (verified via SSH 2026-10-03)

### Access + rules
- `ssh FIT` = 172.29.74.81, user `hlhdang`, key `id_rsa_fit` (~/.ssh/config). Banner notice + watch UI `http://172.29.74.82:8012/`.
- Authoritative reading order: `~/AGENTS.md` (shared-HPC rules, env system, job lifecycle) → `/media02/hlhdang/README.md` (filesystem map + conventions) → `~/admin_note.txt` (store everything under `/media02/hlhdang`).

### Cluster facts (verified 2026-10-03)
| Item | Value |
|---|---|
| Scheduler | Slurm, single partition `batch` |
| Nodes | gpu01 = 8×GPU, 980 GB RAM; gpu02–04 = 4×GPU, ~240 GB RAM; 256 threads each |
| Time | MaxTime 2-00:00:00; DefaultTime 6 h |
| Memory | DefMemPerCPU=2 GB → ALWAYS explicit `--mem` |
| Per-job CPUs | account limit **12** |
| GPU quota | QOS `gpu_limit` = 4 GPUs/user (RUNNING + PENDING both count) |
| Scratch | TmpDisk=0 → /dev/shm (~500 GB gpu01) or /tmp (~1.6 TB); jobs self-clean (`trap … EXIT`) |
| Storage | /home NFS 195 GB cluster-wide (~31 GB free, dotfiles only); /media02 5 TB 96–97% full (~200–240 GB free) |
| Compute nodes | OFFLINE — never pip/download inside a job; stage on login |
| Watch UI | `http://172.29.74.82:8012/` |

### MIGRATION 2026-10-04 (READ FIRST — supersedes the multi-venv facts below)
- The env system is UNIFIED: exactly ONE environment `/media02/hlhdang/uvenvs/main-py312` — CPython 3.12.14, torch 2.11.0+cu130 / torchvision 0.26.0 / torchaudio 2.11.0, LeWM frozen stack, ~6.6 GB [ACTIVE, import-verified].
- All earlier venv rows (t251cu121, t260cu126, t280) are **EVICTED** (deleted after a disk incident; free space ramped ~15 GB → ~67 GB). NEVER rebuild them; NEVER mint sibling venvs ("no new venvs — ever"; a second CUDA-family venv needs explicit human approval).
- Project binding: minimal pyproject deps → `uv lock` on the login node → `UV_PROJECT_ENVIRONMENT=/media02/hlhdang/uvenvs/main-py312 uv sync --locked`. Keep additions minimal and note them in `uvenvs/registry.json`.
- **CRITICAL for pamr**: the sync must NOT dislodge torch 2.11.0+cu130 (one-CUDA-family policy — a PyPI default wheel would swap the cu130 build out from under the collective env). Inspect the ejepwm project's pyproject/lock for the cu130 index pattern first; if all pamr deps already exist in main-py312, prefer NO torch dependency entry at all.
- Budget: env+cache ≤ 60 GB; stop if media02 free < 25 GB; any write expected > 5 GB needs a pre-flight `df -h /media02` + a registry note. Compute nodes OFFLINE (unverified internet) — stage on login.

### GPU fleet probe (2026-10-04, Phase A6) — cu130 torch GPU-DEAD fleet-wide
- Survey (nvidia-smi inside `--gres=gpu:1` jobs): gpu01 = 8× **A100-SXM4-40GB, driver 525.147.05** (CUDA 12.0); gpu04 = V100-DGXS-32GB, driver 535.216.03 (CUDA 12.2); gpu03/gpu02 not directly probed (gpu02 DRAINED per sinfo; gpu03 probes blocked by scheduler transiently). Node probe pattern: `-N1 --nodelist=<node> --gres=gpu:1` (missing `-N1` = "Node count specification invalid").
- **torch 2.11.0+cu130 fails on every current driver** (needs the CUDA-13 driver family, 580+): probe job `pamr-probe` (job 78095) printed `cuda_available False` → `RuntimeError: The NVIDIA driver on your system is too old (found version 12020)` on gpu04. Files: `projects/pamr/reports/probe_78095.out/.err`.
- The unification agent's own logs (`reports/finalize_py312.log`) show ONLY import verification — `IMPORTS-OK torch=2.11.0+cu130` — no GPU job ever ran. The collective env is GPU-sterile on this fleet.
- Phase A completed despite this: staging (src 81 MB diff-verified; data 1.7 GB cksum-verified), uv.lock minted with env-exact pins (torch==2.11.0+cu130 via cu130 index, requires-python >=3.12; 44 packages; `uv run --no-sync` binding test IMPORT-OK), `preprocess.o` compiled (g++ 11.4, c++23), registry noted. Rename leftover swept: configs' `cpp_source/binary_path: pmr/…` → `pamr/…` (would have crashed SLURM compile step). workstation pyproject rewritten (name vlkge→pamr, env-exact pins, transformers/lavis dropped from runtime deps — only `evaluate_zero_shot.py` imports them, out of campaign scope).
- Binding trap recorded: bare `uv run --project` spawns a FORBIDDEN repo-local `.venv`; always export `UV_PROJECT_ENVIRONMENT=/media02/hlhdang/uvenvs/main-py312` (and prefer `--no-sync` to never mutate the collective env). Registry row updated (used_by+=pamr, note records the trap).
- GPU-phase resolution pendente user decision: (a) ask admin for 580-series drivers on A100 nodes, or (b) human-approved second CUDA-family venv `t262cu126-py3XX`-style (cu12x compatible with driver 525/535 — matching the ORIGINAL campaign plan for torch 2.6) for pamr only, or (c) run Phase B CPU-only first (prepare/BFS/graph regen need no GPU) and defer GPU traces.

---

### Environment system (SUPERSEDED 2026-10-04 — historical background)
- Factory: `/media02/hlhdang/scripts/make_uv_env.sh <name> --python <spec> --project <dir>`; forbidden: conda/mamba create, python -m venv, virtualenv, pip --user, pipx, repo-internal `.venv`.
- Names version-keyed lowercase `t<framework-ver>[-cuXXX]-py<XY>`; venvs central in `uvenvs/`, immutable-in-place; conflicts mint siblings.
- Every project MUST have committed `pyproject.toml + uv.lock`; rebuild = `uv sync --locked`.
- Runtime: `source /media02/hlhdang/tools/uvenv.sh` then `uv run --project <dir> python ...`; catalog `uvenvs/registry.json` via `scripts/registry.py list|verify|drop`.
- uv pinned at `tools/uv/uv` (v0.12.19 + sha256 beside it); do not install another uv.

### Venv rows + sacred cow (SUPERSEDED — see MIGRATION above)
- ALL named rows (`t251cu121-py310`, `t260cu126-py312`, `t280-py310`) EVICTED 2026-10-04; historical only.
- Sacred cow: `/media02/hlhdang/AKG-sv` (active collaborator project, 161G repo braid) — DO NOT TOUCH.
- Campaign binding decision (REVISED 2026-10-04): NO new venv — rebind pamr onto `main-py312` via its own uv.lock. torch 2.11.0+cu130 > the 2.6 floor satisfies the CUDA13-driver requirement (the repo's mean-pool fallback stays dormant; uv-locked pandas/numpy need no special handling on py3.12).

### Proven sbatch patterns (projects/ejepwm/slurm/)
- `main.sbatch`: gpu:1, 12 cpus, mem=180G, `--signal=B:USR1@300 --requeue`, `trap 'rm -rf "$SCRATCH"' EXIT`, `-o/-e` under `reports/`, chain via env vars.
- `smoke.sbatch`: 8 cpus, gpu:1, 32G, 1 h, prints `nvidia-smi --query-gpu=name,driver_version,memory.total` + `ACCEPTANCE ...` lines.
- CPU builds: incremental mark files so requeues skip finished work.
- Repo conventions: projects keep README.md, MAY add AGENTS.md; relocation = rsync -a + `diff -r` byte-verify; pip/HF caches on media02, never $HOME.

### Updated program (Phase A onward, FIT-organized)
1. **A1** stage tree → `/media02/hlhdang/projects/pamr` (rsync -a + `diff -r` byte-verify); **A2** data staging (triples CSV 1.04 GB → `projects/pamr/data/`; feature pkls same; size-check first vs the communal-free figure); **A3** `uv lock` inside the project (login node, internet OK) → commit `uv.lock` + rsync it back to the workstation repo; **A4** rebind pamr onto `main-py312` (minimal deps in pyproject → `uv lock` on login → verify the lock keeps torch 2.11.0+cu130 → `UV_PROJECT_ENVIRONMENT=…/main-py312 uv sync --locked` → `registry.py` note+verify); **A5** compile `preprocess.cpp` (worker v2) — check g++ version; **A6** GPU probe sbatch (nvidia-smi + torch CUDA sanity + `ACCEPTANCE ...`).
2. **Phase B** CPU regen = ONE sequential bash chain per bugs.md §3.1–§3.4 runbook (unchanged durations; BFS threads ≤ 12; explicit `--mem ≥ 16G`; flatten prod + h3 the same way).
3. **Phase C/D** GPU = `train_finetune.py` fine-tune → `clears.py` clears → `verify_paths.py` C → CSV refresh → delta vs paperv7 (USR1+requeue+resume for multi-hour runs).
4. **Phase E–G** = seed-1 + 4 ablations (+ hop-3 BFS extraction alongside).
5. **Phase H** = piggyback stage timings (last #14 gap).

### Etiquette (HPC — every sbatch)
Diagnose `reports/<job>.out/.err` before resubmitting; jobs must be self-diagnosing (`ACCEPTANCE ...` lines); wall ≤ 45 h inside the 2-day limit; keep pending GPU pressure ≤ the 4-GPU quota; batch verification jobs.

### Session-integrity lessons (compaction guard)
- TWO degeneration loops hit this session (garbage repetition). On any recurrence: STOP, re-read the last real tool output, restart from the todo list — never keep generating.
- Verify transforms with ast.parse gates BEFORE accepting writes; locate anchors by content search, never fixed offsets.
- Always `git status --short` before any add; stage selectively (the Phase 0 commit mixed pre-existing tree changes — described honestly in its message, but avoid next time).

---

## 5. Reference mapping — file → location index

Purpose: rediscover exact content without re-grepping. Anchors verified 2026-10-03; re-check after future edits.

### 5.1 Code + configs (local repo `src/`)

| Relevant content | File | Anchor |
|---|---|---|
| main() body (training loop), resolve_config, alloc-conf, __main__ | `pamr/scripts/train_pamr.py` | resolve_config 106–117; `def main():` 118–990; `PYTORCH_CUDA_ALLOC_CONF` 121 (first lines of main); loader + scope validation 255–~296; fine-tune gate + prints 298–~320; optimizer AdamW 487; test_scores.csv writer (query_id, candidate_tail_id, score, is_positive) 916–~922; metrics json keys (fine_tune_backbone, lf_temperature) 963–964; `__main__` 992–994 |
| Freeze/fine-tune + temperature + score contract | `pamr/models/pr/pamr.py` | class docstring (score contract ŷ = g + Δ) ~630–648 (PAMR class def); `__init__` freeze gate + comment 670–678 (signature defaults at 655); auto-detect D/is_rotate ~681–690; no-path residual emb (per-relation, zero-init) 764–771; `recon_gate` 771; criterion switch ~781–789; `train()` eval-override 792–797; `_reset_parameters` zero-init block 797–842 (re-zero of residual/recon_gate 824–829); forward def 1011; vlkge no_grad gate 1036–1041; no-path offset 1044–1050; token build uses `vlkge.get_entity_representations` (NOT no_grad) 836–857; listwise temperature in compute_loss (see "lf_temperature" block); `_aggregate_candidates` + QueryGroupBatchSampler 558–650 |
| B1 leakage-verification runner | `pamr/scripts/verify_paths.py` | docstring/usage 1–32; `_iter_graph_rows` 43–47; `check_meta` (nested registry-hash comparison + per-key loop) 50–~108; `build_exclusion_index` (inverse_map from paths-meta manifest) 111–~180; `check_scope` (A) 183–~200; `load_probe_index` ~205–222; `check_own_edge_and_completeness` (B + sample contract + probe completeness) 224–~292; `check_candidate_identity` (C; prepared-queries cross-check) 294–~371; `main` 373–437 (file 438 lines) |
| Micro-test worlds | `/var/folders/0q/3t_tkstj62xbk4dzvv3wt8300000gn/T/opencode/p0test/test_verify_paths.py` | 5 worlds (clean/own-edge/scope-leak/completeness/identity) + real-shape metas — rerun from `src/` via graphml env |
| PAMR edit micro-test | `/var/folders/0q/3t_tkstj62xbk4dzvv3wt8300000gn/T/opencode/p0test/test_pamr_edits.py` | freeze default / fine-tune split / backward reach / temperature |
| Registry manifest shape (why the nested-hash fix matters) | `pamr/relation_registry.py` | `_build_manifest` 130–145 (inverse_relation_id str→int); `to_dict` adds registry_hash inside 162–165; `inverse_id` 150–154 |
| Worker v2 launcher + artifact shapes | `pamr/models/pr/preprocess.py` | module root paths 29–32 (comment pmr/ stale; paths correct); worker-launcher args 114–131; scope table + defensive default 193–217; mask/uniform-flag reads 219–226; output run dir 228–233; graph-scoped writer (graph_{split}.csv header `u,v,u_type,v_type,edge_type,reverse_edge_type`) ~439–455; graph_meta (top-level relation_registry_hash) ~457–470; probe-queries writer (row_index, query_id, head_id, tail_id, query_relation_id, inverse_query_relation_id — NO is_positive) 471–484; entity/relation mapping writers ~344–352; merge keyed row_meta + emit contract 518–620; completeness audit call 620; paths_meta (manifest; NO top-level hash) 622–638 |
| VL-KGE scoring | `pamr/models/vlkge/score.py` | self-contained vs reference modes 82–106; split loop + rel filter 317–344; scores CSV writer (`query_id,score,is_positive`) 366–373 |
| KGE training entry | `pamr/scripts/train_embedding.py` | REPO_ROOT 21; --embedding_scope 197; scope modes table ~355–366; oracle warning 366–370; checkpoint metadata (embedding_scope, registry manifest, mapping hashes) ~485–492; save_path handling 262–263, 438–439; CLI per-model floor values (margin 12, lr 0.1, epochs 20, batch 512) |
| Registry acceptance | `pamr/scripts/check_registry.py` | main 46+, `--expect-raw/--expect-total` |
| PAMR prod config (locked) | `pamr/configs/pr/pamr.yaml` | backbone repoint + protocol comments 18–31; gates (min_eval_coverage 0.95, patience 5, group_batches) ~36–44; relation_filter train_eval 52–59; seed 42 |
| Backbone + struct configs | `pamr/configs/wikiart_mkg_v2/` | transe_clip.yaml = PAMR backbone source (../data relative paths 4–7); transe_struct.yaml (PAR backbone: use_visual/use_textual false); complex/distmult/rotate_clip.yaml = baselines; " 2" duplicates = user's, DO NOT EDIT |
| Ablations | `pamr/configs/pr/pamr_{depth12,cls,finetune,lf_temp,seed1,par,hop3}.yaml` + `wikiart_mkg_v2_paths_h3.yaml` | each = single-diff from pamr.yaml (depth 12; readout cls; fine_tune_backbone true; listwise_softmax + lf_temperature 0.1; seed 1; struct backbone; max_hops 3 + max_tokens 9 + h3 paths_dir) |
| Eval chain | `pamr/configs/eval/wikiart_mkg_v2_{neg_sampling,score,score_complex,score_distmult,score_rotate}.yaml` | K: train 1 / val 10 / test 10; rejection; score chain per model → `<model>_wikiart_mkg_v2_train.pt` |
| Package identity | `pyproject.toml` (entry points pamr-train/pamr-train-pamr; package-data), `requirements.txt` (floors torch>=2.6 etc.) | |
| Registers | `memory/bugs.md` (this file), `memory/wip.md` (paper + plan; §16 compaction snapshot, §17 integral round, §18 Phase 0, §19 defect round + FIT pointer) | |
| Paper source of truth | `assets/paperv7.tex` (ACTIVE), `assets/analysis/integral.md` §7 (notation authority), `assets/analysis/{review,n-mmkg,overview}.md`, `assets/{result,average,count}.csv` (old values per user: ignore, freshness comes from regen) | |

### 5.2 Workstation artifact provenance (evidence for the register's facts)

| Fact | Artifact (local GDrive sync) |
|---|---|
| Reported backbone = ComplEx (scenario A wrong) | `checkpoints/preproc/wikiart_mkg_v2/full/pamr/pamr_training_metrics.json` (`vlkge_checkpoint` = complex_wikiart_mkg_v2.pt; best_epoch 7; wall 448 s; peak GPU 6 GB) |
| d_model=1536 dims echo (all 22 runs) | `full/pamr/2026062*/train.log` (`VLKGE dims: entity=768, rel=1536 → d_model=1536`) |
| Old train/val/test probe shapes + old-test-eval degeneracy (MRR 1.0000 on 235 queries) | `full/pamr/20260629_071706/train.log` (head: paired shapes; tail: test 235/1.0000) |
| Table-1 source values = strict/ns pairs; PAR/PAMR rows' sources | `full/vlkge/comparison_table.{csv,tex}`; PAR ≡ `full/pamr/pathlen/all/pathlen_last/metrics.json` TEST per-relation values |
| Baseline score sets | `full/vlkge/{TransE,ComplEx,DistMult,RotatE}/{full,artist}/` (scores CSVs + metrics.json + logs) |
| Old checkpoint set | root `checkpoints/vlkge/wikiart_mkg_v2/`: `<model>_wikiart_mkg_v2.pt` (~2.6 GB) + `_artist` (~15 MB) variants — all contract-invalid, kept as history |

### 5.3 FIT server (Phase A onward targets)

| Topic | Path (on FIT, ssh FIT = 172.29.74.81, user hlhdang) |
|---|---|
| Authority files | `~/AGENTS.md` (rules), `/media02/hlhdang/README.md` (map), `~/admin_note.txt` |
| Env system | `/media02/hlhdang/scripts/{make_uv_env.sh,uvenv.sh,registry.py,install_uv.sh}`; catalog `uvenvs/registry.json`; pinned uv `tools/uv/uv` (v0.12.19); bootstrap `source tools/uvenv.sh` |
| Proven sbatch patterns | `projects/ejepwm/slurm/{main,smoke,probe_gpu01,build}.sbatch` (gpu:1/12cpu/mem explicit, USR1+requeue, trap EXIT, ACCEPTANCE lines) |
| Env-mint evidence | `reports/{make_t260,uv_lock_*,adaptive_lock*}.log` (py3.10 downgrade hazards) |
| Campaign destinations (Phase A) | tree → `/media02/hlhdang/projects/pamr`; data → `projects/pamr/data/`; venv → `uvenvs/t260cu126-py312-pamr`; scratch → /dev/shm per job; artifacts sync back → `projects/pamr/checkpoints/…` |
| Local SSH anchor | `~/.ssh/config` Host FIT (key `id_rsa_fit`) |

### 5.4 Rediscovery workflow (future turns)

1. Compaction recovery: read `memory/bugs.md` §1 (Phase 0 audit subsection) → §4 (FIT handoff) → this mapping; only then `memory/wip.md` §16–§19 for paper-plan context.
2. Before touching any config: re-check the anchor row in 5.1 (line numbers may shift after edits — re-grep the anchor string, never trust stale offsets).
3. Before any transform: ast-gate + content-search anchors (§4 lessons).
4. Fact conflicts: artifact evidence (5.2) beats register prose; user decisions beat both.
