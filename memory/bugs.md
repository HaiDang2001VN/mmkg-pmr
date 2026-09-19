# memory/bugs.md — PAMR Defect Register

Why PAMR does not outperform VL-KGE despite consuming frozen VL-KGE embeddings as input. Triage view: fully resolved issues, partially resolved issues, and gaps that have not been touched.

Context files: `pmr/models/pr/pamr.py`, `pmr/scripts/train_pamr.py`, `pmr/models/pr/preprocess.py`, `pmr/models/pr/preprocess.cpp`, `pmr/models/embedding/base.py`, `pmr/dataloader.py`, `pmr/scripts/{prepare,train_embedding,evaluate}.py`, `pmr/models/vlkge/score.py`, `pmr/configs/wikiart_mkg_v2/*`, `configs/pr/pamr.yaml`, artifact evidence in `checkpoints/preproc/wikiart_mkg_v2/full/pamr/`.

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

---

## 2. Partially resolved — code present, known gaps

- **P6 — Metric and protocol asymmetry vs VLKGE.** Was: strict-vs-non-strict tie semantics diverged, evaluation conditioned on BFS reachability only (202 evaluated val queries, Hits@10≈0.97 blocks), and `mode: train_eval` accidentally filtered PAMR training instead of eval-only as commented. Since then: the relation filter mode semantics are explicit and enforced, PAMR reports candidate coverage and ranks candidate tails, tie handling is centralized in `pmr/ranking.py` (Section 1), and the small-block conditioning is documented via coverage. Remaining gaps:
  1. score VL-KGE on exactly the same split-local target candidate set as PAMR (no code path exists yet that consumes a reachable-subset scored by the legacy scorer);
- **PAMR sample-cache manifest** — caches are versioned (`v3`), whitelist-, and hop-aware, but do not yet embed the registry/split-policy hash as load-time validation (the checkpoint side is complete). *(Partially resolved in a previous turn.)*
- **Batch-limited candidate grouping** — `grouped_pairwise`/`listwise_softmax` operate on whatever candidate set appears in the batch; a query-group batch sampler is needed to guarantee full candidate groups per batch. *(Partially resolved in a previous turn.)*
- **Per-relation coverage reporting** — path/no-path counts and mean paths per candidate exist at dataset level; the per-relation breakdown inside `pamr_training_metrics.json` still needs these columns. *(Partially resolved in a previous turn.)*
- **Runtime acceptance script** — registry invariants live inside `RelationRegistry`; a standalone runner for the 40-ID/22-raw-count/hash checks remains. *(Partially resolved in a previous turn.)*

---

## 3. Not touched

- **Artifact regeneration (Phase 6)** — nothing has been re-run: old query CSVs, all-split path CSVs, `*.v2.samples.pkl` caches, preprocessed `graph.csv`, and both old VL-KGE (22 relations) and PAMR checkpoints remain on disk and are invalid under the new contract. Mandatory on compute:
  1. compile `preprocess.cpp` (worker output now has 7 `;`-fields with `row_index`);
  2. `prepare.py` → new query CSVs (target relations only, exact-K negatives);
  3. `preprocess.py` → split-scoped `graph_{train,val,test}.csv` and enriched `*_paths.csv` + `*_paths_meta.json`;
  4. retrain VL-KGE scopes (`train`, `train_val`, optional `train_val_test` oracle) with the 40-ID registry;
  5. retrain PAMR per selected scope with `*_v3.samples.pkl` caches.
- **Runtime structural checks** — no runner yet proves "no validation/test edge inside main val/test path graphs" or identical candidate tails between VL-KGE and PAMR score CSVs; deferred because it depends on regenerated artifacts.
- **Paper-metric refresh** — blocked on the regenerated, leakage-safe comparison.

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
