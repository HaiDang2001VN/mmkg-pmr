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

---

## 2. Partially resolved — code present, known gaps

- **P6 — Metric and protocol asymmetry vs VLKGE.** Was: strict-vs-non-strict tie semantics diverged, evaluation conditioned on BFS reachability only (202 evaluated val queries, Hits@10≈0.97 blocks), and `mode: train_eval` accidentally filtered PAMR training instead of eval-only as commented. Since then: the relation filter mode semantics are explicit and enforced, PAMR reports candidate coverage and ranks candidate tails, and the small-block conditioning is documented via coverage. Remaining gaps:
  1. score VL-KGE on exactly the same split-local target candidate set as PAMR (no code path exists yet that consumes a reachable-subset scored by the legacy scorer);
  2. `evaluate.py` still double-counts ties in `MRR_ns` (`np.sum(scores >= pos_score) + 1`) — centralize tie/multi-positive ranking in one shared evaluator.
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
