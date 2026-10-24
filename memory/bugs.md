# memory/bugs.md — PAMR campaign register (single source of truth)

Why PAMR does not outperform VL-KGE despite consuming frozen VL-KGE
embeddings, plus the complete post-submission experimental campaign
(SOICT 2026, paperv7 submitted 2026-09-20; notification 2026-10-12 —
outcome not yet recorded; camera-ready plan = replace numbers with the
fresh campaign's corrected values if accepted).

**This file was fully rewritten on 2026-10-22 (session compaction
point). It supersedes every earlier revision. Read §0 first, then §1
(decisions), then §3 (remaining program).**

Context files: `pamr/models/pr/pamr.py`, `pamr/scripts/train_pamr.py`,
`pamr/scripts/train_embedding.py`, `pamr/helpers.py`,
`pamr/dataloader.py`, `pamr/scripts/prepare.py`,
`pamr/scripts/verify_paths.py`, `pamr/models/vlkge/score.py`,
`pamr/models/pr/preprocess{.cpp,.py}`, `pamr/configs/**`.

---

## 0. READ-FIRST — state snapshot (2026-10-22, compaction entry point)

### COMPLETE and verified

- **Environment (workspace era)**: uv WORKSPACE rooted at
  `/media02/hlhdang/projects/pyproject.toml` (members ejepwm,
  DreamZero-Project-IML, akg-sv, **pamr/src**); ONE shared venv
  `uvenvs/t251cu121-py312` (python 3.12.14, torch 2.5.1+cu121 /
  tv 0.20.1 / ta 2.5.1, numpy 2.5.3, pandas 3.0.6, sklearn 1.9.1);
  union lock `projects/uv.lock` (509 pkgs); runtime =
  `source /media02/hlhdang/tools/uvenv_ws_env.sh` then direct
  `uvenvs/t251cu121-py312/bin/python -m …`. `uv lock --check` passes.
- **Phase B (fresh 2026-10-10 lineage)**: registry PASS (22 raw / 40
  total, hash `ff50ee5aabe9ba5e…`), prepare 709 train / 202 val / 239
  test queries (K=1/10/10; test split is artist-relations only under
  the `train_eval` filter), BFS prod + h3, flatten, **verify A/B PASS**
  on all splits for both configs. Artifacts: preproc root holds
  `{train,val,test}_queries.csv` + `metadata.json` (+ compat copies
  under `full/`); `full/pamr/` and `_h3/pamr/` hold paths CSVs, graph
  CSVs, metas, mappings, manifest, probe queries.
- **ALL 5 BACKBONE SLIM CHECKPOINTS LANDED** in
  `checkpoints/vlkge/wikiart_mkg_v2/` (trained with the cached
  evaluator; bs 2048; 20 epochs; train scope):
  | file | model | best_val MRR | best epoch |
  |---|---|---|---|
  | transe_wikiart_mkg_v2_train.pt | TransE | 0.247489 | 20 |
  | complex_wikiart_mkg_v2_train.pt | ComplEx | 0.311879 | 3 |
  | distmult_wikiart_mkg_v2_train.pt | DistMult | 0.306615 | 3 |
  | rotate_wikiart_mkg_v2_train.pt | RotatE | 0.2501 (final) | — |
  | transe_wikiart_mkg_v2_struct_train.pt | TransE (struct-only) | 0.098016 | 16 |
  All slim-contract-verified: `model_state_dict` + `relation_registry`
  (+ hash `ff50ee5a…`) + `embedding_scope=train` + mapping hashes, 40
  relations, 224,166 entities, d_ent=d_rel=768, **no optimizer keys**,
  stage_ files cleaned.
- **Cached evaluator** (evaluate_kge pooled path): per-relation pool
  representations encoded once; `encode_entities`/`score_from_repr`
  API on all four scorers; parity EXACT on CPU (gates 78779, 78896
  incl. RotatE stack-layout fix) and CUDA-accepted (gate 79035;
  GPU contract = metric-abs 1e-4 because kernel-order fp noise flips
  deep-tail ties deterministically — observed Δ2.2e-12 abs — while
  structural errors move metrics orders of magnitude). Speed: eval
  cycle 1h40m → ~10-15 min per epoch (10×); backbone waves completed
  in single walls (2h58m / 7h58m / 6h18m / 9h24m).
- **Phase D v1 (79379)**: the 4 baseline score chains PASSED (wrote
  `full/vlkge/<TransE|ComplEx|DistMult|RotatE>/{train,val,test}_scores.csv`
  + score metrics). `pamr-prod` then FAILED on the stale June-era
  `queries_dir` layout (`…/full/` vs the fresh preproc-root queries).
  FIXED: all 9 `pamr/configs/pr/*.yaml` now point `queries_dir` at the
  preproc ROOT (compat copies remain under `full/`). Job resubmitted.
- **Audit round complete** (§5 lists findings; F-9 ranking lf leak +
  F-11 cosmetic fixed in `aba04ab`; regression gate 78922 ALL OK).
- **NetGate**: compute nodes HAVE outbound internet (78622: hf 200,
  api.wandb.ai reachable, pypi 200). Staging rule retained as
  robustness (a mid-epoch network flap must not kill 8h runs);
  wandb NOT used (user decision).
- **Aggregators authored + validated**: `scripts/aggregate_campaign.py`
  (June-format grouped result.csv + count.csv + average_macro math
  verbatim; june-selftest mode = INFORMATIONAL only — June CSVs are
  untrusted per locked decision), `scripts/phase_h_timings.py`
  (dry-validated on June metrics).

### IN FLIGHT at compaction

- **Phase 0 diagnostic (2026-10-23; encoder probe 79741, parity
  79756) — TWO DEFECTS CONFIRMED**:
  (1) **torch 2.5.1's MHA fast path is wrong** in eval with a padding
  mask: the default encoder differs from a dense SDPA reference by
  max_abs **3.01** on random weights; setting
  `torch.backends.mha.set_fastpath_enabled(False)` makes it exact
  (0.0). The nested-tensor toggle is irrelevant. Fixed globally in
  `pamr/models/pr/pamr.py` and enforced by the parity gate.
  (2) The **dense** v1 encoder is also degenerate (manual/dense output
  constant, residual std 1.6e-6): the K=1 pairing leaves 699/709 pairs
  already satisfied, so only the 2 pathless negatives carry gradient
  and the model learns a constant. Fixed by K=10 training + pathless
  masking, with the path-dependence gate after training.
  The earlier probe's "default == manual" was a coincidence: both
  outputs were constant.
- **Job 79444 (Phase D) — PASSED 2026-10-22T02:20:10Z** (11 steps,
  ALL-STEPS-DONE; verify-C clean: 0 scope violations, 0 own-edge
  violations, 0 probe defects, candidate identity 239 queries /
  0 problems). Fresh test-protocol metrics (239 queries, 1 pos + 10
  negs, K=10):
  | row | test MRR | H@1 | H@3 | H@5 |
  |---|---|---|---|---|
  | VLKGE-TransE | 0.7006 | 0.5523 | 0.7950 | 0.9163 |
  | VLKGE-RotatE | 0.6171 | 0.4435 | 0.7364 | 0.8787 |
  | VLKGE-DistMult | 0.5948 | 0.4268 | 0.6987 | 0.8285 |
  | VLKGE-ComplEx | 0.5614 | 0.3724 | 0.6695 | 0.8117 |
  | PAMR (best ep1, val .8034) | 0.7215 | 0.5858 | 0.8159 | 0.9247 |
  | PAR (best ep3, val .7457) | 0.6564 | 0.4979 | 0.7699 | 0.8996 |
  PAMR ≡ TransE + 20.6804·1[has_path] (control reproduces MRR to the
  digit; 19 rank-changed queries = the 20 pathless test negatives).
- **Phase E (79455) — PASSED**: seed1, finetune, depth12, cls, lf_temp,
  hop3 all rc=0; hop3's .8044 is the same presence artifact at +53.96
  (control-reproduced). Seeding/depth do not change the ranking (seed1
  and depth12 identical to PAMR: 239/239 ranks).
- **Pilot round 3 (79767) — path-statistic readout**: the feat head
  (hops/has_path/node+edge-type fractions) lifts all-path strict MRR to
  **0.766** (backbone 0.7295; hard arm worse, 0.7157; hard_lf cancelled).
  The length heuristic's strict 0.8245 is a **tie artifact**: non-strict
  0.6328 vs PAMR 0.7660 (both tie-free). **Metric policy: non-strict
  (tie-aware) is primary; strict is reported alongside.** All-path
  non-strict: PAMR vs backbone +0.0366 (CI −2.4e-5..0.0732, marginal);
  vs length +0.1332 (CI 0.0977..0.1704). Decision: analysis paper with
  non-strict primary; 5-seed campaign launched with random_lf
  (79774 group A / 79775 group B) incl. no-feats and all ablations,
  K=10 + K≤100.
- **Pilot round 1 (79760) — FAILED all four arms**: random/hard ×
  pairwise/listwise all give all-path MRR == backbone == presence
  control (0.7295; bootstrap 0.0) with full MRR == 0.7215. Diagnosis:
  **train paths are 80% 1-hop** (mean 1.34) while **test paths are 3–5
  hops** (mean 3.6); the 1-hop train paths use OTHER direct relations
  between the same pair (edge type 17 = isRelatedArtist etc.), which the
  old own-edge mask (query relation + inverse only) did not remove.
  Round 2 (protocol v2.1, job 79763): mask **every** direct head-tail
  edge (any relation) and re-extract all paths (v1, random/hard, K100,
  prod+h3) with the stricter verifier. After the fix the train/test
  path-length distributions are aligned enough to transfer: a trivial
  **length-only** heuristic (-hops) scores **all-path MRR 0.8245** vs
  the backbone's 0.7295 (full 0.8290). The pilot gate now requires
  beating the length heuristic as well as the backbone and presence
  control. Remask 79763 COMPLETE: all splits re-extracted and verified
  with 0 own-pair violations; new hops: train pos/neg mean 2.39/3.17,
  test pos/neg 3.60/4.61 (19/20 pathless negatives). Pilot round 2 =
  job 79766. **FAILED all four arms**: all-path MRR = 0.7295 (exactly
  backbone and presence control) while the length heuristic scores
  0.8245 (bootstrap −0.095, CI [−0.143,−0.046]); residual std only
  0.016–0.064 around a huge constant. The mean-pooled readout is
  effectively length-invariant, so the model cannot learn even the
  strongest path signal. **Decision point (Q8): architecture vs
  analysis paper.**
- **K=100 protocol is "up to 100"**: the artist split-local pools hold
  fewer than 100 eligible candidates for every query, so `prepare` now
  supports `allow_variable_negatives: true`; actual negatives are
  19–61 per query (val median 50, test median 45). This is the
  co-primary harder protocol (documented; not a bug).
- **Automated journal chain submitted (2026-10-24, user away)**:
  - 79850 `v2_lenonly` (GPU): length-only feat ablation, 5 PAMR seeds.
  - 79851 `v2_analysis` (CPU, afterok:79850): logistic + aggregate +
    stats CIs (1-seed baselines) + figures + tables + runtime +
    verify → **reports/checkpoint_J1.md**.
  - 79852 `v2_seed_job 1` / 79853 `v2_seed_job 2` (GPU, 48 h each):
    TransE→ComplEx→DistMult→RotatE sequentially; slims
    `<model>_train_s{1,2}.pt`.
  - 79854 `v2_backbone_rescore` (GPU, afterok:79852:79853): K10/K≤100
    score/eval + path controls for the 8 new checkpoints.
  - 79855 `v2_j2` (CPU, afterok:79854): 3-seed baseline aggregation,
    stats, figures, tables, verify → **reports/checkpoint_J2.md**.
  Expected: J1 ~1.5–2 h after start; J2 ~28–31 h. No await loops; the
  agent resumes at J2. Report tables under `assets/report_v2/`.
- **Campaign v2 COMPLETE (79774 A / 79775 B, 2026-10-24)**:
  random_lf, 5 seeds, K=10 + K≤100, feat/no-feat, all ablations,
  v1-fixed and loser-arm diagnostics. Test non-strict MRR (= strict;
  model scores are tie-free):
  | method | K=10 | std | K=100 | std |
  |---|---|---|---|---|
  | TransE | .7006 | — | .4508 | — |
  | RotatE | .6171 | — | .3790 | — |
  | DistMult | .5948 | — | .3484 | — |
  | ComplEx | .5614 | — | .2838 | — |
  | PAMR | **.7678** | .0055 | **.5982** | .0062 |
  | PAR | .7369 | .0151 | .5645 | .0265 |
  | PAMR-no-feats | .7215 | 0 | .5227 | 0 |
  All-path (presence-free): K10 PAMR .7714 vs backbone .7295;
  K100 .6290 vs .5543; no-feats = backbone exactly. Length heuristic
  strict .8245/.7425 but non-strict .6328/.5242 (tie artifact).
  Ablations K10: hop3 .8052, cls .7709, no_mask .7677, finetune .7663,
  depth12 .7642, lf_temp .7577; K100: hop3 .6478, finetune .6086,
  cls .6060, no_mask .5912, depth12 .5872, lf_temp .5775.
  Diagnostics: v1-fixed .7201/.5261; loser-hard .7095/.5037.
  Tables: `src/assets/report_v2/` (main, per_relation, ablations,
  all_path, controls, diagnostics). **Metric policy: non-strict
  (tie-aware) primary; strict alongside.**
- **Journal plan v2 (approved 2026-10-23)**: K=10 + presence control +
  pathless-probe masking; K=10 and K=100 co-primary; pilot random-K10
  vs hard-K10 (top-10 of 100 sampled); 5 PAMR seeds (checkpoints only
  seed42+PAR); v1-fixed = eval-only + 1-seed retrain; two fallback
  rounds; stop only at a failed pilot gate after fallbacks and at
  final aggregation. v1 diagnostics go in the appendix.
- **Operational note**: the operational sbatch dir is
  `/media02/hlhdang/projects/pamr/slurm/` (a COPY of the repo's
  `src/slurm/`); changes must be rsynced to both. The 79455 phase E
  ran the pre-hardening copy (no disk gate) — harmless, now synced.
- **Static audit round 2 (2026-10-22, pre-start)**: found + fixed two
  latent correctness defects that the first Phase D run never reached:
  (a) evaluate.py could not resolve `prepare_config: configs/eval/…`
  (missing PKG_ROOT candidate) — proven by CPU pre-flight 79454 which
  ran all four evaluate chains end-to-end (ALL-OK; all four per-model
  `metrics.json` were then rewritten in 3–4 s each inside 79444);
  (b) the aggregator read PAMR/PAR nested `Hits` with int keys after a
  JSON round-trip (string keys) → H1/H3/H5 would have been blank. Also
  hardened: clone_paths fail-fast + /media02 free-space gates
  (phaseD 8G, phaseE 12G), ranking per-relation lf NaN guard.

### NEXT ACTIONS (strict order — journal v2, approved 2026-10-23)

1. **Phase 0 remaining**: parity test (default vs manual encoder) +
   path-dependence gate (residual std after training; config
   `min_residual_std`); v1-fixed eval-only on existing checkpoints;
   v1-fixed retrain (K=1, mask, 1 seed).
2. **Phase 1 artifacts**: v2-random train K=10 (prepare + paths prod/h3,
   val/test byte-identical); v2-hard (top-10 of 100 sampled pool
   candidates); K=100 val/test queries + paths + baseline score chains.
3. **Phase 2 code**: masked candidate-level loss (`compute_loss(...,
   has_path)`), `--seed`, in-run K=100 test eval (`extra_tests`),
   external evaluator, presence-control + path-conditional scripts,
   stats/bootstrap, v2 configs, chained sbatch (pilot + campaign with a
   gate sentinel).
4. **Phase 3 pilot**: {random,hard} × {grouped_pairwise,listwise};
   gate = residual path-dependence + all-path MRR > backbone AND >
   presence control (bootstrap CI excluding 0). Fail → 2 rounds → stop.
5. **Phase 4 campaign**: 5 seeds × {PAMR, PAR, finetune, depth12, cls,
   lf_temp, hop3, no_mask}; v1-fixed; loser arm single-seed; K=10 +
   K=100; aggregation/stats.
6. **Phase 5**: journal tables/figures, methods, appendix, register
   sync + mirror after every milestone.

### Hard operating rules (from mistakes)

- Login node = login/launch/inspect ONLY; every compute task is a
  slurm job (CPU jobs without `--gres`, short `--time`).
- Pre-flight `df -h /media02` before >5 GB writes; pause + report if
  free < 4 GB (two concurrent stage files ≈ 4.2 GB).
- Wave scheduling policy: 2 jobs × 1 GPU (user decision); the shared
  account QOS counts PENDING GPU jobs too.
- `--nodelist` requires `-N1`; avoid gpu03 when its free RAM is low
  (co-tenant jobs); ALWAYS explicit `--time` (DefaultTime 6h).
- Diagnose `reports/<job>.out/.err` before any resubmit.
- Sync sbatch changes to BOTH `/media02/hlhdang/projects/pamr/slurm/`
  (operational) and `.../pamr/src/slurm/` (repo mirror).

---

## 1. Locked plan and decision record (chronological)

1. **Compute = FIT cluster** (Slurm, single `batch` partition).
2. **Full regeneration matrix**: retrain TransE (backbone + baseline),
   ComplEx, DistMult, RotatE baselines + structural-only TransE (PAR
   backbone); retrain PAMR + PAR; rescore everything on the prepared
   query CSVs.
3. **Backbone scope = train everywhere** (one train-scope checkpoint
   backs val + test evaluation and baseline scoring).
4. **Evaluation protocol = 10 negative candidates per query**; PAMR
   seeds 42 + 1 (n=2, std ddof=1); baselines single seed 42. (KGE
   *training* uses num_neg_samples=1 per the June-provenance configs.)
5. **Ablations = core 5**: fine-tuned backbone, K=3 (hop3, using the
   `_h3` paths), depth N=12, readout cls, B3 contrastive =
   listwise_softmax + lf_temperature 0.1.
6. rclone/GDrive sync superseded — FIT staging via rsync only.
7. **Camera-ready = correct the numbers** if accepted.
8. **Backbone = TransE-faithful** (`pamr.yaml` → retrained train-scope
   TransE). STANDING CAVEAT: June artifacts prove the published
   numbers actually came from a frozen **ComplEx** checkpoint
   (`vlkge_checkpoint` field in the June pamr metrics JSON; d_model
   1536). The regeneration is a NEW configuration; deltas have two
   sources. Old `assets/{result,average,count}.csv` values are
   UNTRUSTED (H1/H3 swap artifact + different split/source branches) —
   fresh outputs are authoritative.
9. **evaluate_kge vectorization FIRST** (user 2026-10-06) — pooled
   batched evaluator, parity-gated; legacy loop preserved as
   `_evaluate_kge_per_triple`.
10. **Batch size: probe-then-decide → 2048 adopted** (user 2026-10-11)
    for all 5 backbone configs; June provenance was 512/1024; lr 0.1
    unchanged (protocol deviation documented). Probe data: per-triple
    trainstep 305/147/119 µs at bs 512/2048/8192; GPU util ≤9.1% at
    every bs (host-bound; bs only cuts wall time).
11. **Host-path engineering** (user: combine with bs 2048): KGDataset
    tensorized ID access (bit-identical batches, no RNG in dataset) +
    grouped-vectorized negative sampler (distribution-faithful;
    hub-drop + 4-tuple keep-mask contract unchanged) — equivalence
    gates ALL OK (78344/78625).
12. **Cached-scored evaluation redesign** (user 2026-10-12) after the
    measured 1h40m/epoch eval cycle at 97% GPU: encode_entities +
    score_from_repr on all four scorers; pool encoded once per
    relation; parity exact on CPU, 1e-4 metric tolerance on CUDA.
13. **No wandb** (user); self-diagnosing ACCEPTANCE logs + JSON
    metrics are the authority. Compute-node internet confirmed but
    staging stays the rule.
14. **Login-node etiquette** (user 2026-10-06): login = login/launch
    only; CPU tests as slurm jobs.
15. **Static audit before GPU release** (user 2026-10-13): full
    campaign-surface line audit, no execution; findings → fix round;
    audit annex folded into §5/§6 here.
16. **2026-10-06/08 storage incident**: /media02/hlhdang wiped by the
    admin (lease context) and restored from an Oct 6 14:46 UTC
    snapshot. Recovery protocol: integrity check → mirror sync →
    env/Phase-B re-validation → resume. Nothing irreplaceable was lost
    (local git = authoritative).
17. **Phase D baseline scoring output-own** (pre-fix round): each
    baseline writes `full/vlkge/<Model>/` (no shared-root clobber);
    evaluate runs per model with `scores_dir` configs; verify-C reads
    the TransE chain's scores.

---

## 2. Phase record (chronological, condensed)

### Phase 0 + audit (2026-10-03/04)
Provenance audit from synced artifacts; verify_paths runner (checks
A scope / B own-edge+completeness / C candidate identity); rename
sweep `pmr`→`pamr` (47 refs); fine_tune + lf_temperature wired
end-to-end. Two critical defects fixed (d4c2d23): train_pamr missing
`def main():`; verify_paths nested-vs-top-level registry-hash compare.
+ inverse-id hardening; requirements floors; smoke repoints.

### Phase A (2026-10-04)
Staging to FIT verified (md5/cksum); pyproject floors; preprocess.o
compiled (g++ 11.4, `-std=c++23 -O3`). Environment story evolved:
cu130 era was GPU-dead on the fleet (drivers 525/535, CUDA 12.2
ceiling) → escalated → FINAL = uv workspace with torch 2.5.1+cu121.

### Phase B + revalidations (2026-10-04 → 10-10)
Defects found by execution: c1c68e6 (workspace-relative dir collisions
— silent split skips; REPO_ROOT-relative now; loud failures),
cf4d0dd (keep_temp=false deleted row-identity probe queries —
preserved unconditionally), a7f410a (thread/worker etiquette: prepare
pool capped 10, BFS threads 10, OMP/MKL=10). pandas-3 revalidation
job 78287 ALL GREEN. Graph CSVs were reclaimed for disk after the
evidence archive; the Oct-8 restore lacked them → post-restore regen
78592 (ALL GREEN, 13m58s) restored the full contract set (fresh
2026-10-10 lineage).

### Phase C0 (2026-10-05)
CPU machinery smoke 78141 FULLY GREEN (synth ckpt → score.py →
evaluate.py → train_pamr 1 epoch; artifacts in `cpu_smoke/` scratch —
later reclaimed for disk).

### Vectorization + Phase-D pre-fix round (2026-10-06)
81d9b31 pooled evaluate_kge (chunked per-relation forwards, truth
masking, stable argsort; index space provably identical to the legacy
loop); parity harness (scratch/test_loader_sampler_equiv + test_eval
kge parity); gate 78321 ALL OK. 6a025ed Phase-D pre-fix: per-model
score `output_dir`s + 4 `neg_eval_<model>.yaml` + verify-C repoint
(the 4 chains previously clobbered one shared CSV). fc77e6d probe
hardening (the probe step crashed on missing keep-mask alignment,
then parked at shutdown on a non-daemon poller).

### GPU-util / sampler rounds (2026-10-06 → 10-11)
Wave-A v1 failed on a hub-relation sampler fail-fast ("Every candidate
tail … is a known positive") → ef005be whole-triple drop + keep mask
+ train-loop alignment (micro-tests 3/3). 78297/78298 relaunch led to
the EVAL discovery: the process was healthy-but-host-bound in the
per-triple val eval (per-triple `.item()` syncs + Python list sorts of
~216K-candidate pools) → d58e57f protocol-neutral speedups (one
`.cpu()` per batch for sampler inputs; `--eval_batch_size` for
eval loaders; PYTHONUNBUFFERED). Later: KGDataset tensorization +
grouped-vectorized sampler (see §1.11). The split-pool ORDER defect
(43db1a0): `_build_per_relation_pools` built UNSORTED split-local
pools while the batched evaluator's searchsorted masking requires
sorted order — surfaced as a false "tail absent from pool" guard trip
on artist relations (rel 12 isInfluencedBy); census 78623 proved ZERO
membership outliers; fix = sort pools at the source + parity slices
force artist rows.

### Admin GPU-block episode (2026-10-06)
~17:00–18:13 +7 all new GPU submissions were insta-cancelled by root
(lease-renew banner); CPU jobs survived; block lifted 18:13 (canary
completed). Registered as a recurring risk pattern; canary-test before
believing a cancel is a code failure.

### Storage wipe + restore (2026-10-07/08)
/media02/hlhdang vanished cluster-wide; restore = Oct 6 14:46 UTC
snapshot; integrity PASS (tree/env/artifacts/engineered src); the only
deliverable delta = graph CSVs (previously reclaimed) → Phase B regen.
Recovery deltas also cleaned (stale src/uv.lock; stray
src-memory-and-assets dir). Full incident evidence kept in git history
(register commits around 2026-10-08).

### Wave rounds (2026-10-10 → 10-21)
- Pre-wave fixes: resume path crashed on an inline `pamr.utils`
  re-import shadowing (`UnboundLocalError` at load_checkpoint; only
  ever fires with `--resume_from`) → inline import removed.
- 78331/78332 first probes: hub-drop fine; the FIRST pooled-eval GPU
  exercise OOM'd (score-budget chunking ignored the fused forward's
  own (rows×dim) materialization → ~33 GB transient) → afc79eb
  forward-ROWS budget (131,072) + expandable_segments + train
  heartbeat prints (10% granularity; large epochs were otherwise
  silent — that silence cost a misdiagnosis and one sacrificial
  SIGINT).
- All five backbones ultimately trained/completed under the cached
  evaluator with stage-file resume across 12h walls (USR1@300s +
  stage-sync + manual resubmission; the USR1 path exits cleanly, so a
  wall-hit job must be RESUBMITTED — it does not self-requeue).
- Timing evidence: epoch cadence ~15–18 min (train ~4–5 min + eval
  ~10–13 min) vs 1h40m legacy; wave-1 jobs: transe 78920 (2h58m),
  complex 78921 (7h58m); wave-2: distmult 79113 (6h18m), rotate
  79114→79276 (12h wall + 9h24m resume); wave-3 struct 79211.
  Slim best_val table in §0.
- Diagnostics recorded: struct (multimodal-off) val MRR ≈ 0.098 vs
  multimodal 0.25–0.31 — the PAR-vs-PAMR contrast is structural.
- NOTE: the backbone `.out` FINAL TEST blocks (all-entities pools,
  inductive masking) are DIAGNOSTIC-only; paper-grade numbers come
  from Phase D's score/evaluate chains.

### Audit round (2026-10-13)
Full campaign-surface static audit (26 files; no execution). Mechanical
lints clean; config/sbatch matrices clean; per-tier line reads clean
except **F-9** (ranking.py per-relation lf accumulators appended a
stale rank_lf when a query was skipped by the lf-exclude branch —
fixed with a per-query `lf_recorded` gate; no campaign-metric impact)
and **F-11** (duplicate line in score.py padding). Both fixed in
aba04ab; regression gate 78922 ALL OK.

### Phase D (2026-10-22)
79379: 4 score chains PASSED; pamr-prod FAILED on stale June
queries_dir (`…/full/`) vs fresh preproc-root queries → all 9 PR
configs fixed (+ compat copies under full/) → rerun 79444.
Pre-start audit round 2 (before 79444 began running): fixed evaluate
prepare_config resolution (0ea2f65) and aggregate Hits key coercion
(68ea46c); hardened clone_paths + disk gates (cd5ceee) and the
per-relation lf NaN edge (e891aa7). CPU pre-flight 79454 then ran all
four evaluate chains ALL-OK in 26 s. 79444 picked up the fixes at
start and PASSED all 11 steps at 02:20:10Z (see §0 for the results
table and verify-C evidence).

### Phase 0 — journal v2 diagnosis (2026-10-23)
Encoder probe 79741 and parity 79756: the torch 2.5.1 MHA fast path is
wrong in eval with padding masks (default vs dense SDPA max_abs 3.01;
0.0 after disabling) → fixed globally in `pamr.py` + parity gate.
Independently, the dense v1 encoder is degenerate (constant output,
residual std 1.6e-6) because the K=1 pairs are 98.6% satisfied → fixed
by K=10 + pathless masking. Both fixes are required.

---

## 3. Remaining program (exact commands)

1. **Phase D — DONE** (79444 PASSED 2026-10-22; see §0 results table).
   Artifacts:
   - `full/vlkge/<Model>/{train,val,test}_scores.csv` +
     `metrics.json` (per model).
   - `full/pamr/` {pamr_best.pt, pamr_training_metrics.json (incl.
     `dataset_build_timings_s`, test.per_relation), test_scores.csv (probe-level)}; `full/par/` same shape.
   - verify-C PASS (PAMR candidate identity vs TransE scores).
2. **Phase E/G** — IN FLIGHT: `sbatch -N1 --nodelist=gpu01,gpu02,gpu04
   slurm/phase_e.sbatch`. Gates: each ablation's
   `pamr_training_metrics.json` under its output-own dir
   (`full/ablations/{seed1,finetune,depth12,cls,lf_temp}/`; hop3 in
   `_h3/pamr/`).
3. **Phase H + aggregation** (CPU slurm job template):
   - `scripts/phase_h_timings.py` (PAMR-family walls/GPU peaks/build
     timings; per-epoch means) → `assets/report/timings.csv`.
   - `scripts/aggregate_campaign.py --mode fresh` → grouped
     `result.csv` + `count.csv` + `average.csv` (test split, weights =
     test query counts 87/85/33/34 = 239). Optional val sheets.
   - Delta vs June numbers = informational (June untrusted).
4. **Paper-side**: map fresh artifacts to the paperv7 tables (see
   `memory/wip.md`); camera-ready replacement if accepted.
5. **Register + mirror sync after every milestone** (rsync `memory/`
   and any changed code to `/media02/hlhdang/projects/pamr/src/`).

---

## 4. FIT cluster handoff (verified through 2026-10-22)

### Access + authority
- `ssh FIT` = 172.29.74.81 (round-robins login01/login02), user
  `hlhdang`, key `id_rsa_fit` (~/.ssh/config).
- Reading order: `~/AGENTS.md` → `/media02/hlhdang/README.md` →
  `~/admin_note.txt`. `~/AGENTS.md` network row was corrected
  2026-10-10 (NetGate evidence; backup `~/AGENTS.md.pre-netgate.bak`).
- Campaign root: `/media02/hlhdang/projects/pamr/{src,data,
  checkpoints,slurm,scripts,reports}`.

### Environment (FINAL — uv workspace)
- ONE venv `uvenvs/t251cu121-py312` (torch 2.5.1+cu121 trio,
  numpy 2.5.3, pandas 3.0.6, transformers 5.18.0, sklearn 1.9.1).
- Workspace root `projects/pyproject.toml` (virtual; members ejepwm,
  DreamZero-Project-IML, akg-sv, pamr/src); THE lock
  `projects/uv.lock`; pin policy: only torch triad + py3.12 (+ swm
  0.0.6); resolver deadlocks → extend root
  `[tool.uv] override-dependencies` (numpy>=2.2.6,<3,
  transformers>=4.50,<6, tokenizers>=0.21, datasets>=2.19).
- Runtime: `source /media02/hlhdang/tools/uvenv_ws_env.sh`; run via
  the direct venv python. NEVER `uv run --project` (spawns a
  forbidden repo-local `.venv`). Re-lock/sync: `cd
  /media02/hlhdang/projects && uv lock && uv sync --locked --no-dev
  --all-packages` (uv pinned `tools/uv/uv`).
- NO new venvs/conda/pip — ever. Driver family gate map: 535→cu121
  (only runnable family), 550→cu124, 560→cu126, 570→cu128,
  580→cu130; family changes require an on-node smi+matmul proof job
  (evidence 78282). Observed drivers: gpu01 525/555-era A100-80/40,
  gpu02 555.42.02 A100-40, gpu03 580.82.07 (CUDA-13!) A100, gpu04
  535.216.03 V100-32. **torch cu121 runs on ALL of them** (cu121
  wheels need only driver ≥ 525; the CUBLAS/CUDA-13 incompatibility
  applies to cu130 wheels, not cu121).

### Cluster facts
- Slurm single `batch` partition; DefaultTime 6h → ALWAYS explicit
  `--time`; DefMemPerCPU=2 GB → ALWAYS explicit `--mem`;
  `MaxCPUsPerJob`=12 per account; QOS gpu_limit = 4 GPUs/user
  (RUNNING + PENDING both count; the other agent shares this
  account); MaxTime 2-00:00:00.
- GPU nodes: gpu01 8×GPU/980 GB; gpu02-04 4×GPU/~240 GB; 256
  threads. TmpDisk=0 → scratch in `/dev/shm` (trap … EXIT required).
- Storage `/media02` 5 TB shared; free fluctuated 1.9 → 166 GB
  (communal); PRE-FLIGHT `df` mandatory; >5 GB writes need a registry
  note. `/home` NFS ~195 GB (policy: dotfiles only; ~18 GB free at
  last check).
- Compute nodes have internet (NetGate 78622); still stage downloads
  on login for robustness. Nodes go OFFLINE for storage only via NFS.
- Watch UI http://172.29.74.82:8012/ (login-gated; lease-renew banner
  seen 2026-10-06 → the storage wipe followed; treat lease state as
  a first-class risk).
- Known pathologies: fairshare priority can queue 1-GPU jobs for
  hours-to-days (projected start jumped days in dense periods —
  backfill on short-job gaps is the normal start path); transient
  root-cancels of new GPU jobs occur during admin actions (canary
  before diagnosing code); node RAM can be co-tenant-starved (gpu03
  seen at 3.2 GB free; watch `sinfo -N -o '%N %T %e'`).

### Campaign layout on FIT
- Data: `projects/pamr/data/wikiart_mkg_v2/` (triples 994 MB +
  features/ 1.8 GB).
- Preproc: `checkpoints/preproc/wikiart_mkg_v2/` (queries at ROOT +
  `full/pamr/` paths/graphs/metas/manifest/probes; compat query
  copies under `full/`) and `checkpoints/preproc/wikiart_mkg_v2_h3/
  pamr/` (K=3 paths).
- Backbones (5 slims): `checkpoints/vlkge/wikiart_mkg_v2/` (see §0
  table). `stage_<name>.pt` files = in-flight full checkpoints
  (USR1/EXIT sync target); slim copyback removes them at completion.
- Reports: `projects/pamr/reports/` (all job logs; ACCEPTANCE lines).

---

## 5. Defect register (all RESOLVED; historical + audit findings)

| Defect | Fix / commit | Notes |
|---|---|---|
| train_pamr missing `def main()` | d4c2d23 | body unreachable |
| verify_paths nested-hash compare | d4c2d23 | + inverse-id hardening |
| Workspace-relative dir collisions | c1c68e6 | preprocess WS_ROOT, score ckpt path, prepare cwd; loud failure |
| Probe-queries artifact deletion | cf4d0dd | preserve unconditionally; flatten+verify on flattened dirs |
| CUDA metric guards (CPU runs) | ad2fe02 | reset_peak guarded by device |
| Output-own paths_dirs (clobber) | ad2fe02 | PAR → full/par; ablations → full/ablations/<name>; hop3 → _h3 |
| Baseline score-chain clobber | 6a025ed | per-model `output_dir` + 4 `neg_eval_*` + verify-C repoint |
| Hub-relation sampler crash | ef005be | whole-triple drop + keep mask + loop alignment |
| Per-triple CUDA syncs in sampler | d58e57f | one `.cpu()`/batch; parity tested |
| Pooled-eval GPU OOM | afc79eb | forward-ROWS budget 131,072 + expandable_segments |
| Split pools UNSORTED (artist rels) | 43db1a0 | sorted at source; artist rows forced into parity slices |
| Probe bench crash + zombie park | fc77e6d | mirror train loop; daemon poller; bench_safe FAILED lines |
| evaluate_kge per-triple wall | 81d9b31 + f4d5a50 | pooled + cached; parity gates 78321/78779/78896/79035 |
| RotatE cached-scorer ULP tie flips | rotate stack-layout fix | replicate forward's `(f,2)` reduction layout |
| Resume `utils` UnboundLocalError | inline import removed | first-ever resume exercise; killed 78885/78886 at load |
| Train batch loop lacked `i` | e9f1dfd | heartbeat NameError killed 78347/78348 |
| Silent long epochs | heartbeat prints | 10% granularity, flush=True |
| ranking F-9 per-rel lf leak | aba04ab | per-query `lf_recorded` gate; no campaign-metric impact |
| score.py duplicate line F-11 | aba04ab | cosmetic |
| PR `queries_dir` stale (June `full/`) | config fix (9 files) | pamr-prod Phase D failure; root canonical + compat copies |
| evaluate.py prepare_config resolution | 0ea2f65 | missing PKG_ROOT candidate → FileNotFoundError on `configs/eval/…`; first reached only after pamr-prod was fixed; validated by CPU job 79454 (4 chains ALL-OK) |
| aggregate PAMR/PAR Hits keys | 68ea46c | JSON round-trips nested int keys to strings → H1/H3/H5 blanked; `_hits_value` coerces both |
| aggregate macro_average denominator | 048de6a | first fresh run crashed: `np.where(keep[:,None],1.0,0.0).dot(W)` is (n,1)·(4,) → ValueError; rows now divide by `W.sum()` (June math for complete rows) |
| Path own-edge mask too narrow | c7736b0 | masked only the query relation/inverse; direct alternative relations (e.g., isRelatedArtist) gave 1-hop train paths (train mean 1.34 vs test 3.6 hops), so the model learned a non-transferable shortcut. Now masks every head-tail edge (any relation) |
| clone_paths masked cp failures | cd5ceee | trailing `echo` returned rc=0; `|| return 1` + fail-fast |
| Disk pre-flight missing | cd5ceee | phaseD gate 8G / phaseE gate 12G on /media02 (PAMR ckpts ≈1.3–1.5 GB each) |
| ranking per-rel lf NaN | e891aa7 | relation with zero lf ranks emitted nan→NaN JSON; lf fields now skipped for that relation (strict metrics unaffected) |
| MHA fast path (torch 2.5.1) | 515c3b1 | eval-mode padding-mask output wrong (max_abs 3.01 vs dense; 0.0 when disabled); global disable + parity gate |
| PAMR encoder collapse (v1) | K=10 + mask | dense v1 encoder constant (residual std 1.6e-6) because the K=1 pairs are 98.6% satisfied; fix = K=10 + pathless masking + path-dependence gate |
| Harness CUDA tolerance gaps | 1e-4 contract + per-rel checks honor tol | GPU fp-tie determinism documented |
| Known-swap CSVs (June) | untrusted | fresh regeneration authoritative |

---

## 6. Reference mapping — file → location index

(Anchors verified 2026-10-13/22; re-grep before use — line numbers
drift.)

### 6.1 Code + configs (local repo `src/`)

| Content | File | Anchor |
|---|---|---|
| Sampler (grouped-vectorized) + reference loop + cached dispatch + train loop | `pamr/helpers.py` | `negative_sampling_per_relation_fast` ~195–330; `evaluate_kge` dispatch ~290; `_evaluate_kge_per_triple` ~321; `_evaluate_kge_pooled` ~468 (scores_budget 16M; rows_budget 131,072; cand_chunk 8192; use_cached dispatch; pool encode once/relation); `train()` ~989 (resume 1019; heartbeat ~1120; keep-mask alignment ~1089–1106; save on val improve ~1213; final test eval ~1250) |
| Cached-eval API per scorer | `pamr/models/embedding/{transe,distmult,complex,rotate}.py` | `encode_entities` + `score_from_repr` at each class tail; RotatE replicates stack+vector_norm layout; ComplEx needs rel_real+rel_imag parts; TransE normalizes in encode; DistMult raw |
| Fusion / inductive buffers | `pamr/models/embedding/base.py` | `_fuse_embeddings_sparse` ~319+; `seen_in_train` buffer ~200 (num_entities) |
| Tensorized dataset + pools | `pamr/dataloader.py` | `KGDataset` ~28 (ID tensors + NaN guard); `_build_per_relation_pools` ~234 (sorted per-mode + all-pools sorted ~279); `_merge_pool_with_policy` ~258; `compute_filter_map` ~317 |
| Ranker (single source of truth) | `pamr/ranking.py` | `compute_rank_metrics`; candidate aggregation ~53–89; strict/ns ~175–188; lf composite ~193–232; F-9 gate `lf_recorded` ~244 |
| KGE training entry | `pamr/scripts/train_embedding.py` | REPO_ROOT 21; `--eval_batch_size` 147; scope modes ~353; loaders 394–417; scope pools ~427–436; checkpoint metadata ~493–498 |
| PAMR training entry | `pamr/scripts/train_pamr.py` | main 118; scope/registry gates 218–296; fine-tune gate 298–318; relation filter 345–357; dataset builds + `dataset_build_timings_s` ~359–400; optimizer 487; device guards 517/817; test scores writer ~918; metrics JSON ~962–988 |
| PAMR model | `pamr/models/pr/pamr.py` | freeze gate ~670; auto-detect dims ~681; no-path residual ~764–771; `_build_tokens` ~836 (2N+1, node types +1); `_build_rel_tokens` ~903; forward ~1011 (no-grad gate; has_path early return); compute_loss ~1100 (branch-local import-then-use is safe) |
| VL-KGE scoring | `pamr/models/vlkge/score.py` | modes; per-model `output_dir` (~101); checkpoint contract ~185–211; padding (~250–330); split loop ~317–344; CSV writers ~366–373 |
| Baseline metrics | `pamr/scripts/evaluate.py` | resolve ~47; prepare ref ~66; `scores_dir` ~219; metrics.json write ~340 |
| Leakage verifier | `pamr/scripts/verify_paths.py` | checks A ~183, B ~224, C ~294; main 373 |
| Query prep | `pamr/scripts/prepare.py` | FIELD_NAMES ~327 (query_id, head, relation, tail, head_id, relation_id, tail_id, is_positive, split); qid assignment ~437 |
| Path extractor | `pamr/models/pr/preprocess.py` | worker args 114; scope table 193; mask/flag 219; run dir 228; mappings ~344; graph writer ~409; probe writer ~457; merge/contract ~510–638; cleanup keeps probes ~649 |
| Registry | `pamr/relation_registry.py` | manifest ~130; hash ~126; inverse_id ~150 |
| Checkpoint utils | `pamr/utils.py` | save_checkpoint 301; load_checkpoint 336 (symmetric fields; scheduler optional) |
| PR configs (9) | `pamr/configs/pr/*.yaml` | `queries_dir: ../checkpoints/preproc/wikiart_mkg_v2/` (ROOT, canonical); pamr.yaml: bs 8, eval_batch_size 1024, lr 0.01, epochs 50, patience 5, min_eval_coverage 0.95, grouped_pairwise, lf_no_path_policy exclude, relation_filter train_eval; ablations single-diff + output-own dirs |
| Backbone configs | `pamr/configs/wikiart_mkg_v2/*.yaml` | batch_size 2048; lr 0.1; epochs 20; margin 12; downsample isRelatedToArtwork 0.001; num_neg 1; bidirectional_eval False |
| Eval chains | `pamr/configs/eval/` | neg_sampling K=1/10/10; score[,_complex,_distmult,_rotate].yaml with per-model `output_dir: …/full/vlkge/<Model>/`; neg_eval[_<model>].yaml with `scores_dir` |
| Aggregation | `scripts/aggregate_campaign.py` | `--mode fresh|june-selftest` (selftest = informational; June CSVs untrusted); grouped result.csv + headerless count.csv + average (macro/weighted, ddof=1) |
| Timing extractor | `scripts/phase_h_timings.py` | PAMR-family walls/GPU peaks/build timings |
| Census | `scripts/census_eval_pools.py` | val/test tail∈pool scan |
| Probes | `scripts/probe_batch_size.py` | sampler CUDA-vs-CPU, trainstep bs sweep, evalscore; `--num-neg`; bench_safe |
| Parity gates | `scripts/test_eval_kge_parity.py` | part0 synthetic anchors + 0b dispatch; part1 real slices (artist rows forced); CPU exact / CUDA 1e-4; `--models`, `--device` |
| Loader/sampler gate | `scripts/test_loader_sampler_equiv.py` | synthetic + real batch stability + sampler validity |
| Synth ckpt (throwaway) | `scripts/throwaway_synth_cpu_ckpt.py` | full 40-ID enumeration via manifest |
| Sbatch set | `slurm/` | `phase_b_regen`, `phase_c_backbone` (case BACKBONE ∈ transe/complex/distmult/rotate/struct; scratch /dev/shm; USR1+EXIT stage-sync; slim copyback; canonical-exists skip; --time=12h), `phase_c_gate`, `phase_d` (4 score chains → pamr-prod → clone PAR → evaluate ×4 → verify-C), `phase_e`, `verify_regen`, `equiv_check`, `parity_eval`, `cached_eval_gpu`, `netgate`, `census_probe`, `slimsweep`, `slimchk`, `rotgate`, `timingdry`, `selftest` |

### 6.2 Artifact provenance / evidence

| Fact | Artifact |
|---|---|
| June published backbone = ComplEx (d_model 1536) | June `pamr_training_metrics.json` (`vlkge_checkpoint` field; LOCAL mirror only — FIT copy was replaced by the fresh regen) |
| June old-test degeneracy (MRR 1.0, 235) | `full/pamr/20260629_071706/train.log` (local mirror) |
| June Table-1 sources | `full/vlkge/{Model}/{full,artist}/metrics.json` (local June mirror; untrusted numbers) |
| June PAR ≡ pathlen_last TEST | `full/pamr/pathlen/all/pathlen_last/metrics.json` (local June mirror) |
| BFS benchmark series | `src/assets/bfs.csv` (39 rows) |
| Paper | `src/assets/paperv7.tex` (ACTIVE); `assets/analysis/integral.md` §7 notation authority |
| Fresh wave evidence | `reports/backbone_<job>.out` (val curves, FINAL TEST diagnostic blocks, slims); `reports/phaseD_<job>.out`; `reports/slimsweep_<job>.out`; `reports/gpucache_<job>.out` |

### 6.3 FIT paths

| Topic | Path |
|---|---|
| Authority files | `~/AGENTS.md`; `/media02/hlhdang/README.md`; `~/admin_note.txt` |
| Env | `/media02/hlhdang/{tools/{uv/uv,uvenv_ws_env.sh},uvenvs/{t251cu121-py312,registry.json},.uv-cache}` |
| Workspace | `/media02/hlhdang/projects/{pyproject.toml,uv.lock,ejepwm,DreamZero-Project-IML,akg-sv,pamr}` |
| Campaign | `/media02/hlhdang/projects/pamr/{src,data,checkpoints,slurm,scripts,reports}` |
| Backbones | `checkpoints/vlkge/wikiart_mkg_v2/{transe,complex,distmult,rotate}_wikiart_mkg_v2_train.pt` + `transe_wikiart_mkg_v2_struct_train.pt` |
| Baseline scores | `checkpoints/preproc/wikiart_mkg_v2/full/vlkge/<Model>/` |

### 6.4 Rediscovery workflow
1. §0 snapshot → §1 decisions → §3 program.
2. Fact conflicts: artifact evidence > register prose; user decisions
   > both.
3. Re-grep anchors before use; ast/compile-gate every transform;
   selective git staging; commit + rsync after each milestone.
4. Two degeneration loops earlier in the larger session → on any loop:
   STOP, re-read the last real tool output, restart from the todo.

---

## 7. Operating rules and session-integrity lessons

- Login node = login/launch/inspect only; slurm for all compute
  (including CPU micro-tests).
- Every sbatch: explicit `--time`/`--mem`; self-diagnosing ACCEPTANCE
  lines; diagnose .out/.err before resubmitting; same failure twice →
  STOP + `reports/blocker_<name>.md` + escalate.
- Long jobs: USR1@300s + stage-sync + `--requeue` flag (note: the USR1
  path exits 0 cleanly — a wall-hit job needs manual RESUBMISSION to
  resume; the stage file survives).
- GPU jobs: `-N1` with `--nodelist`; avoid gpu03 under RAM pressure;
  canary-test after suspicious root-cancels.
- Pre-flight `df`; reclaimable scratch identified before deleting
  (cpu_smoke was regenerable; graph CSVs regenerable via Phase B).
- Never hand-edit the shared venv; extend override-dependencies for
  lock deadlocks; direct venv python at runtime.
- Commit locally at every milestone; rsync `memory/` + changed code
  to FIT.

---

## 8. Not touched (out of campaign scope)

- Legacy `configs/eval/vlkge_score.yaml` (/workspace paths),
  pathlen_* eval chains, wn9_img/wikiart_mkg_v1 configs, user's
  " 2.yaml" duplicates.
- `evaluate_zero_shot.py`, `evaluate_relatedto.py`,
  `evaluate_linear_probe.py` (lavis/legacy imports; not in the regen
  chain), `pamr/notebook.py`.
- `pamr_score.yaml` + `models/pr/score.py` (June-era alternate
  scorer; consumed but not part of the fresh chain).
- Camera-ready repo fate; paperv4/v5/v6 rounds (see `memory/wip.md`).
