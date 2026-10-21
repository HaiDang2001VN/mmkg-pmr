# memory/bugs.md — PAMR Defect Register + Campaign State

Why PAMR does not outperform VL-KGE despite consuming frozen VL-KGE
embeddings as input. Triage + the post-submission experimental campaign
(SOICT 2026 submitted 2026-09-20, paperv7.tex; notification 2026-10-12).

Context files: `pamr/models/pr/pamr.py`, `pamr/scripts/train_pamr.py`,
`pamr/scripts/train_embedding.py`, `pamr/helpers.py`, `pamr/dataloader.py`,
`pamr/scripts/prepare.py`, `pamr/scripts/verify_paths.py`,
`pamr/models/vlkge/score.py`, `pamr/models/pr/preprocess{.cpp,.py}`,
`pamr/configs/**`, artifact evidence in
`checkpoints/preproc/wikiart_mkg_v2/full/pamr/`.

---

## 0. READ-FIRST — state snapshot (2026-10-06, compaction entry point)

### COMPLETE and verified
- **Phase A** (FIT staging): tree + data cksum/md5-verified on
  `/media02/hlhdang/projects/pamr/`; pyproject floors; preprocess.o
  compiled (g++ 11.4, `-std=c++23 -O3 -pthread`).
- **Phase B** (CPU regen) + **pandas-3 revalidation**: job 78118 (9m44s,
  pandas 2.3.3) and job 78287 (23m54s, pandas 3.0.6 + torch 2.5.1) both
  ALL GREEN: registry PASS (22 raw / 40 total, hash `ff50ee5aabe9ba5e…`),
  prepare 709/202/239 queries (K=1/10/10, 0 skips), BFS prod
  1,417/2,214/2,606 found, BFS h3 1,411/178/213, flatten to
  `full/pamr` + `_h3/pamr`, **verify_paths A/B PASSED on all 3 splits for
  BOTH configs**. The pandas-3 artifacts are canonical for Phase C.
- **Phase C0** (machinery smoke, CPU): job 78141 FULLY GREEN — synth ckpt
  (via get_model, feature-aligned) → score.py → evaluate.py → train_pamr
  1 epoch on real paths (~4 min).
- **Contract migration** (uv WORKSPACE era, FINAL): one shared venv
  `uvenvs/t251cu121-py312` (py3.12.14, torch 2.5.1+cu121 trio, numpy
  2.5.3, pandas 3.0.6, sklearn 1.9.1); workspace members [ejepwm,
  DreamZero-Project-IML, akg-sv, **pamr/src**]; THE union lock
  `projects/uv.lock` (509 pkgs); pamr 1.0.0 import battery OK; registry
  noted. Runtime = direct `…/t251cu121-py312/bin/python -m …`.

### RUNNING (wave relaunch, 2026-10-06 ~21:20 +7)
- **78331** = transe backbone (relaunch, gpu01/02/04 constraint),
  **78332** = complex backbone (relaunch), **78333** = batch-size probe
  (fixed script). All PENDING at submit (78298 draining + 4-GPU quota).
- 78298 killed after 4h13m with ZERO python output: 99.4% single-core
  python, buffered stdout (pre-unbuffered sbatch) hiding progress;
  process = the legacy per-triple val-eval grind (each of ~208K val
  triples rebuilds a ~216K-candidate sorted list in Python). No stage
  file existed (epoch 1 unfinished) → clean kill, no loss.

### BLOCKED → NOW UNBLOCKED (verify before use)
- **Admin GPU-cancel block LIFTED** (18:13 +7): canary 78314 COMPLETED
  normally; the other agent's `wstack` job (78316, gpu03) also runs.
  Earlier (~17:00-17:50 +7) ALL new GPU submissions were insta-cancelled
  by root (3 probes + 1 canary, "CANCELLED by 0"); watch UI showed a
  "Gia hạn thời gian sử dụng (Renew Lease)" banner. Re-verify with a
  canary if anything looks off again.
- **Disk recovered**: /media02 free went 7.5 GB → 2.2 GB (crisis) → 166 GB
  (communal cleanup). Pre-flight `df -h /media02` before any >5 GB write
  remains mandatory.

### Probe 78333 results (rc=0, gpu01, 2026-10-06) — user batch-size decision pending
- trainstep per-triple wall: bs512=305 µs, bs2048=147 µs (2.1×),
  bs8192=119 µs (2.6×) — diminishing returns past 2048.
- GPU util NEVER exceeds 9.1% at ANY batch size (bs512 9.1%,
  bs2048 5.2%, bs8192 2.8%) — the backbone loop is HOST-bound
  (KGDataset .iloc loader + python sampler) at every size; batch size
  alone cannot raise GPU utilization, only cut per-triple wall.
- evalscore per-row flat (~0.08 ms/row, bs512≈bs4096) — loader-side.
- Sampler share: cuda-inputs vs cpu-inputs = 95 vs 80 ms/batch.
- Wall models (transe, 9.0M triples/epoch, 20 epochs): bs512 ≈ 15.3 h
  train+ 2 requeues; bs2048 ≈ 7.3 h; bs8192 ≈ 6 h.

### Host-path engineering round (2026-10-06, user: 512 now + engineer, then combined with bs 2048)
- DECISION EVOLUTION: user first chose "keep 512 + engineer host path",
  then combined it: bs 2048 adopted for ALL five backbone configs
  (transe/complex/distmult/rotate/struct; June was 512/1024; lr 0.1
  UNCHANGED — dynamics caveat registered).
- **KGDataset tensorization**: ID triple built once as int64 tensors;
  __getitem__ = tensor indexing (was per-row pandas .iloc on a
  15.7M-row frame). Batch content bit-identical (shuffle generator
  external; no RNG inside the dataset); NaN guard raises loudly.
- **negative_sampling_per_relation_fast grouped-vectorized**: one
  pre-batched randint per relation group covers the 16-attempt budget;
  per-row rejection walks plain python ints. Uniform-over-eligible
  distribution preserved; hub-drop + 4-tuple keep-mask unchanged; RNG
  stream shift = distribution-faithful (June stream already retired
  with the hub-drop fix).
- **EQUIVALENCE GATE 78344 ALL OK** (rc=0, 2m54s): A/synthetic +
  A/nan-guard + A/stability + B/validity (kept=3,072, dropped=0,
  pool-membership + no-truth clean).
- Probe-based wall model at 2048 + engineering: transe 20 epochs ≈
  6-9 h single wall (vs ~15 h at 512 unengineered).
- Wave relaunched: 78347 (transe) + 78348 (complex) at bs 2048.

### INCIDENT 2026-10-07/08: /media02/hlhdang WIPED (hard blocker, needs user)
- Evidence chain: 78351 (transe) FAILED at 00:45:02 (~16:02 UTC Oct 6,
  during its first pooled eval — output lost with the FS); ssh now
  lands on login02 (round-robin) and /media02/hlhdang is ABSENT on
  login01-ish view, login02 AND compute nodes (gpu04 probe job 78437/
  78438 COMPLETED — account still executes); /media02 root mtime
  Oct 7 11:24 = reorganization event; no hlhdang entry anywhere incl.
  /media02/recovery; `mkdir /media02/hlhdang` = Permission denied
  (parent root-owned) → deliberate admin action, not NFS rot.
- Context: the watch UI earlier showed a "Gia hạn thời gian sử dụng
  (Renew Lease)" banner → consistent with an expired account lease
  and admin storage cleanup. admin_note.txt unchanged (still points
  at /media02/hlhdang). ~/AGENTS.md (home) intact but pre-dates it.
- LOSS ASSESSMENT: nothing irreplaceable — every artifact was
  regenerable; all code/registers/configs/sbatch live in the LOCAL
  git (through e9f1dfd); data 3.4 GB intact on the workstation;
  no slim backbone checkpoint had landed (best in-flight = 78351
  epoch-1, unfinished); Phase B artifacts regenerate in ~25 min.
- SURVIVES: account can run jobs (both probes COMPLETED today);
  /home/hlhdang intact (~18 GB free — policy says dotfiles only).
- UNBLOCK PATH: user renews the lease (watch UI banner flow) or
  contacts the admin to restore /media02/hlhdang ownership; then
  restage from the workstation (rsync tree+data ~3.5 GB) + re-run
  Phase B + gate + waves — fully scripted, ~1 h to full redelivery.

### INTEGRITY CHECK after admin restore (2026-10-08, user-announced) — PASS
- /media02/hlhdang restored as of an Oct 6 14:46 UTC snapshot: ownership
  hlhdang + writable (write-test passed, test file removed).
- Verified present: pamr tree {src,data,checkpoints,slurm,scripts,
  reports}; ENGINEERED src (pooled eval + rows_budget + enumerate fix +
  bs-2048 configs); data 1.7 GB (triples + features); FULL preproc
  artifact set for both full/pamr AND _h3/pamr (paths CSVs + metas +
  graph metas + mappings + manifest + probe queries) + prepare queries
  + metadata.json; env system (tools/uv exec + uvenv_ws_env.sh +
  uvenvs/t251cu121-py312 with torch 2.5.1+cu121 + pandas 3.0.6 +
  registry.json with pamr row); workspace root with pamr/src member +
  cu121 sources + uv.lock (8,738 lines); reports/ 84 files.
- Known deltas to fix during recovery (NOT yet done, per user request):
  (1) FIT bugs.md lacks the incident entries (local-only commits
  bd7769d/dbb903c + later); (2) stale src/uv.lock at FIT src root (the
  relocated one removed locally in a3ff67f); (3) stale
  projects/pamr/src-memory-and-assets/ dir (rsync-mishap artifact).
- Lost with the snapshot boundary: 78351's post-14:46 output (the job
  died at 16:02 UTC; regenerable).

### NetGate W1 round (2026-10-10, user: probe internet; W1 RELAUNCH DEFERRED)
- **NetGate 78622 (gpu04, 1 s): COMPUTE NODES HAVE INTERNET** —
  huggingface.co 200, api.wandb.ai reachable (404 on root), pypi.org
  200, DNS fine, no proxy env. The documented "compute nodes
  UNVERIFIED/never download" is policy-conservative, not a network
  constraint. Implications recorded: HF streaming would work (still
  unnecessary — local-file pipeline); wandb online would work (user
  chose NO wandb — decision stands); staging-everything remains the
  robustness rule for training jobs.
- **W1 v2 outcome before deferral**: 78597 (transe) FAILED rc=1 in the
  val eval — the pooled evaluator's loud guard FIRED
  ("Eval tail absent from its relation candidate pool (relation 12,
  tail 220397)") — its supersets premise is empirically false for at
  least one real val row (parity slice missed it). The legacy
  tolerated this case (candidates = pool - truths + {t}); the batched
  shared-pool path cannot. 78598 (complex) hit the same path;
  CANCELLED at 00:43:34 per prior approval (nothing banked — ckpt
  save happens after eval, which never completed).
- FIX DESIGN (deferred with the relaunch): (1) census probe — full
  val/test scan for tail∉pool counts per relation + anatomy of
  (rel 12, t 220397) incl. which pool-policy branch applies; (2) union
  fix — tail-∉-pool rows route through the exact legacy per-triple
  branch, rest stays batched; synthetic t-outside-pool parity case;
  (3) CPU re-gate; (4) relaunch W1 on healthy nodes.
- Also this round: Phase B REGEN re-run COMPLETE ALL GREEN (78592,
  13m58s) — the restored snapshot lacked graph CSVs only because our
  own pre-incident reclaim deleted them; regen restored the full
  contract set (registry hash ff50ee5aabe9ba5e..., verify A/B PASS all
  splits both configs, fresh 2026-10-10 lineage).

### Phase-H scaffolding round (2026-10-11, while W1 waits on fairshare queue)
- W1 (78651 transe + 78652 complex) still PENDING: account fairshare
  decayed after the heavy week; scheduler projected start 2026-10-12
  17:28 with opportunistic backfill on short-job gaps. Cluster shows
  240 idle CPUs but GPU nodes busy (gpu01 6/8; gpu02 4/4).
- The user chose to build Phase H meanwhile:
  - train_pamr.py: paired/probe dataset-build TIMING instrumented
    (Per split wall in build, surfaced in
    pamr_training_metrics.json as dataset_build_timings_s) — protocol
    neutral, lands with Phase D runs.
  - aggregate_campaign.py: June-format faithful fresh aggregation
    (grouped result.csv + count weights + average_macro math verbatim);
    extractors are split-aware (test|val) for the evaluate.py AND
    train_pamr metric schemas; June PAR row source = pathlen_last
    metrics (evaluate-schema). `june-selftest` mode is INFORMATIONAL —
    June result/average CSVs are UNTRUSTED (registered H1/H3 swap +
    different split/source branches); the selftest validated the
    schema on the restored June artifacts (78668/78669 rc=0, rows=5).
    Fresh weights = test queries (87/85/33/34 = 239, the full test set
    is artist-relations only by the train_eval filter).
  - phase_h_timings.py: PAMR-family timing extractor (train/test
    walls, GPU peaks, dataset-build timings) — dry-validated on the
    June metrics (78670 rc=0, 7 rows).
- Also this round: FIT ~/AGENTS.md network row corrected (NetGate
  78622 evidence, backup AGENTS.md.pre-netgate.bak); wip.md §20 added.
- Aggregation protocol implications REGISTERED: fresh tables = TEST
  split (paper default) with test-matched weights + a VAL sheet for
  reference; June 74/76/26/26 = VAL query counts (202) explained.

### Cached-eval gate round (2026-10-12/13)
- **ALL FOUR scorer families parity-EXACT under the cached evaluator**
  (gate 78779 transe 15x / distmult 16-18x / complex 9.6x; solo gate
  78896 rotate 8.6-9.6x after the fix below). Artist-rel rows forced
  into every slice; per-relation metrics exact vs the per-triple
  reference loop.
- **RotatE ULP defect (78896 fix)**: my sqrt(sum(re2)+sum(im2))
  reordered the reduction vs forward's stack+vector_norm((1,2)) —
  last-bit score noise flipped rank TIES (MRR diffs ~1e-8). Fix =
  replicate the exact stack layout ((b,w,f,2), norm dims (2,3)) —
  sum order now bit-matches the forward for every backend shape.
- Running transe/complex (78651/78652) hit the 12h wall with clean
  USR1 stage-syncs (stage_transe 2.75 GB ~epoch 7 Val MRR 0.2471;
  stage_complex 2.8 GB ~epoch 2, loss 0.021) — resumes requeued as
  78885/78886 WITH the cached code (remaining epochs run the fast
  path; earlier epochs' results stay valid).
- Queue state: 78885/78886 (+ the other agent's jobs) pending on
  fairshare priority; backfill continues to be the start mechanism.

### NEXT ACTIONS (in order)
1. **78321 parity GATE PASSED** (`reports/parity_78321.out`): Part 0 OK
   + Part 1 real slices OK (val 23 rels, sampled-train 16 rels,
   per-relation exact, overall ≤1e-9) → **vectorization ACCEPTED**.
   CPU cost-par note: pooled ≈ ref on CPU (~282 vs ~304 s) — the win is
   GPU-side (per-triple CUDA syncs removed), as measured on 78297.
2. **Watch 78331/78332/78333 start**; diagnose with the srun /proc
   toolkit if any job shows zero-CPU/no-output growth. First-epoch
   stage files should appear within the hour under the vectorized eval.
3. **Collect 78333 probe numbers** → user decides training batch size.
4. **Validate slim-copyback contracts** as each backbone finishes
   (no optimizer keys, registry manifest, embedding_scope=train).
5. Waves B (distmult+rotate) + C (struct) → Phase D (fixed chain) →
   Phase E/G → aggregation (§3 below).

---

## 1. Locked experiment plan (8 interview decisions — unchanged)

1. **Compute** = FIT cluster (Slurm). GPU fleet: gpu01 = 8×A100-40GB,
   gpu02-04 = 4×GPU each (A100 40GB after the live upgrades; gpu02
   driver 555.42.02, gpu03 driver 580.82.07 — RECHECK DRIVERS each
   session, the admin is upgrading nodes live).
2. **Regen matrix = FULL**: retrain TransE (backbone + baseline),
   ComplEx, DistMult, RotatE baselines + structural-only TransE (PAR
   backbone); retrain PAMR + PAR; rescore all on the prepared query CSVs.
3. **Backbone scope = train everywhere** (one train-scope backbone backs
   val + test evaluation + baseline scoring; no matched-scope pairing).
4. **Protocol = 10 negatives × 2 seeds** (PAMR seed 42 + seed 1, n=2
   std ddof=1; baselines single-seed 42).
5. **Ablations = core 5**: fine-tuned backbone, K=3 (fresh h3 paths —
   already extracted), depth N=12, readout cls, B3 contrastive =
   listwise_softmax + lf_temperature 0.1.
6. **rclone/GDrive auth superseded** — FIT staging (rsync) only.
7. **Camera-ready = correct the numbers** if accepted.
8. **Backbone = TransE-faithful** (pamr.yaml → retrained train-scope
   TransE `transe_wikiart_mkg_v2_train.pt`). STANDING CAVEAT: June
   artifacts show the published numbers came from a frozen COMPLEX
   checkpoint (d_model=1536; `vlkge_checkpoint` field in
   `pamr_training_metrics.json`); the regeneration is a NEW
   configuration — deltas have two sources (fixed pipeline + documented
   backbone). Ignore old result/average/count CSV values.

---

## 2. Phase record (chronological, condensed)

### Phase 0 + audit + fix rounds (2026-10-03 → 10-04)
- Provenance audit from synced artifacts; verify_paths.py runner (checks
  A scope-hygiene / B own-edge+completeness / C candidate-identity);
  rename sweep `pmr`→`pamr` (47 refs, 14 modules); fine_tune +
  lf_temperature wired end-to-end; smoke chains repointed to TransE.
- Audit found 2 critical defects, fixed (commit d4c2d23): train_pamr
  missing `def main():` (body unreachable after resolve_config return);
  verify_paths registry-hash comparison nested-vs-top-level mismatch.
  + inverse-id hardening; requirements floors; smoke repoints; fixture
  mirror; registers synced (ba7c335 §5 mapping; 003b5f6 AGENTS.md
  rewrite; 1510410 comment sweep).

### Phase A (2026-10-04)
- A1/A2 staging verified (md5/cksum). A3/A4: env binding — originally
  minted `t260cu126-py312-pamr`-plan, SUPERSEDED same-day by the FIT
  unification (one universal venv) then by the FINAL workspace; pyproject
  rewritten (floors-only, requires-python >=3.12,<3.13). A5 compiled.
  A6 probe: **cu130 torch GPU-dead fleet-wide** (drivers 525/535, CUDA
  12.2-era; RuntimeError "driver too old (found version 12020)") — the
  unification agent had import-verified only, never GPU-verified
  (evidence: reports/finalize_py312.log). Escalated; user delegated env
  resolution to the other agent.

### Phase B + pandas-3 revalidation (2026-10-04 → 10-06)
- Defects found BY EXECUTION (the CPU-first strategy worked):
  - c1c68e6: `../checkpoints` resolved against WS_ROOT (repo parent) in
    preprocess.py/score.py + prepare cwd-dependence → silent split skips
    → all REPO_ROOT-relative now + loud failures.
  - cf4d0dd: keep_temp=false deleted the row-identity probe queries
    (verify B first-class artifacts) → preserved unconditionally; chain
    flattens + verifies on flattened dirs.
  - a7f410a: thread/worker etiquette — prepare pool default
    cpu_count()=256 inside 12-CPU cgroup (observed live) → cap 10;
    pamr.py builder pool cap 10; BFS threads 8→10 (2 reserve); OMP/MKL=10.
  - revalidation job 78287 under the FINAL env: ALL GREEN (counts in §0).
  - pandas-3 drift note: negative identities differ between lineage runs
    (pool ordering changes rejection draws); positives stable; the
    verify contract checks hold — fresh lineage is authoritative.
- Graph CSVs deleted after evidence archived
  (reports/phaseB_revalidation_evidence.txt) — regenerable in ~4 min.

### Phase C0 machinery smoke (2026-10-05, job 78141)
- FULLY GREEN on CPU: synth ckpt → score.py (3 splits, strict contract)
  → evaluate.py (metrics.json) → train_pamr 1 epoch (709 pairs → 9 group
  batches; val 2,222 / test 2,629 probes; d_model=768 auto-detect;
  artifacts under cpu_smoke/ scratch; prod untouched).
- Audit fixes folded (ad2fe02): train_pamr device-guards for CUDA metric
  calls; output-own paths_dirs (PAR → full/par; ablations →
  full/ablations/{finetune,depth12,cls,lf_temp,seed1}/; hop3 keeps
  _h3/pamr) preventing artifact clobber; smoke proxy configs absolute
  vlkge_config paths.
- Turnkey sbatch set authored + FIT-staged + sbatch --test-only
  validated: phase_c_gate (torch acceptance), phase_c_backbone (per-model
  export, scratch-save + USR1/EXIT stage-sync + slim copyback,
  --time=12:00:00 vs DefaultTime-6h hazard), phase_d (gated chain),
  phase_e (ablations, clone-before-run).

### Contract migration to the FINAL workspace era (2026-10-05, 90a5dda)
- The env system finalized by the other agent: uv WORKSPACE (root
  `projects/pyproject.toml`, members ejepwm/DZ/akg-sv + pamr/src), union
  lock 509 pkgs, shared venv `t251cu121-py312`, torch 2.5.1+cu121 (the
  ONLY runnable family on driver 535; cu126/128/130 verified failing
  on-node, jobs 78104/78105; family gate map: 535→cu121, 550→cu124,
  560→cu126, 570→cu128, 580→cu130/13.x — family adoptions require an
  on-node smi+matmul proof job first, evidence job 78282).
- Pamr joined as 4th member: `uv lock` (Added pamr v1.0.0) +
  `uv sync --locked --no-dev --all-packages` (mostly no-op, 335 checked)
  + registry note (used_by += pamr). Import battery OK.
- Binding trap: bare `uv run --project` spawns a forbidden repo-local
  `.venv` (deleted once); with the workspace contract
  `uvenv_ws_env.sh` exports UV_PROJECT_ENVIRONMENT → shared venv used;
  runtime is direct venv python (README-blessed).
- Transient disk squeeze (2.2 GB free) self-recovered communally.

### Wave A + GPU-util round (2026-10-06)
- Wave A v1 (78291/78292) FAILED rc=1: sampler fail-fast "Every
  candidate tail for head 222220 under relation 29 is a known positive".
  Grounding: relation 29 = internal inverse_of 7 (isCreatedByArtist
  family); hub triples whose pool is fully covered by known truths.
- Fix ef005be: whole-triple drop contract in
  negative_sampling_per_relation_fast (4-tuple return + keep_rows mask;
  coherent for num_neg_samples>1; train loop aligns pos_scores via the
  mask; fully-hub batches skipped; avg_loss over loss-bearing batches;
  the banned corruption class stays banned; filters-vs-pool disagreement
  keeps its own fail-fast). Micro-tests 3/3 on FIT.
- Wave A relaunched (78297 transe@gpu02 + 78298 complex@gpu03, 16:12 +7).
- GPU-util diagnosis (user directive "probe best batch size to maximize
  GPU use"): trainer host-bound. Evidence chain: ps → 99.7% single-core
  CPU; pmon → 44-46% SM sustained, 9.1 GiB/40; ctx-switch sampling →
  zero voluntary switches in 5 s; SIGINT traceback (killed 78297 at
  1:50:32, my diagnostic sacrifice) → the process was INSIDE
  `evaluate_kge` (helpers.py:361) doing per-triple `.item()` CUDA syncs
  over the 208K-triple val eval. CONCLUSION (correcting an earlier
  "hung" hypothesis): NOT hung — the per-epoch eval is the wall hog
  (per-triple syncs × 208K, amplified by driver contention under
  multi-tenant nodes); the earlier zero-switch reading = tight
  ioctl+spin loop, not a hang. Fixes landed (d58e57f): sampler inputs
  get ONE .cpu() per batch (semantics + RNG stream identical, parity
  micro-tested), --eval_batch_size (default 4096) for val/test/sampled
  loaders (protocol-neutral). The per-triple .item() INSIDE evaluate_kge
  remains — vectorized-eval rewrite proposed (needs parity tests).
- PYTHONUNBUFFERED=1 added (be433e1) — running jobs predate it (their
  logs flush at 8 KB boundaries only).
- Batch-size probe authored (520a0aa) — blocked by the admin block
  (3 cancels + canary), which LIFTED at 18:13 (canary 78314 COMPLETED).

### Vectorization + Phase-D pre-fix round (2026-10-06, decisions: vectorize first + probe-then-decide)
- **Phase D baseline-scoring clobber defect found by careful re-read**
  (would only bite at Phase D runtime): all four score configs had NO
  `output_dir` → score.py falls back to the prepare root → four baseline
  chains overwrite one shared `{split}_scores.csv`; Phase D's single
  evaluate step would rank only the LAST chain (RotatE) and verify-C
  would compare PAMR against RotatE instead of TransE. Fixed: per-model
  `output_dir …/full/vlkge/<Model>/` (June layout) in the 4 score
  configs + 4 new `neg_eval_{transe,complex,distmult,rotate}.yaml`
  (+`scores_dir`) + phase_d.sbatch runs evaluate ×4 and verify-C reads
  `full/vlkge/TransE/test_scores.csv`. `neg_eval.yaml` retained unused.
- **evaluate_kge vectorized** (81d9b31): dispatch (not bidirectional +
  pools) → `_evaluate_kge_pooled` (chunked per-relation forwards.
  Truth-mask −inf + true-tail unmask + stable argsort = candidate index
  space identical to the reference; forwards flattened to 1D because
  the fusion/inductive paths are 1D-only). Reference loop preserved
  verbatim as `_evaluate_kge_per_triple`. Parity harness
  `scripts/test_eval_kge_parity.py` (Part 0 synthetic anchors: ranks
  1,7,1,1,4 → rel0 5/7, overall 19/28, H@1 3/5 — first-run anchor
  arithmetic was botched, implementations agreed throughout; Part 1
  real slices val 400 + sampled-train 200, per-relation exact).
  Parity gate = slurm job 78321 (CPU-only).
- **AGENTS.md refreshed to the workspace era** (env block, runtime
  contract, torch-family gotcha, §6.4 pointer, slurm-first commands).
- Login-node etiquette locked by user: ALL compute — including CPU
  micro-tests — runs as slurm jobs; login node serves login/launch only.

---

## 3. Remaining program (Phases C→H)

1. **78298 completion check** → slim ckpt landing validation.
2. **Probe** (unblocked): `sbatch slurm/phase_probe_bs.sbatch` → numbers.
3. **User decisions**: training batch size (probe data vs locked 512);
   evaluate_kge vectorization (before or after relaunch).
4. **Phase C waves** (2 jobs × 1 GPU): A' = transe (relaunch) + complex
   (78298, running); B = distmult + rotate; C = struct.
   Every backbone: `sbatch --export=BACKBONE=<name> slurm/phase_c_backbone.sbatch`
   (BACKBONE ∈ transe|complex|distmult|rotate|struct). Scratch-save +
   USR1/EXIT stage-sync + slim copyback (no optimizer; consumers need
   model_state + metadata). --time=12:00:00 explicit. Post-wave gates:
   consumer contract checks (relation count 40 / registry hash /
   embedding_scope=train).
5. **Phase D** (one GPU job): `sbatch slurm/phase_d.sbatch` — gate on 5
   ckpts → score.py ×4 (score chains into output-own
   `full/vlkge/<Model>/` dirs; TransE chain doubles as the
   TransE-baseline row) → PAMR prod (pamr.yaml) → clone+PAR
   (pamr_par.yaml) → evaluate.py ×4 (per-model `neg_eval_*.yaml` →
   `full/vlkge/<Model>/metrics.json`) → verify_paths check C (identity,
   test split, PAMR scores vs `full/vlkge/TransE/test_scores.csv`,
   prepared-queries cross-check).
6. **Phase E/G** (one GPU job): `sbatch slurm/phase_e.sbatch` — seed1 +
   finetune + depth12 + cls + lf_temp (clone-before-run) + hop3 (on
   _h3/pamr paths). n=2 std (ddof=1) for PAMR rows.
7. **Phase H** piggyback: BFS timings done (Phase B logs); paired-build/
   epoch/inference timings from log_metrics fields.
8. **Aggregation + paper**: per-relation metrics → fresh
   result.csv/average.csv (grouped-header format) → delta vs paperv7 →
   camera-ready refresh if accepted (notification 2026-10-12).
9. **PAMR-family scoring caveat**: baseline score chains score ONLY the
    prepared target-relation queries into per-model dirs
    (`full/vlkge/<Model>/`); verify C reads PAMR `test_scores.csv`
    (paths_dir) vs the TransE chain's `test_scores.csv`
    (`full/vlkge/TransE/`).

---

## 4. FIT cluster handoff (FINAL state — verified via SSH 2026-10-03→06)

### Access + rules
- `ssh FIT` = 172.29.74.81, user `hlhdang`, key `id_rsa_fit` (~/.ssh/config).
- Reading order: `~/AGENTS.md` → `/media02/hlhdang/README.md` →
  `~/admin_note.txt` (store everything under /media02/hlhdang).

### Cluster facts (verified 2026-10-03; nodes upgraded live since)
| Item | Value |
|---|---|
| Scheduler | Slurm, single partition `batch` |
| Nodes | gpu01 = 8×GPU, 980 GB RAM; gpu02-04 = 4×GPU, ~240 GB RAM; 256 threads |
| Time | MaxTime 2-00:00:00; DefaultTime 6 h → ALWAYS explicit `--time` for long jobs |
| Memory | DefMemPerCPU=2 GB → ALWAYS explicit `--mem` |
| Per-job CPUs | account MaxCPUsPerJob = **12** |
| GPU quota | QOS gpu_limit = 4 GPUs/user (RUNNING+PENDING both count) |
| Scratch | TmpDisk=0 → /dev/shm (~500 GB gpu01) or /tmp; trap … EXIT self-clean |
| Storage | /media02 5 TB shared (free fluctuates 2-166 GB — PRE-FLIGHT df) |
| Network | login OK; compute nodes OFFLINE — stage everything on login |
| Watch UI | http://172.29.74.82:8012/ (lease-renew banner seen) |

### Environment system (FINAL — uv workspace)
- ONE shared venv: `uvenvs/t251cu121-py312` (py3.12.14, torch
  2.5.1+cu121 / tv 0.20.1 / ta 2.5.1, numpy 2.5.3, pandas 3.0.6,
  transformers 5.18.0, sklearn 1.9.1, stable-worldmodel 0.0.6,
  stable-pretraining 0.1.6 --no-deps post-sync injection).
- Workspace root `projects/pyproject.toml` (virtual; members ejepwm,
  DreamZero-Project-IML, akg-sv, pamr/src); THE lock =
  `projects/uv.lock` (509 pkgs); per-member locks archived *.pre-ws.bak.
- Pin policy: ONLY torch triad + py3.12 (+ swm 0.0.6) pinned; everything
  else floats; resolver deadlocks → extend root
  `[tool.uv] override-dependencies` (numpy>=2.2.6,<3, transformers>=4.50,
  <6, tokenizers>=0.21, datasets>=2.19) — never hand-drop pins.
- Runtime: `source /media02/hlhdang/tools/uvenv_ws_env.sh` (exports
  UV_PROJECT_ENVIRONMENT + cache paths); run via direct
  `uvenvs/t251cu121-py312/bin/python -m …` (README-blessed; avoids
  uv-run project resolution).
- Re-lock/sync: `cd /media02/hlhdang/projects && uv lock && uv sync
  --locked --no-dev --all-packages` (uv pinned at tools/uv/uv v0.12.19).
- NO new venvs/conda/pip-venv — ever; family adoptions need on-node
  proof jobs (gate map: 535→cu121 … 580→cu130; proof job 78282).
- Budget: env+cache ≤ 60 GB; <25 GB free stop for env work; >5 GB writes
  need pre-flight df + registry note.

### Campaign layout on FIT
- Tree: `/media02/hlhdang/projects/pamr/{src,data,checkpoints,slurm,scripts,reports}`
- Data: `projects/pamr/data/wikiart_mkg_v2/` (triples 1.04 GB + vf_clip
  653 MB + tf_clip 23 MB + README).
- Artifacts: `projects/pamr/checkpoints/preproc/wikiart_mkg_v2/{full/pamr,_h3/pamr}`
  (paths CSVs + probe-identity queries + graph metas + mappings +
  manifest) + `{train,val,test}_queries.csv` + metadata.json at
  preproc root; `cpu_smoke/` scratch subtree.
- Checkpoints (target): `projects/pamr/checkpoints/vlkge/wikiart_mkg_v2/<model>_wikiart_mkg_v2[_struct]_train.pt`
  (slim: model_state + metadata; stage_ files = in-flight full ckpts).
- Reports: `projects/pamr/reports/` (job .out/.err, evidence files).
- Local mirror: workstation repo `src/` rsynced both ways after changes.

### Etiquette (every sbatch)
- Diagnose reports/<job>.out/.err BEFORE resubmitting; same failure twice
  → STOP + write reports/blocker_<name>.md + escalate.
- Login node = login/launch/inspection ONLY — zero compute there,
  including CPU micro-tests (slurm CPU-only jobs instead; no --gres,
  `--cpus-per-task=12 --mem` sized, short --time).
- Self-diagnosing ACCEPTANCE lines; wall ≤ 45 h; pending GPU pressure ≤
  4-GPU quota; batch verification jobs; cancel stale PENDING.
- srun --overlap for in-job inspection; -N1 required with --nodelist
  (missing -N1 = "Node count specification invalid"); watch node RAM
  (sinfo %e) — gpu03 was at 6.8 GB free once, 3.2 GB under two jobs.

---

## 5. Defect register (consolidated; all RESOLVED unless noted)

- **train_pamr missing main()** (d4c2d23) — body unreachable; verified
  ast-clean after fix.
- **verify_paths nested-hash mismatch** (d4c2d23) + inverse-id hardening.
- **Workspace-relative dir collisions** (c1c68e6) — preprocess WS_ROOT
  bug (silent skips), score checkpoint_path, prepare cwd-dependence;
  loud failure on missing prepared queries.
- **Probe-queries artifact deletion** (cf4d0dd) — kept unconditionally.
- **CUDA metric guards** (ad2fe02) — reset_peak guarded by device.
- **Output-own paths_dirs** (ad2fe02) — clobber prevention for PAR + 5
  ablations; clone-before-run in chains.
- **Hub-relation sampler crash** (ef005be) — bounded whole-triple drop
  + keep mask + train-loop alignment; micro-tested 3/3.
- **Per-triple CUDA syncs in sampler** (d58e57f) — one .cpu() per batch;
  parity micro-tested. evaluate_kge per-triple .item() superseded by
  the pooled vectorization (below, parity-gated).
- **Probe bench crash + zombie park** (fc77e6d) — probe step() called
  compute_logistic_loss with 1-D neg scores (real loop reshapes to
  (n_kept, num_neg) + aligns keep-rows) → RuntimeError; the non-daemon
  util-poller then blocked interpreter shutdown → job looked RUNNING
  ~90 min with zero progress. Hardened: mirror-loop semantics,
  --num-neg arg, daemon poller + stop.set(), bench_safe FAILED lines,
  peak-mem reset per point.
- **Pooled-eval GPU OOM (first real GPU exercise)** (afc79eb) — chunk
  bound was score-matrix elements (16M) but the fused multimodal
  forward materializes ~a dozen (rows×768) fp32 tensors/call →
  b×w = 229×8192 ≈ 1.9M rows = ~33 GB transient on A100-40; OOM at
  `result + structural` (5.77 GiB alloc) after epoch-1 train finished
  cleanly (41 min). Budget is now forward ROWS 131,072 (~6-8 GB
  transient) + expandable_segments in the backbone sbatch. Lesson:
  chunk bounds must be designed for the forward's own materialization,
  not only the score tensor. helpers.train also gained a 10% heartbeat
  print (large epochs were otherwise silent — that cost a misdiagnosis
  and one wasted SIGINT).
- **78298 legacy-eval grind** (killed, vectorize-first policy) — 4h13m
  at 99.4% single-core in the per-triple val eval; stdout fully
  buffered (pre-unbuffered sbatch era). Lesson: zero output + high
  single-core CPU on an OLD-code job = legacy eval; check the sbatch
  env era before assuming IO stall. State-class signatures: D+rpc =
  NFS stall (may self-clear); Sl + futex only = parked threads; R +
  99% single core = python grind.
- **Thread/worker etiquette** (a7f410a) — pool caps, BFS 10 threads,
  OMP/MKL caps, PYTHONUNBUFFERED (be433e1).
- **Stale-name hygiene** — evaluate.py usage line, path comments,
  configs cpp_source/binary_path pmr→pamr (2989e77).
- **requirements floors** (d4c2d23) — torch>=2.6 (now superseded by the
  workspace-era torch 2.5.1+cu121 pin; the CUDA-13-driver CUBLAS bug
  family does not apply to driver 535/555; dormant mean-pool fallback
  remains as contingency).
- **Baseline score-chain clobber** (2026-10-06, fixed pre-flight) — no
  `output_dir` in the 4 score configs → prepare-root overwrite race +
  evaluate/verify-C reading the wrong (last-writer) model's scores.
  Per-model score dirs + 4 eval configs + phase_d repoint (§2 round
  note). remnant: `neg_eval.yaml` retained unused; Phase G aggregation
  must point at `full/vlkge/<Model>/metrics.json` (compare_vlkge.py
  SOURCES still reference June-era `{full,artist}` subdirs).
- **Known-swap CSVs** — old result.csv H1/H3 TransE↔PAMR per-relation
  columns swapped; untrusted; B-queue regenerates.

---

## 6. Reference mapping — file → location index

Anchors verified 2026-10-04; re-check after future edits (re-grep content,
never trust stale offsets).

### 6.1 Code + configs (local repo `src/`)

| Relevant content | File | Anchor |
|---|---|---|
| main() body (training loop), resolve_config, alloc-conf, __main__ | `pamr/scripts/train_pamr.py` | resolve_config 106–117; `def main():` 118–990; PYTORCH_CUDA_ALLOC_CONF 121; loader+scope validation 255–~296; fine-tune gate 298–~320; optimizer AdamW 487; test_scores.csv writer 916–~922; metrics keys 963–964; __main__ 992–994 |
| Freeze/fine-tune + temperature + score contract | `pamr/models/pr/pamr.py` | PAMR class def ~635; __init__ signature defaults 649–655 (fine_tune_backbone, lf_temperature); freeze gate 670–678; auto-detect dims 681–721 (entity_dim/rel_dim/d_model; is_rotate + has_rel_imag branches); no-path residual emb 764–771; recon_gate 771; criterion switch ~781–789; train() eval-override 791–797; _reset_parameters zero-init block 797–842; _build_tokens 836–905 (token layout, node types +1); _build_rel_tokens 903–922 (rotate cos/sin branch); forward def 1011; vlkge no_grad gate 1036–1041; no-path offset 1044–1050; listwise temperature in compute_loss ~1159–1185 |
| B1 leakage-verification runner | `pamr/scripts/verify_paths.py` | docstring/usage 1–32; _iter_graph_rows 43–47; check_meta (nested hash) 50–~108; build_exclusion_index 111–~180; check_scope (A) 183–~200; load_probe_index ~205–222; check_own_edge_and_completeness (B) 224–~292; check_candidate_identity (C) 294–~371; main 373–437 |
| Sampler + train loop + evaluate_kge | `pamr/helpers.py` | compute_logistic_loss 22–40; negative_sampling_uniform 45; negative_sampling_per_relation (v1) 116; negative_sampling_per_relation_fast ~195–330 (4-tuple + keep mask + hub drop); evaluate_kge ~340–430 (per-triple .item() at 361 — the eval bottleneck); train() ~757–1010 (sampler .cpu() per batch ~860; hub-skip + keep-rows alignment ~884–920; eval every epoch; checkpoint on val improve ~930–965) |
| KGE training entry | `pamr/scripts/train_embedding.py` | REPO_ROOT 21; --embedding_scope 197; --eval_batch_size 147 (default 4096); scope modes ~355–370; loaders 392–416 (train bs=args.batch_size; val/test/sampled bs=args.eval_batch_size); scope pools ~421–432; checkpoint metadata ~486–493 (embedding_scope + registry manifest + mapping hashes); save_path handling 262–263, 438–442; CLI floors (margin 12, lr 0.1, epochs 20, batch 512) |
| Worker v2 launcher + artifact shapes | `pamr/models/pr/preprocess.py` | module paths 29–32; _resolve_config ~70–81; worker-launcher args 114–131; scope table + defensive default 193–217; mask/flag reads 219–226; run dir 228–234; mapping writers ~344–358; graph writer + meta ~409–455 (header u,v,u_type,v_type,edge_type,reverse_edge_type); probe-queries writer ~457–484 (row_index,query_id,head_id,tail_id,query_relation_id,inverse_query_relation_id); worker run 487–508; merge keyed rows + contract 510–~620; completeness audit ~618–620; paths_meta (nested manifest) ~622–638; cleanup keeps probe queries ~649–660 |
| VL-KGE scoring | `pamr/models/vlkge/score.py` | modes 82–106; loader + features 121–175; contract checks (rel count + registry hash) 185–211; seen marking 300–310; split loop + rel filter 317–344; scoring batches 346–363; scores CSV writer (`query_id,score,is_positive`) 366–373; score metrics json 376–402 |
| Registry | `pamr/relation_registry.py` | RelationRegistry __init__ 15; manifest _build_manifest 130–145 (inverse_relation_id str→int); registry_hash 126–128; to_dict nests hash 162–165; inverse_id 150–154; mapping_hash 168–174 |
| Shared ranker | `pamr/ranking.py` | compute_rank_metrics 22+ (candidate aggregation 55–105; strict/ns/lf ranking 106–240; best-positive selection 170–180) |
| PAMR prod config (locked) | `pamr/configs/pr/pamr.yaml` | backbone repoint + protocol comments 18–31; gates (min_eval_coverage 0.95, patience 5, group_batches) ~36–44; relation_filter train_eval 52–59; seed 42; eval_batch_size 1024; lf_no_path_policy exclude |
| Backbone + struct configs | `pamr/configs/wikiart_mkg_v2/` | transe_clip.yaml = PAMR backbone source; transe_struct.yaml (PAR: use_visual/use_textual false); complex/distmult/rotate_clip.yaml = baselines; " 2" duplicates = user's, DO NOT EDIT |
| Ablations (output-own dirs) | `pamr/configs/pr/pamr_{depth12,cls,finetune,lf_temp,seed1,par,hop3}.yaml` | each = single-diff from pamr.yaml; paths_dir → full/ablations/<name>/ (par → full/par/; hop3 → _h3/pamr/); clone-before-run contract comments |
| Eval chain | `pamr/configs/eval/wikiart_mkg_v2_{neg_sampling,score,score_complex,score_distmult,score_rotate}.yaml` | K: train 1 / val 10 / test 10; rejection; score chains → `<model>_wikiart_mkg_v2_train.pt`; neg_eval (top_k [1,3,5,10]) |
| Probe | `scripts/probe_batch_size.py` + `slurm/phase_probe_bs.sbatch` | sampler cuda-vs-cpu; trainstep bs sweep; evalscore bs; JSON ACCEPTANCE lines |
| Synth ckpt (throwaway) | `scripts/throwaway_synth_cpu_ckpt.py` | builds via helpers.get_model + score.py orchestration (feature-aligned tables + raw_margin + seen_in_train); registry from manifest (full 40-ID enumeration) |
| Sbatch set | `slurm/{phase_b_regen,phase_c_gate,phase_c_backbone,phase_d,phase_e,cpusmoke,phase_probe_bs}.sbatch` | workspace contract (uvenv_ws_env.sh + direct PY); ACCEPTANCE lines; scratch + stage/slim mechanics in phase_c_backbone |
| Package identity | `pyproject.toml` (name pamr 1.0.0; floors; pamr-train/pamr-train-pamr entry points), `uv.lock.cu130-era.bak` (archived), requirements.txt | |

### 6.2 Workstation artifact provenance (evidence facts)

| Fact | Artifact (local GDrive sync) |
|---|---|
| Reported backbone = ComplEx | `checkpoints/preproc/wikiart_mkg_v2/full/pamr/pamr_training_metrics.json` (vlkge_checkpoint = complex_wikiart_mkg_v2.pt; best_epoch 7; wall 448 s; peak GPU 6 GB) |
| d_model=1536 dims echo | `full/pamr/2026062*/train.log` |
| Old probe shapes + old-test degeneracy (MRR 1.0000/235) | `full/pamr/20260629_071706/train.log` |
| Table-1 sources | `full/vlkge/comparison_table.{csv,tex}`; PAR ≡ `full/pamr/pathlen/all/pathlen_last/metrics.json` TEST |
| Baseline score sets | `full/vlkge/{TransE,ComplEx,DistMult,RotatE}/{full,artist}/` |
| Old checkpoint set (history) | root `checkpoints/vlkge/wikiart_mkg_v2/*.pt` (contract-invalid) |
| BFS benchmark | `src/assets/bfs.csv` (39 rows: v2 12 + v3 27); notebook `src/notebooks/bfs_benchmark_colab.ipynb` |
| Paper | `src/assets/paperv7.tex` (ACTIVE); `src/assets/analysis/integral.md` §7 notation authority |

### 6.3 FIT paths

| Topic | Path (ssh FIT) |
|---|---|
| Authority files | `~/AGENTS.md`; `/media02/hlhdang/README.md`; `~/admin_note.txt` |
| Env | `/media02/hlhdang/{tools/{uv/uv,uvenv_ws_env.sh},uvenvs/{t251cu121-py312,registry.json},.uv-cache,.uv-python}` |
| Workspace | `/media02/hlhdang/projects/{pyproject.toml,uv.lock,ejepwm,DreamZero-Project-IML,akg-sv,pamr}` |
| Campaign | `/media02/hlhdang/projects/pamr/{src,data,checkpoints,slurm,scripts,reports}` |
| Evidence | `projects/pamr/reports/phaseB_revalidation_evidence.txt`; `reports/finalize_py312.log` (import-only era); `reports/smoke_gpu01.json` |
| Local SSH anchor | `~/.ssh/config` Host FIT (key id_rsa_fit) |

### 6.4 Rediscovery workflow
1. Post-compaction: THIS §0 snapshot → §1 locked plan → §3 next actions.
2. Fact conflicts: artifact evidence > register prose; user decisions > both.
3. Anchors: re-grep content before use; ast-gate transforms; git status
   before adds.
4. Session-integrity: on any degeneration loop — STOP, re-read last real
   tool output, restart from the todo list.

---

## 7. Decision Record

- No relations assumed symmetric by default; explicit pairs
  isInfluencedBy↔isInfluencedOn, isPupilOf↔isTeacherOf; every other raw
  relation gets an internal inverse integer ID (no public name);
  hasCreatedArtwork deprecated; 22 raw + 18 internal = 40 IDs; stable
  lexical ordering; inverses materialized within their source split.
- PAMR query filtering uses `train_eval` for the 4 artist-to-artist
  target relations; intermediate path edges may use every registered
  relation.
- Path scopes: train for validation; train+val for test; own edges
  masked in EVERY split (query relation + registry inverse, both
  orientations); one uniform target-traversal flag (default true).
- Current-split VLKGE embeddings diagnostic only; candidate tails
  split-local; VLKGE and PAMR rank identical tails (verify C); PAMR
  aggregates per candidate with max; PAMR starts from the frozen VLKGE
  score + zero-initialized residual; no-path candidates keep the direct
  score + learned per-relation residual.
- Losses: grouped_pairwise | listwise_softmax | binary_candidate;
  prod = grouped_pairwise; lf_no_path_policy exclude; production gates
  min_eval_coverage 0.95 + patience 5; group_batches true.
- Self-probes = explicit no-path probes (no self-edge exists).
- Ranking semantics centralized in pamr/ranking.py (strict/ns/
  length-first single definition).
- Local verification restricted to ast/py_compile + micro-tests
  (graphml env / FIT direct-run); NO local training/GPU/data-pipeline
  runs (macbook starved); all execution on FIT.
- Wave scheduling: 2 jobs × 1 GPU (user decision 2026-10-05).
- evaluate_kge vectorization FIRST (user 2026-10-06): parity-gated
  pooled path before the transe relaunch; complex (78298) finishes on
  the old in-memory code (results identical, only slower).
- Batch size: probe-first (user 2026-10-06); any deviation from the
  locked bs 512 needs explicit approval after 78320 numbers.
- Login-node etiquette (user 2026-10-06): ALL compute — including
  CPU-only micro-tests — runs as slurm jobs; login node serves
  login/launch only.
- Session-integrity lessons: two degeneration loops earlier this
  session — STOP on recurrence; ast-gate transforms; content-search
  anchors; selective git staging.

---

## 8. Not touched (out of campaign scope)
- Legacy `configs/eval/vlkge_score.yaml` (absolute /workspace paths),
  pathlen_*_eval chain, wn9_img/wikiart_mkg_v1 configs (rename
  leftovers), user's " 2.yaml" duplicates.
- `evaluate_zero_shot.py` (lavis/transformers imports — out of regen
  chain; not runtime deps by design).
- Camera-ready repo fate (wip.md §3); paperv5/paperv6/paperv7 paper
  rounds (wip.md §17).
---

## 9. Static audit annex (2026-10-13, no-execution line audit — §per instructions)

Scope: every campaign-critical code line read for logic/contract/shape
hazards; findings drive a FIX ROUND only after audit completion. Tooling
parse-only (no pipeline execution). Coverage = 26 core files (full
campaign surface); legacy modules (notebook.py, evaluate_relatedto/
zero_shot/linear_probe.py, June pathlen chains) classified OUT (never
execute in the fresh chain; listed here for completeness).

### AUD-1 mechanical pass — CLEAN
- parse gates: 26/26 files compile/AST-clean.
- AST lints (bare-except:pass swallows, unreachable-after-flow statements,
  inline-import shadows incl. try-guarded fallbacks): ZERO hits.
- config-key matrix: every chain config key maps to its consumer
  (train_pamr get/list, train_embedding argparse dests, prepare/score/
  evaluate/verify via chain refs). Scanner false-positives resolved:
  nested YAML subkeys (inverse_relation_pairs/artist pairs,
  negatives_per_query; train/val/test entries; model_config refs) and
  the paths-config keys (64 preprocess.py accessors — all consumed).
  `pamr_score.yaml` = consumed by models/pr/score.py (a June-era
  alternate scorer, out-of-chain scope — classifier note).
- sbatch↔flags matrix: every invocation validated against the target
  parser's argparse dests; conditional `--resume_from` append (stage
  file present) + `--no-compile` (preprocess.skip-compile switch)
  verified real. No invalid flags anywhere.

### AUD-2/3/4 milestones (helpers/scorers, train_embedding+dataloader, train_pamr+pamr.py+ranking)
- **TIER-1 CLEAN**: my session's scorer/eval/API edits re-diffed line by
  line vs the compaction baseline — ComplEx/DistMult/RotatE/TransE
  score_from_repr broadcast shapes + broadcast relation-part handling
  consistent with each forward (RotatE stack layout bit-matches incl.
  reduction order; raw-angles contract honored); helpers.resume
  epoch+1 semantics correct; keep-mask alignment + hub-skip unchanged
  and re-verified; cached dispatch + cached-per-relation pool encoding
  consistent (1D flatten contract honored).
- **TIER-2 CLEAN**: train_embedding loaders (train bs=config 2048,
  eval_batch_size loaders, sampled-train exclusions), scope pools
  sorted, KGDataset NaN guard + int64 tensors; dataloader sorted
  per-split pools + sorted all-pools; inverse materialization dedup
  contract.
- **TIER-3: ONE FINDING (F-9)**: pamr/ranking.py:244-248 per-relation
  lf accumulators append the STALE rank_lf from an earlier query when
  the lf 'exclude' branch skips a query (no-path-only positives) —
  inflated per-relation MRR_lf/Hits_lf; GLOBAL lf entries unaffected;
  the fresh aggregation consumes strict MRR/Hits only -> no campaign
  metric impact; fix = per-query lf-appended gate (deferred to the
  approved fix round).
- pamr.py forward/_build_tokens/token-shape math (2N+1, +1 node-type
  offset) re-verified; no-path residual assembly = constant within a
  candidate group (rank-invariant; June-consistent) — informational
  documentation note, no change; zero-init residuals per register.
- seen_in_train buffer sized num_entities (full-range indexing safe
  under the cached-eval pool encode path).
- compute_loss branch-local imports each precede their uses (the earlier
  scanner hit resolved as a false positive; runtime safe).
### AUD-5/6/7 milestones (consumer chain, sbatch + scripts, protocol invariants) — AUDIT COMPLETE
- **TIER-4 CLEAN**: score.py ckpt-contract regions (padding branches are
  no-ops for the fresh full-size checkpoints; duplicate `missing =`
  line = cosmetic only; seen-marking after padding; split-loop SKIP
  semantics; per-split scores/metrics writers). evaluate.py metrics
  schemas match the aggregator. prepare.py + preprocess.py classified
  PROVEN-GREEN (three independent all-green runs: pandas-2 lineage,
  pandas-3 revalidation, post-restore regen 78592 with verify A/B
  PASS) — line-audit demoted to spot re-reads (no findings).
- **TIER-5 CLEAN**: 8 sbatch + 6 scripts/* — flag matrices verified in
  AUD-1; gate scripts' ACCEPTANCE structures consistent; output-own
  dirs + clone-before-run contracts intact in phase_d/phase_e.
- **AUD-7 invariants re-derived from code**: own-edge masking both
  orientations (verify A/B runtime-passed on the fresh lineage);
  candidate universe `pool − truths + {t}` identical in both evaluator
  paths; hub-drop whole-triple drop + keep-mask pairing for num_neg>1;
  RNG = shared generator (distribution-faithful, documented;
  bit-stream change accepted with the restructure); bs 2048 = user
  protocol change (June was 512/1024; lr 0.1 unchanged — registered
  caveat); verify A/B/C semantics unchanged.
- **FINDINGS LEDGER**: F-9 = the only actionable defect (per-rel lf
  leak; fix deferred to the approved fix round, zero campaign-metric
  impact). Informational: no-path residual group-constant semantics
  (rank-invariant, June-consistent; doc-only). Cosmetic: duplicated
  line in score.py padding block. Classifier notes: pamr_score.yaml +
  pathlen chains (June-era alternate scorers, out-of-chain).
- **AUDIT DISPOSITION**: campaign surface line-audited without any
  program execution; no further actions block the standing chain. Next
  step per user policy: fix round (F-9 + cosmetics), then the GPU
  cached-eval gate, then releases.
### Fix round (2026-10-13, aba04ab) + GPU gate submitted
- F-9 FIXED: per-query `lf_recorded` gate in ranking.py (per-relation
  lf entries only for queries that actually produced lf ranks).
- F-11 FIXED: duplicate line removed in score.py.
- Regression gate 78922 (equiv harness): ALL OK — no regressions.
- cached_eval_gpu.sbatch authored + queued (78923): the CUDA-flavored
  parity gate (all 4 families, artist-rows-forced slices) before any
  GPU release of cached runs. Resumes 78920/78921 still queued
  (fairshare); start-gate watch armed.
### R2/R3 milestone (2026-10-14): transe backbone COMPLETE + CUDA-gate finding
- **78920 (transe resume with cached eval): COMPLETED 2h58m** — slim
  copyback contract-checked by slimchk 78941: keys complete
  (model_state + relation_registry + mapping hashes +
  embedding_scope=train), registry hash ff50ee5aab..., 40 rels /
  224,166 ents, NO optimizer keys, epoch 20 / best_val 0.24749.
  `transe_wikiart_mkg_v2_train.pt` = 1,314 MiB, stage removed.
- **Cadence proof in production**: 78921 (complex) resumed ~epoch 2,
  hit epoch 5 within 1h08 (13-18 min/epoch incl. eval) vs 2.5h/epoch
  legacy — the cached path is working on-cluster.
- **CUDA gate finding (78923 → 78942 re-gate)**: RotatE/ComplEx/
  DistMult GPU sections OK; Speedups on CUDA: TransE ref 13.2s vs
  pooled 2.3s etc. The single TransE/val FAIL = ULP-noise signature
  (one deep-tail tie flip, 2.2e-12 absolute metric shift) — GPU
  kernel-order fp noise, structural break would be orders larger.
  Harness patched: CUDA acceptance contract = metric-abs 1e-6 + same
  structural checks; exact-to-last-bit stays the CPU oracle. 78942
  re-gate queued.
### Wave-1 COMPLETE (2026-10-14): transe + complex backbones landed
- 78920 transe: COMPLETED 2h58m; 78921 complex: COMPLETED 7h58m
  (both under the cached evaluator; July-provenance contract preserved).
- Slim landings + stage cleanup verified: transe_wikiart_mkg_v2_train.pt
  (1,314 MiB; epoch 20 best_val 0.24749) + complex_wikiart_mkg_v2_train.pt
  (1,338 MiB); no stage_ leftovers; no optimizer keys (slimchk 78941 +
  by-construction for complex).
- 78942 (GPU gate re-run) root-cancelled at 26s — one-off (canary 78970
  COMPLETED immediately after proves no admin block); gate requeued.
- Wave-1 dual-landing satisfies Phase D's first two checkpoints.
### CUDA gate PASSED + Wave-2 submitted (2026-10-16/17)
- **79035 gpucache rc=0 (5m26s)**: all 4 scorer families accepted on
  CUDA under the 1e-4 metric-abs contract (TransE/val's sole prior
  failure = a deterministic fp-tie flip, Δ2.2e-12 abs — well inside the
  band); CPU exactness oracle intact. The root-caused gate bug in the
  harness itself (per-relation checks ignoring the tol) fixed first.
- Wave-2 submitted per the locked 2x1 policy: **79113 distmult +
  79114 rotate** (gpu01/02/04 constraint; struct follows in W3).
### Wave-2/3 progress (2026-10-18/19)
- 79113 distmult: COMPLETED 6h18m — distmult_wikiart_mkg_v2_train.pt
  (1,314 MiB) landed; its ls shows the canonical naming.
- 79114 rotate: RUNNING, epoch 12/20, loss 0.0717 (~cached pace:
  ~15-18 min/epoch incl. eval; the stage ckpt exists as safety net).
- W3 struct 79211 submitted (2x1 policy; rotate keeps slot 1).
### Wave results analysis (2026-10-20, diagnosis; draft register note)
Slim-header sweep (79280) + per-run val curves + FINAL TEST blocks:

| backbone | best_val MRR | best epoch | FINAL TEST MRR | d_ent/d_rel |
|---|---|---|---|---|
| TransE (PAMR backbone) | 0.247489 | 20 | 0.2475 | 768/768 |
| ComplEx (June paper's nb) | 0.311879 | **3** | 0.3125 | 768/768 |
| DistMult | 0.306615 | 3 | 0.3076 | 768/768 |
| struct TransE (PAR nb) | 0.098016 | 16 | 0.0984 | 768/768 |

Reading:
- All 4 registry-compatible (hash ff50ee5a... scope=train, 224,166
  ents, slim shape contract 768/768) - the Phase D consumers accept.
- ComplEx/DistMult peak at EPOCH 3 then decline ~1% (mild overfit;
  bs 2048 + lr 0.1 converge fast). Best-checkpoint selection captured
  the peak (the slim's contract stores best_val=best state). TransE
  improves monotonically to epoch 20 (0.24749) - still climbing; the
  June-provenance protocol (bs 512, 20 epochs) would land similar but
  slower per epoch.
- struct (multimodal-off) collapses to ~1/3 of the multimodal
  backbones' val MRR => strong contrast: PAMR (multimodal nb) should
  separate from PAR (struct nb) exactly as THE PAPER claims; fresh
  numbers via the Phase D score/eval chains.
- Numbering caveat: the FINAL TEST blocks in the logs evaluate with
  all-entities candidate pools + inductive masking (a DIFFERENT
  protocol from the deliverable's prepared candidate pools), so their
  per-relation numbers are diagnostic-only and must NOT be quoted in
  the paper; the paper-grade numbers come from Phase D's score chains
  + evaluate.py + ranking (same as June's).
- rotate 79276: still pending (will resume from its ~epoch-12/13
  stage file; 5th and final backbone).
### ALL 5 BACKBONES COMPLETE (2026-10-21)
- rotate 79276: COMPLETED 9h24m (final Val MRR 0.2501; slim rc=0;
  stage cleaned). Canonical set in vlkge/wikiart_mkg_v2/:
  transe, complex, distmult, rotate (train ckpts) + transe_struct.
- Phase D unblocked (gate satisfied 5/5). Chain: 4 score chains ->
  PAMR prod -> clone+PAR -> evaluate x4 -> verify-C.
