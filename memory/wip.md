# WIP — PAMR paper status (SOICT 2026, paperv7) — rewritten 2026-10-22

**Paper status register. Campaign/engineering ground truth lives in
`memory/bugs.md` (read its §0 first). This file keeps only what the
paper needs.**

---

## 0. READ-FIRST

- **Current draft**: `assets/paperv7.tex` — SOICT 2026 submission,
  submitted **2026-09-20**. Notification date **2026-10-12** (outcome
  not yet recorded in this register — confirm with the user before any
  camera-ready action).
- **Fallback drafts**: `assets/paperv3.tex` (keeps Per-Relation
  Analysis + merged per-relation figure). History: paperv0/1/2.tex,
  `assets/paper.tex`.
- **Context**: original venue submission was rejected 2026-09 (no
  rebuttal needed); `assets/review.md` issue numbers #1–#14 guide
  future-venue hardening only. Tags: [Original] vs [SOICT].
- **What is happening now**: the post-submission campaign
  (bugs.md §0/§3) is regenerating ALL numbers under the corrected
  pipeline. Camera-ready plan = replace the paper's numbers with the
  fresh campaign's authoritative values (locked decision: "camera-ready
  = correct the numbers"). Old `assets/{result,average,count}.csv`
  values are UNTRUSTED (H1/H3 column swap artifact; mixed split/source
  branches) — do not quote them.

---

## 1. Where things live

| Need | File |
|---|---|
| Campaign state, decisions, defect register, FIT handoff | `memory/bugs.md` (single source of truth; §0 snapshot) |
| Paper prose/draft | `assets/paperv7.tex` (ACTIVE) |
| Notation authority | `assets/analysis/integral.md` §7 |
| Original review issues | `assets/review.md` (#1–#14; [Original] tags) |
| BFS benchmark sweep | `assets/bfs.csv` (39 rows: v2 12 + v3 27) |
| June-era metrics/scores (UNTRUSTED, format reference only) | local mirror `checkpoints/preproc/wikiart_mkg_v2/full/{vlkge,pamr}/` |

---

## 2. Paper identity (locked)

- Method: **PAMR** (Path-Aware Multimodal Reasoning) — frozen VL-KGE
  embeddings as tokens in a transformer; per-candidate path residual
  `ŷ = g + Δ` (Δ zero-initialized); no-path candidates keep the direct
  score + learned per-relation residual.
- Position vs prior work: VL-KGE uses explicit negative-sampled
  evaluation (10 candidates per query); PAMR must beat it *under that
  protocol*. The open question (bugs.md headline): why PAMR does not
  outperform VL-KGE despite consuming its frozen embeddings.
- From paperv4 onward: runtime/scalability analysis replaced the
  per-relation analysis (v3 fallback keeps per-relation figures).

---

## 3. Paper history (condensed — details recoverable from git history)

- **paperv0–v2**: original-venue rounds; issue list #1–#14; rejected
  2026-09 (no rebuttal).
- **paperv3**: fallback with Per-Relation Analysis + merged figure.
- **paperv4**: SOICT revision; runtime section replaces per-relation
  analysis; N-MMKG taxonomy round (v4 only); §3.1 Overview rewrite
  (structural/methodological round, v4 only).
- **paperv5–v7**: SOICT hardening rounds — compression, humanization,
  audit, compliance; integral-analysis notation round (pre-revision);
  final submission 2026-09-20.
- **Post-submission (2026-10 →)**: campaign regeneration (bugs.md §2/§3);
  Phase H timings destined for the runtime table; ablations for the
  ablation table; fresh Table-1 numbers from Phase D/E artifacts.

---

## 4. Numbers provenance and campaign → paper mapping

### 4.1 What the published numbers are

- June artifacts prove the published backbone was a **frozen ComplEx
  checkpoint** (d_model 1536; `vlkge_checkpoint` field in the June
  `pamr_training_metrics.json`). June Table-1 sources: June
  `full/vlkge/{Model}/{full,artist}/metrics.json`; PAR ≡
  `pathlen_last/metrics.json` TEST; PAMR ≡ June prod metrics.
  The fresh campaign is **TransE-faithful** (locked decision) → the
  regeneration is a NEW configuration; deltas have two sources (fixed
  pipeline + documented backbone change). Report both honestly.

### 4.2 Fresh artifacts → paper tables (when Phase D/E complete)

| Paper item | Fresh artifact (FIT paths under `projects/pamr/checkpoints/preproc/wikiart_mkg_v2/`) |
|---|---|
| Table 1 rows: VLKGE-TransE/RotatE/DistMult/ComplEx | `full/vlkge/<Model>/metrics.json` (per-relation MRR/Hits@1/3/5/10 + overall; per-model output-own dirs) |
| Table 1: PAMR | `full/pamr/pamr_training_metrics.json` (`test`, incl. `per_relation`) + probe-level `test_scores.csv` |
| Table 1: PAR (Path-Aware Reasoning) | `full/par/pamr_training_metrics.json` (`test`) |
| Ablation table (5 core ablations) | `full/ablations/{seed1,finetune,depth12,cls,lf_temp}/pamr_training_metrics.json` + hop3 in `_h3/pamr/` |
| Runtime/scalability table (Phase H) | `scripts/phase_h_timings.py` → `timings.csv` (train/test walls, GPU peaks, dataset-build timings, per-epoch means) + `reports/regenB_*.out` (BFS walls) |
| Aggregated CSVs (fresh) | `scripts/aggregate_campaign.py --mode fresh` → `assets/report/{result,count,average}.csv` (June format; TEST split with test-matched weights) |

### 4.3 Protocol facts that must match the prose

- 10 candidate negatives per query; PAMR seeds 42 + 1 (n=2, std
  ddof=1); baselines seed 42.
- Artist-to-artist target relations under `relation_filter:
  train_eval`; per-relation candidate pools; own query edges masked in
  every split (both orientations).
- Backbone scope = train everywhere. BFS prod H=5/F=100/δ=0.5; hop3
  ablation uses fresh `_h3` paths.
- Backbone training batch size = **2048** (deviation from June's
  512/1024; lr 0.1 unchanged — a documented protocol change; do not
  hide it in the paper's methods text).
- Evaluation-path implementation notes (cached evaluator, CUDA 1e-4
  tolerance) are engineering facts, not protocol changes — scores are
  parity-verified against the reference implementation.

---

## 5. Open paper items

1. **Camera-ready numbers** (if accepted): run the aggregation after
   Phase D/E; produce the grouped result/average CSVs; replace every
   table/figure number; keep the ComplEx-provenance caveat in the
   methods or appendix as decided.
2. **Notification follow-up**: outcome of SOICT 2026 (notification was
   2026-10-12) is not yet in the registers — ask the user; then update
   this file's §0.
3. **Ablation table**: values from Phase E/G artifacts (seed-mean
   presentation for PAMR rows decided at aggregation time — n=2,
   ddof=1).
4. **Runtime table**: Phase H timings + BFS walls; check the paperv7
   runtime prose still matches the measured regime (cached eval ⇒
   ~10–13 min eval per epoch; train ~4–5 min).
5. **Repo/artifact availability footnote**: June decision was to
   remove the GitHub footnote; revisit only if the venue requests code.
6. **Consistency sweep**: any claim about "ComplEx backbone" in
   paperv7 must be reconciled with the TransE-faithful regeneration
   (see §4.1).

---

## 6. Key decisions affecting the paper (mirror of bugs.md §1)

- Backbone = TransE-faithful; June = ComplEx (caveat above).
- Full regeneration matrix, train-scope backbones, 10-negative eval,
  seeds 42+1, core-5 ablations.
- Old CSVs untrusted; fresh outputs authoritative.
- bs 2048 protocol deviation (documented, not hidden).
- Camera-ready = correct the numbers if accepted.
