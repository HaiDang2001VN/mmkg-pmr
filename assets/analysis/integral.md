# Integral revision plan — paperv4 → paperv5 (§3 Method onward)

**READ THIS FIRST (after any compaction) — SUBMISSION MILESTONE 2026-09-20.**
The user REPORTS the paper has been SUBMITTED to SOICT 2026 (deadline day).
Submission build = `paperv7.tex` (the active draft at submission time:
paperv5 + paren round + grammar-typo round + sweep-claim/abstract-tail
round, plus user manual abstract edits — see blocks below; 634 lines at
last record). Frozen history: paperv5 = pre-paren-round copy; paperv4 =
pre-integral rollback; paperv6 never existed (numbering skipped).
MAIN PROJECT GOAL (SOICT 2026 submission, ≤12 pages excl. refs,
single-blind llncs) is therefore MET as far as local work goes — any
remaining work is POST-SUBMISSION only: (1) regenerate KNOWN-SWAPPED
result.csv/average.csv H1/H3 TransE↔PAMR columns (data repair, paper
already submitted with the correct swapped-in values); (2) B1–B4
experiment queue (wip.md); (3) camera-ready decisions after
notification (acknowledgment block + AI-disclosure comment currently
COMMENTED OUT in the credits block — decide before camera-ready);
(4) repo/publication decisions (wip §16). Earlier compaction pointers
(PAREN/GRAMMAR-TYPO/SWEEP-CLAIM blocks) remain below.

Written 2026-09-20. Companion to `assets/analysis/{review.md, n-mmkg.md, overview.md}`
and `memory/wip.md` (§11 symbol map, §16 handoff). This file is the *deep*
analysis round: every claim below is code- or data-verified, and the per-subsection
moves are concrete enough to execute without re-deriving anything.

**Status.** `assets/paperv5.tex` is a byte-exact copy of paperv4 (41,863 bytes,
655 lines). **No LaTeX content has been edited yet.** All line numbers below
refer to paperv4 = paperv5 as of this copy.

**Constraints set by the user (2026-09-20).**
1. Integral editing only: rewrite/merge sentences so inserted ideas and old
   statements read as one text; no more scattered insert/delete patches.
2. Scope: §3 Method → §5 Conclusion. Front matter (abstract, intro, related
   work) is out of scope except for consistency echoes (noted where relevant).
3. Keep the paper **generic**: no deep implementation detail in the text.
4. Page-count guard: SOICT limit is 12 pages excluding references. Every
   addition must be paid for by a merge or trim; the round must end ≈ length
   neutral (target: not more than +3 net lines over paperv4).
5. De-AI style rules from wip.md §16 stay in force; notation stays per §11;
   no numbers change; paperv4 remains untouched as rollback.

---

## 1. Verification results (code- and data-backed)

Seven questions were verified directly in the code and data. Verdicts here
decide what paperv5 may safely say.

### V1 — Configuration provenance and readout mode
- Evidence: `pmr/configs/pr/pamr.yaml` (prod). Architecture/training fields
  match the paper's stated hyperparameters **exactly**: N=8 layers, h=8 heads,
  d′=2048, dropout 0.1, lr 0.01, weight decay 0.01, batch 8, 50 epochs,
  K=5 (`max_hops`), L=13 (`max_tokens`), margin ranking loss
  (`loss_mode: grouped_pairwise`, margin 1.0 in code), seed 42.
- Stale fields: `vlkge_config`/`vlkge_checkpoint` point to the ComplEx
  checkpoint ("ComplEx → 1536" comment) — these pre-date the scenario-A
  correction (reported runs used TransE, D=768; paper §4.4 already says
  TransE). A `transe_clip.yaml` VL-KGE config exists
  (`pmr/configs/wikiart_mkg_v2/`), consistent with the TransE runs.
- Readout: `readout: average` in prod config; `readout="cls"` only in smoke
  configs. `average` = mean pooling over valid tokens (`pamr.py:969-973`).
- Verdict: the reported checkpoint almost certainly used **mean pooling**.
  The paper never commits to a readout (paperv4:333-335 lists both strategies
  without choosing). → One committing sentence in §3.4.3 or §4.1, pending
  final user confirmation (the run-time config on Vast is not archived
  locally; no local run logs exist).

### V2 — Frozen table covers all of 𝒱* (𝒱 vs 𝒱* question)
- Evidence: PAMR obtains entity tokens from the frozen VL-KGE model's entity
  table over the shared registry (`pamr.py:847-856`,
  `get_entity_representations`); the same `entity_to_id` registry is used by
  `preprocess.py` and `prepare.py`. The VL-KGE model was trained on the full
  graph (224,166 entities: 216,564 artworks + 7,602 attributes; dataset
  README). Artwork (multimodal) nodes therefore have unified embeddings.
- Verdict: the frozen table covers **𝒢*** entirely, both partitions.
  §3.2 currently writes 𝐄 ∈ ℍ^{|𝒱|×D} and "for each entity v ∈ 𝒱"
  (paperv4:187-198), which is the base-graph symbol from §3.1. → Harmonize
  to 𝒱* (two small math edits, no added length). This also strengthens the
  N-MMKG story: multimodal nodes are first-class *and* embedded.

### V3 — "Query and answer differ in modality" is factually wrong
- Evidence: dataset README (`pmr/data/wikiart_mkg_v2/README.md`): entity
  partition = **visual entities (artworks): 216,564; textual entities
  (attributes): 7,602**; modality table: "Artworks: Visual features (images).
  Artists, Styles, Movements, etc.: Textual features (names/descriptions)."
  The four evaluation relations connect artist→artist (README relations
  17-20). So both query and answer endpoints are **textual** entities; the
  clause "query and answer differ in modality" (paperv4:441-443, inherited
  from the original paper.tex:482-483) cannot be right as written.
- Related fact found in the same README: the excluded triple
  (isRelatedToArtwork, isRelatedToArtist, hasCreatedArtwork) is **not** "three
  preprocessing inverses". Only hasCreatedArtwork is an inverse (of
  isCreatedByArtist); the two isRelatedTo* relations are similarity links
  that the dataset's own evaluation treats with separate retrieval metrics.
  paperv4:432-434 gives the wrong reason.
- Defensible replacement (keeps the established "cross-modality" term,
  defines it at path level): the four artist relations are the only
  evaluation relations whose connecting evidence is carried by the *other*
  modality — query and answer are artists (textual), while the artworks that
  can mediate their paths are visual — so this is exactly where
  modality-as-entity structure carries multimodal information. (Wording to
  be finalized with the user; see §6.)
- Propagation: "cross-modality" appears in the abstract, §4.2, §4.4, and the
  Table 1 caption. Whichever wording wins must stay consistent in all four.

### V4 — Padding semantics: the paper's formula is correct
- Evidence: `pamr_collate` (`pamr.py:368-435`) pads to the batch-maximum
  **node** count (`max_N`), builds the interleaved token layout as
  2·max_N+1 tokens (CLS + query relation + N entities + N−1 edges,
  `pamr.py:837-838`), and marks padding in the token mask. So
  N_max(ℬ) = max(ℓ_p+1) entity tokens, token length 2·N_max+1 = 2ℓ+3 —
  exactly the paper's Eq. (1) and §3.5 formula.
- Verdict: keep the math unchanged. Only the cross-reference dedup applies
  (§3.5 re-derives what §3.4.2 already uses — see §3.5 below).

### V5 — The score is a residual on the frozen triple score (absent from the paper)
- Evidence: score contract docstring `pamr.py:640-646`;
  `forward` (`pamr.py:1002-1084`): the frozen VL-KGE triple score is computed
  under `torch.no_grad()` (`pamr.py:1022-1025`); probes whose BFS found no
  path receive a learned per-relation offset (`pamr.py:1030-1035`);
  path probes receive the readout MLP output as a residual added to the
  frozen score (`pamr.py:1079-1083`). Final layer of the MLP is
  zero-initialised so untrained PAMR reproduces the frozen scorer exactly
  (`pamr.py:741-749, 803-806, 813-818`).
- paperv4 says the MLP maps the aggregated representation "to a scalar score
  ŷ" (paperv4:336-338) — it presents the MLP output as *the* score. That is
  not what runs. → **Factual correction required**, stated generically: the
  readout produces a *path residual* added to the frozen triple score. No
  implementation minutiae (no init story, no no-path enumeration beyond one
  clause in §4.3) — the user asked for a generic paper.
- Consequences: §3.4.3 rewrite (one sentence); §4.4 crossover mechanism can
  now be stated truthfully (see §4.4 plan); §5's limit sentence stays valid.

### V6 — Multi-shortest-path aggregation rule (closes wip.md #6a)
- Evidence: `pmr/ranking.py:22-45, 53-89` — probes are aggregated per
  (query, candidate tail) by **max** reduction before ranking; the reported
  MRR/Hits are the *strict* ranking over the candidate pool (gold + 10
  negatives; `ranking.py:175-188`). Hop metadata feeds only the separate
  length-first diagnostics (`lf_*` keys), not the reported metrics.
- wip.md §2/#6(a) said the selection rule was "text-fixable ONLY after B1
  inspection". B1-equivalent inspection is now done. → §4.3 may state the
  rule in one generic sentence: candidates rank by the maximum path score
  among their extracted paths; candidates without an extracted path keep the
  frozen triple score (plus its learned offset — one clause).
- Also verified: `lf_no_path_policy: exclude` affects only the lf
  diagnostics; reported metrics rank every prepared candidate. §4.3's
  "every candidate object" claim is correct as-is.

### V7 — Token-side components omitted by Eq. (1) (record, do NOT add)
- Code adds an edge-direction embedding (`pamr.py:714-717, 863`) and a node
  type embedding (types 1=artwork, 2=artist, 3=attribute;
  `pamr.py:724-728, 885-890`; derivation in `preprocess.py:317-341`) on top
  of the sequence Eq. (1) describes. Per the user's generic-paper
  constraint, **these stay out of paperv5** (recorded here only so the
  omission is a deliberate choice, not an oversight). If a reviewer ever
  asks, the code answers; the text need not.

### V8 — Readout activation and initialisation details
- The readout MLP uses **ReLU**, not the GELU the paper's ŷ formula states
  (`pamr.py:744-749` vs paperv4:337). The "all linear layers use Xavier
  uniform initialisation and zero bias" sentence (paperv4:339-340) is wrong
  for the residual output layer (zero-init by design). Both vanish anyway
  once §3.4.3 is rewritten around the residual contract (the init trivia can
  be trimmed rather than expanded — generic-paper constraint).
- Also verified while checking: Post-LN encoder with GELU is correct
  (`pamr.py:730-739`); CLS init 𝒩(0,0.02²) correct (`pamr.py:789`);
  positional embedding over {0,…,2K+2} correct (`max_tokens=13`,
  `pamr.py:722, 837-838`); dropout position (after activation) correct.

### V9 — Sweep numbers in §4.5 all match the raw data
- `assets/bfs_sweep_grid.csv` at K=5,F=100: found 2208, reach 99.37% ("99.4%"
  ✓), identical across δ ✓, avg hops 4.50/4.54/4.38 → "average 4.5 hops" ✓,
  wall 3.84/8.65/14.76 s → "3.8, 8.6, 14.8 s" ✓. At F=10: 1542/1974/2115
  paths ✓; reach mean±(sample) std = 84.47±13.43 → "84.5% ± 13.4" ✓. K=2 →
  0 paths at every F/δ ✓ ("minimum viable bound K=3" is empirical).
- Val split 202 queries × 11 candidates = 2,222 probes ✓ (`count.csv`:
  74+76+26+26=202).

### V10 — The three-hop floor is structural, not just empirical
- Evidence: the relation taxonomy (dataset README, 22 relations): the only
  artist↔artwork edges are creation edges (isCreatedByArtist artwork→artist,
  plus the added inverse hasCreatedArtwork), and a creation edge names
  exactly one artist. No artwork node therefore connects two distinct
  artists, so an artwork-mediated route between two artists needs at least
  three hops (artist→artwork→artwork→artist — exactly the case-study Dali
  path). isRelatedToArtwork/Artist are artwork-artwork / artist-artist
  similarity links and cannot carry an artist→artwork hop.
- Verdict: paperv4:571-574's "connecting two artists through shared artwork
  nodes takes exactly three mediation hops (Section 3.1)" is structurally
  sound but cites the wrong anchor — §3.1 contains no such fact. → Move the
  one-sentence reason into §3.3 and re-point §4.5's reference there.

---

## 2. What the verification changes about the revision strategy

1. **The residual fact must go in, generically.** It is the one place where
   the paper currently says something false (the MLP output is *the* score).
   Cost: one sentence in §3.4.3 + one clause in §4.4's crossover discussion
   + one clause in §4.3. Nothing else about the contract (init, no-path
   residual internals) enters the text.
2. **PAR and PAMR become one design statement.** With V5 known, the correct
   framing is: PAR and PAMR share the frozen backbone and the same residual
   scorer; they differ *only* in whether extracted paths may leave the
   conceptual subgraph. §4.4's current differential sentence
   (paperv4:492-494) half-says this. Making it explicit turns the topology/
   modality confound defense (wip #2) from prose into design, and it is the
   honest reading of PAR 0.221 < TransE 0.363: a structural-only residual
   *hurts* the frozen score, a multimodal one *helps*. One sentence, high
   value.
3. **The crossover explanation can drop its speculation.** paperv4:536-540
   attributes the Hits@1/3 crossover to "how the models spread probability",
   which is hand-waving; the mechanical truth (base term preserves TransE's
   head, the residual reorders where path evidence disagrees) is now
   available and shorter. Keep the observed crossover description, replace
   the speculative cause.
4. **Length ledger.** Additions: residual sentence (~2 lines), aggregation
   sentence (~2 lines), §4.2 justification rewrite (±0), §3.3 structural
   reason (~2 lines), readout commitment (~1 line). Offsets: §3.3 restructure
   (drop `\paragraph` heading, one duplicate PAR sentence, smoother
   transitions, −3 to −4 lines), §3.5 orphan merge (−1), §4.5 merge (−2),
   §4.6 merge (−1), §3.4.3 init trivia trim (−1). Net ≈ −2 to +2 lines:
   page-count safe.

---

## 3. Per-subsection integral plan

Each entry: current structure with paperv4 line numbers → seams (why it reads
stitched) → integral moves (numbered) → length impact.

### 3.1 §3.1 Overview (lines 119-180) — light; already a full rewrite round
Current: four paragraphs — KG/A-MMKG (122-132), N-MMKG definition with the
three verbal constraints (134-151), path objective on 𝒢* (153-164),
four-component pipeline (166-173).

Seams:
- S1: para 1 ends with the bolted-on parenthetical "(Section 3.2 gives the
  aggregation)" (132).
- S2: para 2 opens "Our framework instead operates on a node-based multimodal
  knowledge graph (N-MMKG), the modality-as-entity realization that
  externalizes modality-bearing attributes into first-class entities" —
  "modality-bearing attributes" does not name the object just described.

Moves:
- M1 (S1): weave the pointer into the sentence: "...into a single point
  𝐯 ∈ ℍ per node, an aggregation whose form Section 3.2 defines." (±0)
- M2 (S2): one-clause integral stitch naming the contrast object:
  "...first-class entities — externalizing precisely the values the
  attribute-based formulation folds into one point." (±0)
- No structural change; no new facts.

### 3.2 §3.2 Embedding Module (lines 182-216) — three paragraph bridges
Current: frozen KGE + 𝐄/𝐑/g definitions (185-192); multimodal entities +
VL-KGE unified embedding + averaging (194-205); dimensionality + projections
(207-216).

Seams:
- S1: para 1→2 topic jump ("we use a frozen pre-trained KGE model" → "In a
  multimodal KG, entities possess heterogeneous modalities"). The *need* for
  the unified embedding (PAMR's tokens must already carry multimodal
  content) is never stated; VL-KGE arrives as a non sequitur.
- S2: para 2→3 jump ("The effective dimensionality … depends on the KGE
  architecture"). The causal chain (one token width → align the frozen
  vectors) is inverted into a fact dump.
- S3 (V2): the table is written over 𝒱, but PAMR tokenizes every node of
  𝒢*, multimodal nodes included.
- S4 (V1/V8): "two learnable linear projections align entity and relation
  token dimensions to d" — true for wider backbones; for the reported
  TransE backbone (D = d = 768) the projections reduce to identity
  (`pamr.py:708-713`). Unconditional phrasing overstates.

Moves:
- M1 (S1): rewrite para 2's opening from need to mechanism: "In a multimodal
  KG the tokens PAMR consumes must already carry each entity's heterogeneous
  modalities. VL-KGE~\cite{efthymiou2026vlkge} constructs exactly this: a
  unified embedding for every entity, obtained by averaging over its
  available modalities …" — the averaging formalism then follows as the
  mechanism, unchanged. (±0)
- M2 (S2): role-first rewrite of para 3: "Because the Transformer consumes a
  single token width, the frozen vectors must be aligned to it. Backbone
  width varies: complex-valued models output 2D dimensions and real-valued
  ones D, so d = max(d_V, d_R), with two learnable linear projections
  aligning the two sides where they differ; for the reported TransE
  backbone, D = d = 768 and the projections reduce to identity. These
  projections are learned while the KGE model remains frozen." (≈ +1 line,
  absorbed by S5 trim below)
- M3 (S3): 𝒱 → 𝒱* in both spots (|𝒱|×D shape sentence, "for each entity
  v ∈ 𝒱*"), plus "every node of 𝒢* — conceptual and multimodal alike —
  receives a unified embedding" folded into M1's sentence. (±0)

**EXECUTED 2026-09-20** with two user decisions overriding the draft:
(a) projections: generic conditional ONLY — "where the widths differ, two
learnable linear projections align entity and relation tokens to d"; no
TransE instance clause, no "auto-detected"-style flavor anywhere (the
§4.1 instance sentence keeps the factual TransE data but loses the
flavor, see §3.6 note). (b) MMKGE terminology: §3.2 now says "frozen
pre-trained MMKGE model" (KGE would denote non-multimodality); VL-KGE
naming + citation DEMOTED out of §3.2 → transferred to §4.1 (§3.6
below). 𝒱* harmonization done per recommended option: para 1 generic
𝒱, instance sentence v ∈ 𝒱*, coverage clause in M1's opening. Net
paperv5 delta so far: §3.1 +1 line, §3.2 +2 lines.

### 3.3 §3.3 Path Extraction (lines 218-266) — the main restructure
Current: intro (221-224); `\paragraph{Multimodal and non-multimodal
configurations.}` (226-235); entanglement/ablation paragraph (237-251);
leakage paragraph (253-259); formal path definition (261-266).

Seams:
- S1: the `\paragraph` heading (226) governs two paragraphs that are not
  "configurations" — a leftover of an older structure.
- S2: PAR is defined twice ("relies solely on structural connectivity
  without multimodal signals", 234-235; "serves precisely as this
  topology-only counterpart …", 243-245).
- S3: the leakage paragraph (253-259) interrupts the ablation argument and
  the formal definition with no bridge; the formal definition arrives *last*,
  after leakage — backwards.
- S4 (V10): the structural reason for the three-hop floor lives only in §4.5
  and cites §3.1, which does not contain it.

Moves (target order: intro → formal definition → configurations + entanglement
merged → leakage close):
- M1 (S1): drop the `\paragraph` heading; §3.3 becomes continuous prose.
- M2 (S3): move the formal paragraph (261-266) directly after the intro;
  smooth the ℓ≤0 appositive: "Paths with ℓ ≤ 0, i.e. unreachable pairs and
  trivial self loops, carry no structural reasoning signal and are excluded
  from training and evaluation."
- M3 (S2): merge the two PAR statements into one definition + one role:
  the concept-only configuration 𝒢_concept "is the Path-Aware Reasoning
  (PAR) configuration, the topology-only counterpart that isolates the
  contribution of modality-mediated connectivity."
- M4 (S4): append the structural floor to the configuration paragraph:
  "The hop bound is not a free knob either: a creation edge names exactly
  one artist, so no artwork node connects two distinct artists, and any
  artwork-mediated route between two artists takes at least three hops."
  (This also feeds §4.5's empirical K=2 result without duplicating it.)
- M5 (S3): leakage paragraph moves last, bridged as an extraction-integrity
  close: "Beyond configuration, extraction integrity is protected against
  leakage: …" — content unchanged; optional half-clause "traversal through
  the target node itself remains allowed; only the query edge and its
  inverse are masked" if the user wants the precision (default: omit).
- Net: −3 to −4 lines.

**EXECUTED 2026-09-20** with two user decisions overriding the draft:
(a) L-WORD BAN (new standing paper-wide rule): no "leak/leakage" anywhere
— skimming reviewers may reject on the word alone; the extraction facts
are stated as neutral protocol description, implicit. The paragraph now
opens "Extraction itself follows the evaluation protocol: every probe
masks the query edge together with its inverse in both orientations
before the search begins..." and closes the section; NO sibling-facts
precision clause (implicit-only directive). (b) The structural three-hop
reason stays OUT of Method (too dataset-bound) → §4.5 carries it inline
(see §3.10 M2 note). M2/M3 applied as drafted (formal para after intro,
smoothed appositive, single PAR definition, "this pairing is the only
coherent control" opener); \paragraph heading deleted; old formal block
deleted (leakage/extraction paragraph automatically became the section
close). Net §3.3 ≈ −3 lines.

### 3.4 §3.4 PAMR Architecture (lines 268-340)
Current: 3.4.1 Token Interleaving (271-297); 3.4.2 Transformer Encoder
(298-324); 3.4.3 Readout and Scoring (326-340).

Seams:
- S1 (V4): N_max(ℬ) is used in 3.4.2 (310-311) but defined only in §3.5
  (365-366) — a forward dependency the reader hits twice.
- S2: 3.4.1 never anchors its tokens to §3.2's embeddings ("obtained from
  the unified entity embedding via a learnable projection", 276-277 — which
  projection? see V8 conditional issue, resolved in §3.2's M2).
- S3 (V5): 3.4.3 presents the MLP output as the score. False per code.
- S4 (V8): the init sentence (339-340) is both wrong for the final layer and
  implementation trivia; the activation in the ŷ formula is wrong (GELU vs
  ReLU).
- S5 (V1): readout mode never committed.

Moves:
- M1 (S1): define at first use in 3.4.2: "…for paths shorter than the batch
  maximum, N_max(ℬ) = max_{p∈ℬ}(ℓ_p + 1) entity tokens." §3.5 then keeps
  only the reference (see §3.5 M3).
- M2 (S2): open 3.4.1 with the path anchor and the §3.2 tie: "Each extracted
  path (Section 3.3) is encoded as a flat sequence: entity tokens carry the
  unified embeddings of Section 3.2 through the projections defined there,
  interleaved with edge tokens, a classification token, and the query
  relation." Then the existing Eq. (1) machinery unchanged. (±0)
- M3 (S3+S4, the factual rewrite): replace the readout/scoring paragraph's
  tail with the residual contract, generically:
  "A two-layer MLP maps the aggregated representation to a path residual Δ
  (ReLU activation, dropout p = 0.1), and the path's score adds this
  residual to the frozen triple score g(s, r, ô) of Section 3.2: ŷ = g(s, r, ô) + Δ.
  Probes for which extraction finds no path keep the frozen score plus a
  learned per-relation offset. The residual output layer starts at zero, so
  untrained PAMR reproduces the frozen scorer exactly." — then DELETE the
  Xavier/zero-bias trivia sentence (S4) to pay for it. Net ≈ +2 lines.
  (If the user prefers, the no-path clause moves to §4.3 only; see §6.)
- M4 (S5): commit the readout in the same paragraph or §4.1: "PAMR uses the
  mean-pooling readout in all reported runs." (Pending user confirmation, V1.)

**EXECUTED 2026-09-20** with both open decisions resolved by the user:
readout = **mean pooling CONFIRMED** (§3.4.3 now aggregates "by mean
pooling over valid tokens"; no §4.1 sentence — single home); no-path
clause home = **§3.4.3 only** ("Probes for which extraction finds no
path keep the frozen score plus a learned per-relation offset."). M4's
final form: residual contract ŷ = g(s, r, ô) + Δ with ReLU + dropout
0.1, anchored via \ref{sec:embedding-module}; Xavier/zero-bias init
sentence DELETED (generic-paper constraint; also was wrong for the
final layer). M1-M3 applied as drafted (path anchor opening with
conditional projection phrasing; N_max defined at first use in 3.4.2;
GELU/Post-LN scoped to "every encoder layer"). Notation sweep clean:
g intact in §3.2, Δ collision-free, 𝐯_j-vs-unified-𝐯 relation stated,
GELU(encoder)/ReLU(readout) split factual. Owed notes: §3.5 defines ŷ±
via the contract by reference (its S± wording still says "path scores");
§4.3 grounds "every candidate object" via \ref{sec:readout}; §4.1
dedups its N_max/padding back-reference to §3.4.2 (its old
sec:paired-sampling pointer is stale). Round cost +4 wrapped source
lines (M1's fuller opening); printed impact ≈ +2.

### 3.5 §3.5 Training (lines 342-385)
Current: orphan intro (346-349); 3.5.1 Paired Sampling (350-368); 3.5.2 Loss
Function (370-385).

Seams:
- S1: the two-sentence intro is an orphan; "guarantees both balance and
  fairness" is vague, and "fairness" never returns.
- S2 (V4): 3.5.1 re-derives padding ("Variable length paths are zero-padded
  to the batch maximum N_max(ℬ) = max_{p∈ℬ}(ℓ_p+1) entity tokens, with
  padding masks 𝐌 as defined in Section 3.4.1", 365-368) — duplicates §3.4.2
  and points at 3.4.1 where the mask is only introduced (the mask is
  specified in 3.4.2's paragraph, paperv4:310-311).
- S3: the loss closing "This loss directly optimises the ranking objective
  and prevents score collapse." (384-385) dangles; "score collapse" is
  unexplained and disconnected from the intro's equal-gradient motivation.
- S4: negatives without extracted paths silently exist (V5/V6); the loss
  text pretends every probe has a path.

Moves:
- M1 (S1): merge the intro's motivation into 3.5.1's first sentence:
  "Supervision must stay stable regardless of how many negative paths a
  query contributes, so training pairs each positive probe with the query's
  negatives and guarantees a balanced batch." Then delete the orphan
  paragraph. (−1 line)
- M2: keep the "one positive + |S−(q)| negatives" and "every negative is
  visited exactly once per epoch" sentences (verified true:
  `pamr.py:475-497`); no change.
- M3 (S2): replace the padding sentence with: "Variable-length paths are
  zero-padded to N_max(ℬ) with the padding masks of Section 3.4.2." (−1 line)
- M4 (S3): replace the closing with the stitch that closes the loop the
  intro opened: "The per-query normalisation 1/|S⁺(q)||S⁻(q)| is what keeps
  every query's gradient contribution independent of its negative count;
  the shared margin keeps positive and negative scores apart." (±0)
- M5 (S4): one clause in the S± definition: ŷ± are candidate scores, and
  "a negative probe without an extracted path contributes through its frozen
  score" — half a line, ties to §3.4.3's no-path sentence (keep whichever
  home the user picks; see §6).

**EXECUTED 2026-09-20** (no open questions — all pre-decided). Final
forms, superseding the M-drafts above where noted: M1 final = keep the
lead-in sentence only, DELETE intro sentence 2 ("We construct a paired
sampling dataset that guarantees both balance and fairness.") —
"fairness" gone, balance claim lives in 3.5.1's own sentence. M2 final =
"Variable-length paths are zero-padded to the batch maximum N_max(ℬ)
with the padding masks 𝐌 of Section~\ref{sec:transformer-encoder}"
(single N_max formula now lives in §3.4.2 only; mask ref corrected to
its actual defining paragraph). M3 final = S± sentence gains "each ŷ
scored under the contract of Section~\ref{sec:readout}" — the no-path
case stays in §3.4.3 only (single-home, Q2 decision); M5's extra clause
NOT added here. M4 final = closing stitch "The per-query normalisation
makes every query's gradient contribution independent of its negative
count, and the margin keeps positive and negative scores apart."
("score collapse" gone). Eq. (4) and all symbols untouched (code-verified
earlier). Owed notes discharged: §3.5 S± tie done; padding dedup done;
§4.1's stale sec:paired-sampling pointer still owed in its round. Round
cost ≈ 0 wrapped lines (paper 657).

### 3.6 §4.1 Implementation Details (lines 389-421)
Current: hardware/software (392-395); `\paragraph{Path extraction.}`
(397-401); `\paragraph{Training data and GPU memory.}` (403-408);
`\paragraph{Training configuration.}` (410-421).

**Transferred from §3.2 (2026-09-20 execution) — owed in THIS round:**
§3.2 no longer names VL-KGE or cites it (generic MMKGE there), so the
frozen-model instance lives here and ONLY here inside Method/Experiment:
1. Rewrite the "auto-detected" sentence factually but generically, e.g.:
   "The frozen MMKGE model is VL-KGE~\cite{efthymiou2026vlkge} with TransE
   embeddings ($D = 768$), and $d = 768$. The VL-KGE base model remains
   frozen during the training of PAMR." — keep TransE/D=768 (factual
   instance, implementation section is the right home), drop "auto-
   detected" and any backbone-explanation flavor (user decision 2026-09-20).
2. Terminology alignment: say "MMKGE model" where §3.2's term is used;
   plain "KGE" stays the family term (§3.1, captions).
3. Readout commitment (V1) lands in §3.4.3, NOT here (single home).

**EXECUTED 2026-09-20** — transfer discharged. Final forms: M3 instance
sentence = "The frozen MMKGE model is VL-KGE~\cite{efthymiou2026vlkge}
with TransE embeddings ($D = 768$), and the token width is $d = 768$;
the VL-KGE base model remains frozen during the training of PAMR."
("auto-detected" gone; instance + the only in-Experiment citation now
live here). M1 applied with the §3.5 tie + GPU hinge; the padding
clause DROPPED entirely (single-home: defined §3.4.2, referenced §3.5);
stale sec:paired-sampling pointer gone. M2: NEW factual catch — "cached
on disk as pickle files" was WRONG (pipeline writes per-split CSV path
tables; preprocess.py docstring + PAMRDataset/paths_csv) → "cached on
disk for reuse" (format claim dropped). PLUS new user word ban:
"contract" is inapplicable in academic context — the §3.5 S± phrase
"scored under the contract of" became "scored as defined in"; `grep -i
contract` on paperv5 must stay ZERO (added to §4 banned list). Consistency
sweep: η/λ/m/B/N/h/d′/dropout all match §3.4/§3.5; D=d=768 consistent
with §4.4. Round cost ≈ 0 wrapped lines (paper 657).

**REDUCTION PASS 2026-09-20 (user directive: strip machinery trivia):**
(a) headings \paragraph → \subsubsection (uniform with Method; Title
Case, no periods); (b) Path Extraction: GCC 13.2/C++23 versions dropped,
adjacency/8-threads/merge machinery dropped → "The bidirectional BFS is
implemented in C++ and called from Python; extraction is
multi-threaded, and each split's path table is cached on disk for
reuse." (c) Training Data and GPU Memory: "per-query pairing across CPU
cores" dropped (pairing = §3.5's home), memory-manager trivia
("expandable segments", "periodic cache defragmentation") dropped →
"On the GPU side, the frozen embedding table dominates memory." — note
OWED for §4.5's round: its "GPU memory is dominated by the frozen
embedding table" now duplicates §4.1's instance fact (single-home
check needed there); also §4.5's "compressed adjacency" /
"byte-deterministic" phrasing is the same machinery flavor — reduce in
that round. (d) Training Configuration: "(16 samples per batch with the
guaranteed balanced positive/negative composition)" → "(16 samples per
batch)" (rationale duplicated §3.5). Hardware/software sentence KEPT
(machine + version stack = reproducibility level, not machinery). Paper
653 lines (net −2 vs paperv4).

Seams (historical, pre-execution):
- S1: "Training data and GPU memory" merges two topics with no hinge.
- S2: wrong back-reference — "the padding masks of Section
  \ref{sec:paired-sampling}" (407) points at the *user* of the mask, not its
  definition (3.4.2).
- S3: the paragraphs read as four bolted-on inserts (per-round artifacts).

Moves:
- M1 (S1): open with the §3.5 tie and a memory hinge: "The paired dataset of
  Section 3.5 is rebuilt once per epoch from the precomputed path tables,
  with per-query pairing across CPU cores. On the GPU side, the frozen
  embedding table dominates memory, managed through expandable segments and
  periodic cache defragmentation after initialization." (±0)
- M2 (S2): fix the reference to Section 3.4.2 (or 3.4). (±0)
- M3 (S3): light connective tissue only ("The bidirectional BFS is
  implemented in C++ … and called from Python" keeps its paragraph; no new
  facts). Optional: readout commitment lives in 3.4.3, not here, to avoid
  splitting it.
- No length change.

### 3.7 §4.2 Datasets (lines 423-456) — split + correct two justifications
Current: one 25-line paragraph (426-450) assembled across rounds; BFS
settings paragraph (452-456).

Seams:
- S1: the paragraph carries four rounds of insertions (stats → protocol →
  excluded relations → the four relations → benchmark uniqueness → leaf
  argument → future work) and needs a real joint.
- S2 (V3): "Three relation types introduced as preprocessing inverses
  (isRelatedToArtwork, isRelatedToArtist, hasCreatedArtwork) are excluded
  from evaluation" — wrong reason for two of the three (README: similarity
  relations with separate retrieval metrics; only hasCreatedArtwork is an
  inverse).
- S3 (V3): "query and answer differ in modality" — false (both artists are
  textual). The load-bearing justification for the relation choice and the
  "cross-modality" term needs a truthful, generic replacement.
- S4: terminology drift — "artist-only" (454) vs "concept-only" (455-456
  and §3.3).

Moves:
- M1 (S1): split at the natural joint into (a) dataset + protocol (stats,
  modalities, negatives, chronological split) and (b) evaluation-relation
  choice, bridged by: "The choice of evaluation relations follows from the
  graph's construction." (±0)
- M2 (S2): replace the exclusion reason: "Three relation types are excluded
  from evaluation: the two similarity relations isRelatedToArtwork and
  isRelatedToArtist, which the dataset's original evaluation treats with
  separate retrieval metrics, and hasCreatedArtwork, the preprocessing
  inverse of isCreatedByArtist." (Pending user wording check, V3.)
- M3 (S3): replace the differ-in-modality clause with the path-level
  definition (generic): "…and they are the only relations whose endpoints
  are both conceptual artist nodes while the evidence connecting them can
  be carried by visual artwork nodes — exactly where modality-as-entity
  path structure carries multimodal information." Two wording options are
  in §6; the winner must propagate to the abstract/§4.4/caption. (±0)
- M4 (S4): "artist-only" → "concept-only". (±0)

**EXECUTED 2026-09-20** — TWO USER CORRECTIONS supersede the M-drafts and
the V3 evidence framing above. **(1) README DEPRECATED:** the dataset
README is VL-KGE-era documentation, outdated for PAMR — NOT an evidence
source (background only); PAMR facts must be verified against PAMR-era
code/configs or the user. **(2) No exclusion narrative:** nothing is
"excluded from evaluation"; the truth is preprocessing-side — inverse
relations are generated for UNIDIRECTIONAL relation types, while
isRelatedToArtwork/isRelatedToArtist are BIDIRECTIONAL and need none
(config-consistent: inverse_relation_pairs covers only the two artist
pairs). hasCreatedArtwork is entirely irrelevant and is now ABSENT from
the paper. **(3) Relation choice = comparability rationale (user):** the
four artist relations are chosen because they can be evaluated
consistently under both the concept-only and the modality-as-entity
settings, so PAR-vs-PAMR demonstrates the modality-as-entity benefit
over the classic formulation ("A-MMKG/N-MMKG settings" rendered with
the paper's configuration terms). The falsifiable "differ in modality"
and "only cross-modality" clauses are GONE. **(4) Term resolution:**
"cross-modality" purged paper-wide; the three outer occurrences became
GENERIC per user ("mentioning artist stuff is too specific"): abstract
"over its four evaluation relations", caption "the same four evaluation
relations", conclusion "four evaluation relations". M1 split applied
(stats/protocol para vs evaluation-relations para; the planned bridge
sentence dropped — the preprocessing fact is the hinge). M4 applied.
PLUS the paper-wide semicolon sweep (see §4 item 7) landed in this
subsection too. Paper 656 lines. AUDIT NOTE: §4.5's inline three-hop
reason ("a creation edge names exactly one artist") derived partly from
the now-deprecated README — empirical K=2 result stands on sweep data;
the structural phrasing awaits the user's one-word confirmation against
PAMR-era data (or soften to empirical-only).

### 3.8 §4.3 Evaluation Metrics (lines 458-466)
Current: one paragraph; candidate set = gold + 10 negatives (reconciled with
review #9 already), descending rank, MRR + Hits@k overall and per relation.

Seams:
- S1 (V6): the aggregation rule for multi-path candidates is unstated (the
  reviewer's open sub-question, now answerable).
- S2: no link to how no-path candidates enter the ranking (V5/V6) — without
  this, "assigns a scalar score to every candidate object" quietly assumes
  every candidate has a path (they do not: reach 99.4% at the operating
  point).

Moves:
- M1 (S1+S2): add two generic sentences after the first sentence: "When
  several extracted paths reach one candidate, the candidate ranks by the
  maximum of their path scores; candidates without an extracted path keep
  the frozen triple score of Section 3.2 plus its learned offset (Section
  3.4.3)." (+2 lines)
- M2: tie "per relation type" forward: "the per-relation breakdown used in
  the discussion below." (±0)

**EXECUTED 2026-09-20** (no open questions). Final form of the added
sentence — semicolon rule compliant and single-home honoring: "When
several extracted paths reach one candidate, its score is the maximum
over their path scores, and candidates without an extracted path are
scored as defined in Section~\ref{sec:readout}." — the no-path case is
REFERENCED (§3.4.3 single home), not restated; max rule stated once
here (closes wip #6(a) in the paper). M2 DROPPED with rationale: the
per-relation tie lives in §4.4's round as a backward
\ref{sec:evaluation-metrics} instead (leaner §4.3). First/MRR-Hits
sentences untouched. Paper 660 lines.

### 3.9 §4.4 Results (lines 468-544)
Current: baselines paragraph (471-478); Comparative Study intro (482-494);
Table 1 (496-523); ordering paragraph (525-530); multimodal-gain paragraph
(532-544).

Seams:
- S1: "PAR uses only structural paths and PAMR additionally exploits
  multimodal token representations" (492-494) repeats the §4.4 intro's
  PAR-as-structural-baseline statement (477-478) without adding anything.
- S2: the ordering paragraph (525-530) and the multimodal-gain paragraph
  (532-544) are two inserts with no hinge between them.
- S3: the crossover explanation (536-540) speculates ("possibly because…",
  "probability spreading") where the mechanism is now known (V5).
- S4: the per-relation sentence (541-544) sits tacked onto the crossover
  paragraph; §4.3 now promises a per-relation breakdown it should deliver.

Moves:
- M1 (S1): replace with the differential/design sentence (§2 item 2):
  "PAR and PAMR share the frozen backbone and the same residual scorer; they
  differ only in whether extracted paths may leave the conceptual subgraph."
  (±0)
- M2 (S2): bridge with one sentence: "This band structure frames the
  multimodal question: does leaving the conceptual subgraph pay?" (±0;
  alternative: merge the two paragraphs if the compile shows slack.)
- M3 (S3): replace the speculative cause with the mechanical one, keeping
  the observed crossover: "The crossover traces to the residual design: the
  frozen TransE score preserves the head of the ranking, while the path
  residual reorders candidates whose multimodal evidence disagrees with the
  triple score — trading some top positions for depth of recall." (±0)
  If adopted, §5's "possibly because eight encoder layers are not enough"
  speculation can stand unchanged (it is a separate, hedged hypothesis) —
  or be softened in the same spirit; see §6.
- M4 (S4): keep the per-relation sentence but open it with the §4.3 tie:
  "Per relation (Section 4.3's breakdown), PAMR leads most metrics on…" (±0)
- "Computationally the trade is affordable (Section 4.5)" stays as the single
  forward reference (wip §13 rule).

**EXECUTED 2026-09-20 — DATA CORRECTION + full rewrite.** User report:
the TransE↔PAMR Hits@1/Hits@3 cells were WRONGLY PLACED in the table;
user accepted whole-cell swap (mean±std travel together, verified vs
average.csv avg+wstd pairing). **Table 1 now: PAMR bold on ALL FOUR
metrics, TransE underlined on all four** (H1: TransE 0.345±0.027 /
PAMR 0.387±0.044; H3: TransE 0.576±0.060 / PAMR 0.625±0.044). The
crossover story DIED; M3's residual-mechanism draft superseded by the
all-metric narrative ("margin largest at strict cutoffs and narrows
toward Hits@5, consistent with the residual design...").
Per-relation sentence recomputed from the swapped assignment: PAMR
leads most metrics on isInfluencedOn and isPupilOf, split on
isInfluencedBy/isTeacherOf (isPupilOf H1 tie 0.394 survives). User
chose FULL CONSISTENCY propagation: conclusion limits/future-work
reframed honestly (false H1/H3-loss sentence + "deeper encoders" item
DELETED; contrastive objective kept with neutral framing; "next
benchmark"→"next setting" folded in), abstract echo ("and leads every
reported metric, with Hits@5 at 0.782"), contribution 3 ("surpasses
the strongest embedding baseline on every reported metric").
"Computationally the trade is affordable" → "the added reasoning is
affordable" (the trade no longer exists). **KNOWN-SWAP CAVEAT:
result.csv/average.csv per-relation H1/H3 columns for TransE↔PAMR are
KNOWN-SWAPPED (no re-run available) — never trust them for H1/H3
TransE/PAMR analysis; post-deadline B-queue: regenerate corrected
CSVs.** Review issue #4 (H1/H3 drop) effectively EVAPORATES — it was
based on the mislabeled numbers. M2's band bridge NOT applied
(ordering paragraph left unchanged — the all-metric paragraph now
follows it naturally). Paper 666 source lines, −24 raw words vs
paperv4 (printed ≈ −1; wrapping noise from long-line rewrites).

### 3.10 §4.5 Runtime and Scalability (lines 546-607)
Current: profiling setup (549-555); extraction costs + epoch times
(557-566); sweep analysis (568-585); figure (587-599); complexity orphan
(601-607).

Seams:
- S1: the closing paragraph opens "Scoring a candidate costs O(L²h)…"
  (601-603) immediately after the sweep paragraph ends on L = 13 (583-585) —
  same subject split across a paragraph break.
- S2 (V10): "(Section 3.1)" after "exactly three mediation hops" (571-574)
  cites a section that does not contain the reason; §3.3 now will (M4
  above).

Moves:
- M1 (S1): move the O(L²h) sentence up to attach to the L=13 sentence ("…
  exercise their full length budget of L = 13, and scoring one candidate
  costs O(L²h) attention over its token sequence"), leaving the final
  paragraph as bottlenecks + the Section 6 forward reference. (−2 lines)
- M2 (S2): change the parenthetical to "(Section 3.3)". (±0)
  **EXECUTED 2026-09-20** (superseded form, user decision: the structural
  reason moved HERE from Method): the "(Section 3.1)" anchor is replaced
  by the inline cause — "because no artwork node connects two distinct
  artists (a creation edge names exactly one artist), so any
  artwork-mediated route between two artists takes at least three hops" —
  explaining why K=2 yields zero paths. No dangling reference remains.
  **M1 + M3 EXECUTED 2026-09-20** as drafted (O(L²h) merged into the
  L=13 sentence; final paragraph = latency + bottlenecks + Section 6
  forward ref). **PLUS the flavor-reduction pass (user's §4.1-derived
  directive):** "compressed adjacency" and "byte-deterministic" dropped
  → "The worker parses the split graph once and is deterministic across
  runs"; "Graph loading and adjacency construction" → "Loading the
  graph"; the meta-sentence "and the shape carries more information
  than a plain budget" DELETED. **Single-home resolved: §4.5's "GPU
  memory is dominated by the frozen embedding table" sentence REMOVED —
  §4.1 ("Training Data and GPU Memory") is the single home for that
  fact**; §4.5 keeps latency/bottleneck scaling only. Caption run-
  condition furniture (deterministic worker, threads, masking) kept.
  Paper 665 lines.

### 3.11 §4.6 Case Study (lines 609-635)
Current: two-sentence intro (612-615); figure (617-627); PAR paragraph
(629); PAMR paragraph (631); ranking/kinship paragraph (633); auditability
paragraph (635).

Seams:
- S1: the intro is a two-sentence orphan before the figure.- S2: "Compared with the purely conceptual graph" (635) — nonstandard term
  for the established "concept-only configuration/graph".
- S3: the auditability close restates contribution 3's caveat but does not
  visibly close the loop with it.

Moves:
- M1 (S1): merge the intro into the first analysis paragraph after the
  figure: "Figure 4 shows where multimodal entities matter most: the test
  query (Pablo Picasso, isInfluencedOn, ?) with two candidates, Salvador
  Dali and Lee Krasner. Restricted to artists, PAR's bidirectional BFS can
  only walk through colleague relations: …" (−1 line)
- M2 (S2): "compared with the concept-only graph". (±0)
- M3 (S3): echo contribution 3's framing verbatim enough to be felt:
  "Although the learned scorer itself is not inherently transparent, its
  judgement here rests on paths a reader can audit — which artworks the
  route visits, and what each visit contributes." (±0, replaces the first
  sentence of 635 which already says nearly this in a stitched way.)

**EXECUTED 2026-09-20** (no open questions). M1 applied with two
standing-rule adaptations: the merged intro keeps BOTH original
sentences ("Figure X shows where multimodal entities matter most. The
query (…) is taken verbatim…") — the draft's colon form
("matter most: the test query") was rejected by the colon rule, and the
two-item "to compare: Salvador Dali and Lee Krasner" colon is kept as a
legitimate listing; the standalone orphan block is deleted and the
figure block remains a float before the merged paragraph. M2 applied
("compared with the concept-only graph"). M3 applied with wording
aligned to contribution 3 VERBATIM ("Although the learned Transformer
scorer itself is not inherently transparent…") — the draft's em-dash
form was banned; the existing colon enumeration after "audit" is kept
(two-item listing). Path narrative (631→635) untouched. Paper 663
lines.

### 3.12 §5 Conclusion (lines 637-642)
Current: summary + traceability (640); limits + future work (642).

Seams:
- S1: "Extending the modality-as-entity formulation to dynamic or temporal
  modalities is the natural next benchmark" (642) — a formulation is not a
  benchmark.
- S2: if §4.4 adopts the residual mechanism (M3 there), the limits sentence
  "possibly because eight encoder layers are not enough to sharpen the head
  of the ranking" can stay (it is a separate hedged hypothesis) — but a
  minimal alternative is to fold it into the residual framing. Default:
  leave unchanged.

Moves:
- M1 (S1): "…is the natural next setting." (±0)
- M2: optional one-clause recognition of the residual design in the summary
  sentence ("structural paths are scored by a Transformer over frozen KGE
  tokens" → "a Transformer over frozen KGE tokens augments the frozen triple
  score with a path residual") — only if the user wants the design named in
  the conclusion; default off (generic-paper constraint).

**EXECUTED 2026-09-20.** M1 was folded into the §4.4 data-correction
round ("next benchmark" → "next setting" landed there, together with
the limits/future-work rewrite that superseded the S2 options above).
This round's final move: the consistency echo "attained the best MRR
on" → "led every reported metric on" (mirrors the abstract and
contribution 3; the understating phrase is gone paper-wide — 3
occurrences of "every reported metric" now align). M2 RESOLVED BY
USER: keep the generic summary sentence ("structural paths are scored
by a Transformer over frozen KGE tokens" unchanged). Para 2 final from
§4.4's round; credits/AI-disclosure untouched. **This closes ALL
per-subsection rounds (§3.1 → §5) and ALL five §6 open decisions.**
Paper 663 lines.

---

## 4. Cross-cutting passes (after all subsection edits)

1. **Terminology grep (must be clean):** `artist-only`, `purely
   conceptual`, `monolithic`, `leverages`, `not only`, `contract`,
   `cross-modality` (now purged paper-wide per §4.2 round),
   em-dashes,
   `---` in prose. Also confirm a single spelling of "concept-only" across
   §3.3/§4.2/§4.4/§4.6. **NEW standing rule (2026-09-20, user decision):
   `grep -i leak` over paperv5 must return ZERO — the word is a
   rejection trigger for skimming reviewers; extraction-protocol facts
   are stated neutrally and implicitly instead.**
   Also: cross-section references must be dynamic `\ref{}` — never
   hardcoded "Section x.y" (user rule 2026-09-20). **NEW standing
   heading rule (2026-09-20, user): uniform heading style — named blocks
   use `\subsubsection` everywhere (Method AND Experiment), Title Case,
   no trailing periods; `\paragraph` run-ins are banned. Applied to
   §4.1 (Path Extraction / Training Data and GPU Memory / Training
   Configuration); note: subsubsection headings take more vertical
   space than run-ins (≈ +3 printed lines), watch the page gate.**
2. **Notation audit vs wip.md §11:** K/k/δ/λ/D/d/d′/N/h/F/L/ℓ/𝐌/μ/
   𝒱_concept/𝒱_mod/ℛ_med/ℛ_concept; χ stays retired. The new 𝒱* in §3.2
   must not collide with anything (it does not; 𝒱* is §3.1's own symbol).
3. **Residual-contract consistency:** the formulation "frozen triple
   score $\mathcal{S}(s, r, \hat{o})$ + path residual $\tilde{y}$" must
   appear identically in §3.4.3, §3.5's loss context (candidate scores),
   and §4.3; §4.4 references it without redefining. (Updated for the
   notation round: g→𝒮, Δ→ỹ.)
4. **\ref audit:** every \label referenced at least once; the three-hop
   explanation lives inline in §4.5 (no dangling anchor; user decision
   2026-09-20 moved it out of Method).
5. **Length check:** `wc -w` / `wc -l` diff paperv4 vs paperv5; target net
   −2 to +2 lines; then the standing compile gate on prism (≤12 pages
   excluding references) decides whether any trim lever is pulled.
 6. **De-AI voice:** banned-word grep from wip.md §16; no new em-dashes or
    "not only…but also" constructions introduced by the rewrites.
 7. **Semicolon rule (NEW standing, user 2026-09-20):** semicolons only
    in listing contexts; non-listing uses get period splits or
    contrast rewrites ("while"). PAPER-WIDE SWEEP DONE 2026-09-20
    (§3.1 pointer, §3.1 constraint clause, §3.2 widths, §3.3 ablation,
    §4.1 ×2, §4.2 ×3, §4.4 ×4 + caption, §4.5 ×4 + figure caption,
    case study, conclusion; math argument separators `MHA(...; M)` and
    LaTeX comments exempt). Future rounds: no NEW non-listing
    semicolons; the §4.4/§4.5/caption period-splits below are already
    applied — their planned rounds keep only the substantive rewrites.
    NOTE: one caption edit briefly duplicated a bfs_sweep caption line
    (caught by read-back, fixed in the same round).
 8. **Colon rule (NEW standing, user 2026-09-20):** colons only for
    listings, equation introductions, and caption panel furniture;
    explanation-introducing colons ("X varies: explanation") are an
    AI-voice tell. PAPER-WIDE SWEEP DONE 2026-09-20 — 11 fixes (intro
    ×3, §2.2, §3.2 "Backbone width varies" sentence dropped to
    "Complex-valued backbones output...", §3.3 ×2, §4.4 residual-design
    clause, §4.5 ×2, conclusion "setting, in which..."). Kept as
    legitimate: §2.1/§2.2/§2.3/§2-close/§4.4-backbones/§4.4-ordering/
    §4.5-limits/case-study listing colons; all math/equation
    introductions; caption (b) panel colon; §3.4.1 flat-sequence
    enumeration; title. Future rounds: no NEW explanation-like colons.
  9. **Parenthesis rule (NEW standing, user 2026-09-20):** parentheses
     carry ONLY abbreviation introductions, math tuples, listings,
     `\ref` cross-refs, and caption attributions. Explanation-based
     parentheses (gains, conversions, clauses, semantics) become verbal
     prose (appositives, "i.e.", relative clauses). PAPER-WIDE SWEEP
     DONE 2026-09-20 — 7 fixes (abstract gain appositive; §3.4 mask
     semantics "whose entries"; §4.1 batch size "i.e."; §4.3
     candidate-set clause; §4.4 four gain-parentheticals → one lead
     gain + verbal percentages; §4.5 "i.e.," → "i.e." British
     normalization; bfs_sweep caption "(dashing)" → verbal). Kept as
     legitimate: all (A-MMKG)/(N-MMKG)/(PAMR)/(PAR)/(MRR)/(KGs)/(MMKGs)
     intros, $(s,r,?)$-type tuples, relation listings, §4.2 dataset
     breakdown "224,166 entities (216,564 artworks and 7,602
     attributes)" (enumeration-style), case-study "(by Dali)"
     attributions. Future rounds: no NEW explanation-like parentheses.

**GLOBAL PASS EXECUTED 2026-09-20 — COMPLETE, paper FINAL pending
compile.** (1) Terminology greps: ZERO hits for leak / contract /
monolithic / leverages / not only / crossover / cross-modality /
artist-only / purely conceptual / pickle / auto-detect / fairness /
score collapse / deeper encoders; zero em-dashes in prose. (2)
Notation audit: PASS — χ retired (0); λ = AdamW weight decay only (1
use, §4.1); decay = δ everywhere; Hits@k uses $k$; zero hardcoded
"Section x.y"; single N_max formula (§3.4.2). (3) \ref audit: 21 refs
resolve, ZERO dangling; 19 zero-reference labels (4 equation + 15
section, incl. sec:overview) LEFT AS-IS by user decision — harmless,
standard practice, audit intent (no dangling refs) satisfied. (4)
Length ledger: paperv4 655 lines / 5255 raw words → paperv5 663 lines
/ 5203 raw words = **−52 raw words** (delta in lines = long-line
re-wrapping only); no double blanks. (5) §4.5 structural three-hop
reason CONFIRMED by user against PAMR-era data (stays as-is). Every
standing rule set in this session is currently satisfied. NEXT: user
compiles on prism (paperv5.tex + refs.bib + figures/{overview,case,
bfs_sweep}.svg; pdflatex → bibtex → pdflatex ×2; ≤12 pages excl.
refs); trim levers if over: \tabcolsep 2pt→1.5pt, Related-Work
second-paragraph squeeze (~0.3 p), case-study trim. Post-submission:
KNOWN-SWAP CSV regeneration + B1-B4 queue + camera-ready repo decision
(wip §16).

**NOTATION ROUND EXECUTED 2026-09-20 (user-specified re-specification,
applied to integral.md §7 FIRST, then paperv5 section-by-section).**
New system: $\mathcal{G}=(\mathcal{V},\mathcal{R},\mathcal{E})$ with
$\mathcal{E}\subseteq\mathcal{V}\times\mathcal{R}\times\mathcal{V}$;
direct N-MMKG via the total mapping
$\mathcal{M}:\mathcal{V}\to\mathbb{M}$,
$\mathbb{M}=\{\text{concept},\text{image},\text{text},\ldots\}$
(conceptual = $\mathcal{M}(v)=\text{concept}$); NO asterisks ($\mathcal{G}$ IS
the multimodal graph); $\mathcal{G}_c$ replaces
$\mathcal{G}_{\text{multi}}/\mathcal{G}_{\text{concept}}$; §3.2 reduced
to the frozen-model signatures $g_V:\mathcal{V}\to\mathbb{E}$,
$g_R:\mathcal{R}\to\mathbb{E}$,
$\mathcal{S}:\mathcal{V}\times\mathcal{R}\times\mathcal{V}\to\mathbb{R}$
(fusion/averaging/dimension bookkeeping ALL removed);
$\tilde{\mathbf{X}}$/$\mathbf{X}$ replace $\mathbf{T}$/$\mathbf{T}^{\text{in}}$;
$L(\mathcal{B})$ replaces $N_{\max}(\mathcal{B})$ (mask over
$2L(\mathcal{B})+1$; per-path length is just $2\ell+3$);
$S^{\pm}(q)$ unified roman (script $\mathcal{S}$ reserved for the
scorer); $p^{\pm}_q$ → $p_o$/$p_{o^-}$; $d'$ and the init distribution
$\mathcal{N}(0,0.02^2)$ removed from Method (d′ survives ONLY as the
§4.1 instance $d'=2048$; d defined in §3.4.1, value in §4.1);
residual $\Delta$ → $\tilde{y}$; positional $\mathbf{p}_i$ →
$\mathbf{e}_i$; §4.5 budget "$L=13$" → "$2K+3=13$",
$O(L^2 h)$ → $O(\ell^2 h)$; intro contribution 1 "unified embeddings"
→ "frozen embeddings" (definition moved out from under the term).
Retired list in §7 updated (17 entries); retired-symbol sweep over the
whole paper: ZERO hits; ℛ_med/ℛ_concept retained (user spec silent on
relations; needed for constraints + query definition). Length:
663→629 lines, **−285 raw words vs paperv4** (5255→4970).
[Note: ℛ_med/ℛ_concept were later verbalized in the bib round — see
below; the "retained" statement here is historical.]

**BIB-INTEGRITY ROUND EXECUTED 2026-09-20 (user request: ℛ pruning +
bib correctness; this section is the COMPACTION FLUSH for this round —
read it first after any compaction).**
1. ℛ_med/ℛ_concept verbalized (user chose "verbalize both" after the
   count was corrected: ℛ_med = 1 use, ℛ_concept = 2); §3.1 now has
   ZERO subscripted relation sets; §7 row retired.
2. **Backbone citations added** at §4.4: TransE~\cite{bordes2013translating},
   RotatE~\cite{sun2019rotate}, DistMult~\cite{yang2015embedding},
   ComplEx~\cite{trouillon2016complex} (were named uncited while the
   canonical entries sat unused — the one genuine mismatch found).
3. **Fabricated references DISCOVERED and RESOLVED** (verification via
   arXiv export API + Crossref API + DuckDuckGo HTML; Semantic Scholar
   and DBLP were rate-limited/bot-walled — usable sources: export.
   arxiv.org/api/query, api.crossref.org/works?query.bibliographic=,
   html.duckduckgo.com/html/?q=):
   - zhu2022genkgc was FULLY FABRICATED (fake arXiv 2210.02115 = astro
     paper; fake "Zhu, Y and others"). Real GenKGC = "From
     Discrimination to Generation: Knowledge Graph Completion with
     Generative Transformer", WWW 2022, doi 10.1145/3487553.3524238,
     authors Xie/X Zhang/Li/Deng/H Chen/F Xiong/M Chen/Huajun Chen.
     KEY KEPT (zhu2022genkgc) — key names are historical IDs, not
     author-accurate.
   - tang2020orthogonal author list was WRONG + truncated ("Ding,
     Guanghui and Zhao, Kyunghao and others"); real OTKGE (arXiv
     1911.04910, ACL 2020): Tang/Huang/Guangtao Wang/Xiaodong He/
     Bowen Zhou.
   - xu2022multimodal: real title "Relation-enhanced Negative Sampling
     for Multimodal Knowledge Graph Completion" (order differs!), 5
     authors Derong Xu/Tong Xu/Shiwei Wu/Jingbo Zhou/Enhong Chen, doi
     10.1145/3503161.3548388 (Crossref).
   - chen2022mkgformer: real title "Hybrid Transformer with Multi-level
     Fusion for Multimodal Knowledge Graph Completion", real doi
     10.1145/3477495.3531992 (bib had .3532042 — WRONG), 9 authors
     (arXiv 2205.02357).
   - wang2021star: real title "...Representation Learning for Efficient
     Knowledge Graph Completion", doi corrected 10.1145/3442381.3449830
     → 10.1145/3442381.3450043, 6 authors (Crossref).
   - wang2019coke: full 9-author list via arXiv:1911.02168.
   - **3 entries were UNVERIFIABLE ANYWHERE and REMOVED** (user chose
     drop/swap): wang2019transae (fake arXiv 1910.06604 = math paper),
     wang2021rsme (no trace anywhere), chen2023csprompter (its arXiv
     2303.03922 = KGTransformer, a DIFFERENT paper). Paper edits:
     §2.2 "IKRL first aligned..." (TransAE dropped), "Later systems...
     OTE..., and MMRNS biases negative sampling with multimodal
     relations" (RSME clause dropped); §2.3 "CSPrompter bridges
     symbolic logic and large language models through structure
     prompting" → "KGTransformer transfers pretrained symbolic
     structure across KG tasks through prompt tuning" with NEW KEY
     zhang2023kgtransformer (entry = Structure Pretraining and Prompt
     Tuning for Knowledge Graph Transfer, WWW 2023, 8 authors, no DOI
     recorded to avoid guessing). The 3 fabricated entries DELETED
     from refs.bib (distinct from the 14 verifiable uncited leftovers,
     which the user chose to LEAVE).
4. Final bib state: 52 entries, 38 cited (all resolve, zero "and
   others" among cited), 14 uncited leftovers kept per user choice.
   §2.2 wording notes: MMRNS description now "biases negative sampling
   with multimodal relations" (attention claim was unverifiable).
   paperv5 = 635 lines.
5. Process lesson (add to wip): the AI-era bib entries were partly
   fabricated — every "and others" entry and every arXiv-ID-bearing
   entry must be verified against arXiv/Crossref before trusting;
   keys may be kept after corrections (historical IDs), but fabricated
   entries are deleted outright.

**VERIFY.MD-COMPARISON ROUND EXECUTED 2026-09-20** (user supplied
assets/analysis/verify.md — a Gemini audit in TWO PASSES: a
preliminary search/classification pass, then a user-prompted
per-entry verification pass demanding correct bibtex or "cannot be
found"; this block records the comparison and the final refs.bib
state). Verdicts per entry (primary sources = Crossref/arXiv direct
fetches; both Gemini passes are secondary):
- zhu2022genkgc: Gemini's PROMPTED pass hallucinated an EMNLP-2022-
  Findings record — its anthology link (2022.findings-emnlp.118)
  resolves to "Investigating Ensemble Methods for Model Robustness..."
  (different paper); ALSO its preliminary pass miscited 2303.03922 as
  "differential privacy" when it is KGTransformer. OUR fix stands:
  WWW 2022 Companion, doi 10.1145/3487553.3524238, 8 authors
  (Crossref-confirmed, matches arXiv:2202.02113 exactly).
- liu2019mmkg: BOTH of us were partially wrong — the bib had two
  fabricated authors ("Padia, Diego"/"Bogdanov, Antony") + mangled
  names, while Gemini's 4-author list omitted two REAL authors.
  Crossref DOI record (authoritative): Liu, Ye / Li, Hui /
  García-Durán, Alberto / Niepert, Mathias / Oñoro-Rubio, Daniel /
  Rosenblum, David S. APPLIED + series/volume (ESWC LNCS 11503).
- zhang2019quaternion: mangled "Linayao, Aston" → real NeurIPS'19
  roster Zhang/Tay/Yao/Liu + pages 2735-2745 (uncited leftover,
  hygiene).
- efthymiou2026vlkge: Gemini's bibtex matches ours; publisher = {ACM}
  added. DOI resolves via Crossref (Gemini's "cannot verify" was
  wrong).
- wang2021rsme: Gemini's passes CONTRADICT each other (part 1: "KBS
  vol 234 2021"; prompted pass: "cannot be found"); my Crossref KBS
  search finds nothing — the user's earlier removal stands.
- wang2019transae / niu2025survey: unverifiable (transae already
  removed; niu is a harmless uncited leftover).
Final refs.bib: 52 entries, 38 cited (all resolve, zero "and others"
among cited), 14 uncited leftovers. **Main tex: NO adjustments needed**
— every affected key (liu2019mmkg, efthymiou2026vlkge) is cited with
wording independent of author lists; §2.3's GenKGC description matches
the arXiv-verified method. LESSON (strengthened): even prompted
secondary-AI verifications must be checked against primary records —
verify.md produced two distinct hallucinations (the anthology page;
the "differential privacy" DOI content) that direct Crossref/arXiv
fetches disproved. Crossref/arXiv API remain the only authority.

**POST-VERIFICATION ADJUSTMENTS 2026-09-20** (paperv5a diff audit +
user decisions): the round's edits all mapped to the request or its
consequences; two judgment calls resolved by user — (1) contribution 1
REVERTED to "unified embeddings provided by VL-KGE" (paperv5a wording),
re-anchored by a new §3.2 sentence: "Both $g_V$ and $g_R$ are learned
unified representations that project multimodal entities and relations,
whatever their modality, onto the universal space $\mathbb{E}$" (the
word "unified" now appears exactly twice: contribution 1 + §3.2);
(2) §3.1's contrast clause reworded with the new symbol $\mathcal{X}$:
"whereas the attribute-based formulation (A-MMKG) keeps multimodal
features $\mathcal{X}$ attached to individual entities" (𝒳 ×1;
N-MMKG focus preserved). §7 updated: 𝒳 row added; g_V/g_R meanings
note the unified-representation property. Paper 632 lines.

**PAGE-PRESSURE ROUND 2026-09-20 (user; paper at 12.5 pages):**
(a) Table 1 body footnotesize → **scriptsize**; (b) ALL captions →
footnotesize via the new preamble line `\captionsetup{font=footnotesize}`
(after subcaption; applies to the 3 figure captions + Table 1);
(c) runtime caption ISOLATED — run-condition parenthetical
(validation split / 202 queries / 2,222 probes / two CPU threads /
own-edge masking / deterministic worker) omitted fully (ALL of it
already stated in §4.5's setup paragraph — zero information loss);
the two config-notes moved INTO the prose: error-bar definition as a
comma appositive on "84.5% ± 13.4" and "with coverage identical across
decay at $F \geq 50$"; caption now describes only the two panels;
(d) plot_bfs.py: (a)-legend title "±1 std over decay" → "±1 std";
BOTH panel titles' LaTeX `vs.\ ` backslashes removed ("vs. fanout",
"vs. memory"); (b) legends SWAPPED — δ-dashing now lower right,
F-markers now upper left (the old lower-right F legend overlaid the
δ=0.2 line-end cluster); bfs_sweep.svg REGENERATED (conda env ml;
CSVs deterministic, data byte-identical). USER TODO: copy the new
assets/bfs_sweep.svg into the compile workspace's figures/.

**PAREN-ROUND EXECUTED 2026-09-20 (user audit: explanation-based
parentheses → verbal prose):** audit found 6 offenders (all listed in
§4 item 9) against a furniture backdrop of ~30 legitimate parens; all
6 fixed + 2 consequential fixes (i.e. normalization; "while"-echo in
the Results rewrite → "with Hits@1 improving"). Results sentence now:
"MRR rises from 0.363 to 0.379, a 4.4\% gain, with Hits@1 improving by
12.2\%, Hits@3 by 8.5\%, and Hits@5 by 1.3\%." (versus-pairs dropped;
values live in Table 1/abstract; the strict-cutoffs sentence now reads
directly off 12.2 > 8.5 > 1.3). Battery after: zero "improvement)"
leftovers, zero "i.e.," outliers, banned-word greps clean, "unified"=2,
634 lines unchanged, headings intact (no swallow).

**GRAMMAR-TYPO ROUND EXECUTED 2026-09-20 → `paperv7.tex`** (user: fresh
part-by-part audit found 30 issues; scope=ALL; paperv5.tex kept as
frozen pre-round copy, paperv6 SKIPPED by user numbering, edits applied
to the paperv7 copy only). 32 edits: 8 objective fixes (self-loops,
dangling infinitive→"we require each query", positive-to-negative
hyphens, "To our knowledge," comma, "Computationally," comma, two
fronted-clause commas in §4.5, "84.5% ± 13.4%" unit); 9 long-sentence
splits/disambiguations (§3.1 whereas-chain, §3.3 so-and-therefore chain,
§3.4.1 "as well as" misparse fix, §3.4.3 MLP/score split, §4.2 ×2,
§4.5 K=2 triple-causality split, §4.3 clause already done in paren
round, conclusion "and, as…," split, "Together these measurements");
13 word-choice items (Moreover, let self-attention contextualise,
cheaper cross-encoding, however-position, extraction produces,
pretraining on, under the same protocol, the choice among the three
scoring functions, scale linearly, the same 2,208, $\delta = 0.2$, drop
"its", uninformative). CONSISTENCY CALLS: D1 initialization → "cost"
unified in all three spots (abstract/intro/RW-close; deadlock+dead end
retired); D2 "CLS token" → "learnable classification token" (caption;
matches §3.4.1/Eq. 1). ONE REFINEMENT vs literal ledger: §4.4
"the choice among them" → "the choice among the three scoring
functions" placed as "Beyond TransE, the choice among the three scoring
functions has limited impact on this graph." (avoids double "among").
Battery on paperv7: zero leftover old patterns, banned words zero,
"unified"=2, semicolons math-only, 635 lines (net +1 from splits),
headings intact.

**SWEEP-CLAIM + ABSTRACT-TAIL ROUND EXECUTED 2026-09-20 (paperv7):**
(a) "every reported metric" retired paper-wide — the self-referential
"reported" invites the reviewer flip "authors report only metrics they
win"; replaced by "all four ranking metrics" (numerically anchored to
Table 1's four rows: MRR + Hits@1/3/5) in 4 spots: abstract ("surpasses
it on all four ranking metrics"), contribution 3, §4.4 ("past the
strongest baseline on all four ranking metrics:"), conclusion ("led all
four ranking metrics"). User considered "surpasses it on most of the
remaining ranking metrics" — REJECTED after analysis: undersells the
true sweep, "most" is hedging-puffery, plants the "which metric lost?"
question. (b) Abstract final sentence deflaked: "Each prediction comes
with … so every answer stays traceable to actual artworks …" → "Every
prediction is accompanied by its extracted multimodal path, making the
answer traceable to artworks, artists, and shared attributes in this
experiment." (kills Each/every pair, "actual", "stays", "comes with";
"in this experiment" scopes the list to the dataset per user). USER
MANUAL ABSTRACT EDITS (sanctioned, logged): "a Transformer over
**learned** knowledge graph embedding tokens" (learned-vs-frozen drift
deliberate — future audits do NOT flag); "Separating path modelling…"
sentence removed from abstract ("initialization cost" now only intro
L75 + RW L122, D1 still holds); FUTURE RULE: abstract-region hunks in
paperv5↔paperv7 diffs are user-manual, not regressions. Battery: zero
retired phrases, "all four ranking metrics"×4, banned words zero,
"unified"=2, quantifier-pair zero, 634 lines, headings intact.

## 5. Execution order (when the revision is green-lit)

1. `paperv5.tex` already exists (byte-exact copy). Edit only paperv5.
2. §3.3 restructure first (M1-M5) — the biggest move; then §3.2 (M1-M3),
   §3.4 (M1-M4), §3.5 (M1-M5), §4.2 (M1-M4), §4.3 (M1-M2), §4.4 (M1-M4),
   §4.5 (M1-M2), §4.6 (M1-M3), §5 (M1), §3.1 (M1-M2) last.
3. Run the cross-cutting passes; diff against paperv4; report the length
   ledger to the user.
4. Append the round as §17 in `memory/wip.md` (evidence pointers, open
   decisions closed).
5. Hand the compile gate back to the user (prism; ≤12 pages excl. refs).

Rollback at any point: paperv4.tex is untouched.

## 6. Open decisions for the user (settle before or during execution)

1. **Readout commitment (V1):** confirm the reported checkpoint used the
   prod config's `readout: average` (mean pooling). Then the one-line
   commitment goes in.
2. **§4.2 wording (V3):** pick the replacement for "query and answer differ
   in modality":
   - Option A (path-level, keeps "cross-modality"): the endpoints are both
     conceptual artist nodes while the evidence connecting them can be
     carried by visual artwork nodes — the path crosses modality.
   - Option B (endpoints-level, drops "cross-modality" everywhere including
     the abstract — bigger change, not recommended before submission).
3. **Excluded-relations reason (V3):** confirm the README-based rewrite
   (similarity relations + one inverse) against `prepare.py`'s actual
   `add_inverse_relations` handling before finalizing the sentence.
4. **No-path clause home:** §3.4.3 (as drafted in §3.4 M3) or §4.3 only
   (M1) — keeping it in both places costs half a line; one home is enough.
5. **Conclusion residual mention (§3.12 M2):** default off.

Resolution log: the §3.2 projection question (former V8 follow-up) was
resolved 2026-09-20 as "generic conditional only" (no TransE instance,
no "auto-detected" flavor; §4.1 keeps the factual instance without the
flavor). §3.2's 𝒱/𝒱* resolved as generic-𝒱 + instance-𝒱*. §3.3's two
questions resolved 2026-09-20: masking-scope precision clause REJECTED
(new standing L-word ban instead — see §4 item 1); three-hop reason home
RESOLVED as §4.5-inline (Method stays generic). §3.4's two questions
resolved 2026-09-20 by user confirmation: readout = mean pooling
(item 1 CLOSED — prod config trusted), no-path clause home = §3.4.3
only (item 4 RESOLVED). Remaining open: §4.2 items 2 + 3; §5 item 5.
UPDATE 2026-09-20: items 2 + 3 RESOLVED BY USER (comparability
rationale; inverse-generation mechanism; hasCreatedArtwork dropped;
"cross-modality" purged with generic "four evaluation relations" in
abstract/caption/conclusion). Remaining open: §5 item 5 only. **NEW standing caveat: the dataset README is VL-KGE-era, deprecated as an
evidence source for PAMR facts.** §4.4 resolutions 2026-09-20: H1/H3
swap = whole cells (confirmed), propagation = full consistency
(abstract + contribution 3 echoes applied), future-work = reframe
honestly (deeper-encoders item dropped, contrastive reframed). NOTE:
review issue #4's rebuttal premise (H1/H3 loss) is void after the
correction. §5 item 5 RESOLVED 2026-09-20 (keep the generic summary —
user choice). **ALL §6 open decisions are now closed.**

---

## 7. Notation table (LIVING — update every round; review when all rounds finish)

Snapshot after the NOTATION ROUND 2026-09-20 (paperv5, 663 lines →
updated in the same round). The notation system was re-specified by the
user: direct N-MMKG (no asterisks), total mapping $\mathcal{M}$, 
embedding functions $g_V$/$g_R$ and scorer $\mathcal{S}$, token
sequence $\tilde{\mathbf{X}}$/input $\mathbf{X}$, batch length
$L(\mathcal{B})$, residual $\tilde{y}$, positional $\mathbf{e}_i$,
unified roman $S^{\pm}(q)$, paths $p_o$/$p_{o^-}$. Cross-checked
against the paper text itself; retired entries at the bottom.

### Graph & objective (§3.1)
| Symbol | Meaning | Defined | Used |
|---|---|---|---|
| $\mathcal{G}=(\mathcal{V},\mathcal{R},\mathcal{E})$ | the node-based multimodal (N-MMKG) knowledge graph, assumed directly — no asterisk variants | §3.1 | §3.1-§3.5 |
| $\mathcal{V}$ / $\mathcal{R}$ / $\mathcal{E}\subseteq\mathcal{V}\times\mathcal{R}\times\mathcal{V}$ | entities / relation types / labelled edges (factual triples $(s,r,o)$) | §3.1 | §3.1 |
| $\mathcal{M}:\mathcal{V}\to\mathbb{M}$ | total multimodal mapping; $\mathcal{M}(v)=\text{concept}$ marks conceptual entities | §3.1 | §3.1, §3.3 |
| $\mathbb{M}=\{\text{concept},\text{image},\text{text},\ldots\}$ | modality domain (concept included) | §3.1 | §3.1 |
| $\mathcal{X}$ | attribute-based multimodal features (contrast term only) | §3.1 | §3.1 |
| $\mathcal{R}_{\text{med}}$ / $\mathcal{R}_{\text{concept}}$ | mediation / conceptual relations | §3.1 | §3.1 |
| $(s,r,?)$, $o$, $\hat{o}$ | query / gold object / predicted object | §3.1 | §3.1, §3.3, §3.5 |
| $P(\hat{o}\mid s,r,\mathcal{G})$, $P(p_o\mid s,r)$ | reasoning objective and its path reduction | §3.1 | §3.1 |
| $\hat{f}_\theta(p_o,r)$ | learned path scoring function (generic form; $\theta$ = trainable parameters) | §3.1 | §3.1 |

### Embeddings (§3.2)
| Symbol | Meaning | Defined | Used |
|---|---|---|---|
| $\mathbb{E}$ | embedding space of the frozen MMKGE model | §3.2 | §3.2 |
| $g_V:\mathcal{V}\to\mathbb{E}$ | frozen entity embedding function; learned unified representation onto the universal $\mathbb{E}$ | §3.2 | §3.2, §3.4.1 |
| $g_R:\mathcal{R}\to\mathbb{E}$ | frozen relation embedding function; learned unified representation onto the universal $\mathbb{E}$ | §3.2 | §3.2, §3.4.1 |
| $\mathcal{S}:\mathcal{V}\times\mathcal{R}\times\mathcal{V}\to\mathbb{R}$ | frozen triple scorer | §3.2 | §3.2, §3.4.3, §4.4 |

### Paths & configurations (§3.3)
| Symbol | Meaning | Defined | Used |
|---|---|---|---|
| $K$ | maximum hop count (BFS bound) | §3.3 | §3.3, §3.4.1, §4.2, §4.5 |
| $\ell$ | hop count of a path ($\ell\leq 0$ excluded) | §3.3 | §3.3-§3.5 |
| $p$ / $p_o$ | a path / the path reaching object o (also the positive training path) | §3.3 / §3.1 | §3.1-§3.5 |
| $\mathcal{G}$ (full) / $\mathcal{G}_c$ | configurations: full graph / concept-only subgraph on $\{v:\mathcal{M}(v)=\text{concept}\}$ | §3.3 | §3.3, §4.2 |
| PAR | Path-Aware Reasoning (concept-only scorer) | §3.3 | §3.3, §4.2, §4.4 |

### Tokens & encoder (§3.4)
| Symbol | Meaning | Defined | Used |
|---|---|---|---|
| $\mathbf{v}_j$ | entity token for path entity $j$, obtained from $g_V$ | §3.4.1 | §3.4.1 |
| $\mathbf{r}$ / $\mathbf{r}_i$ | query relation token / edge tokens | §3.4.1 | §3.4.1 |
| $\mathbf{c}$ | learnable classification token at position 0 | §3.4.1 | §3.4.1 |
| $\tilde{\mathbf{X}}$, $\mathbf{X}$ | token sequence / Transformer input ($\mathbf{X}_i=\tilde{\mathbf{X}}_i+\mathbf{e}_i$); per-path token count $2\ell+3$ | §3.4.1 | §3.4.1, §3.4.2, §4.5 |
| $\operatorname{PE}$, $\mathbf{e}_i$ | learned absolute positional embeddings | §3.4.1 | §3.4.1 |
| $d$ | Transformer hidden dimension (value in §4.1 only) | §3.4.1 | §3.4.1-§3.4.3, §4.1 |
| $N$, $h$ | encoder depth (8) / attention heads (8) | §3.4.2 | §3.4.2, §4.1 |
| $\mathbf{H}^{(l)}$, $\mathbf{H}^{(N)}$ | layer-l / final contextualised sequence | §3.4.2-3.4.3 | §3.4.2, §3.4.3 |
| $\tilde{\mathbf{H}}$ | pre-FFN intermediate in each encoder layer (Eq. 3) | §3.4.2 | §3.4.2 |
| $L(\mathcal{B})$ | batch maximum entity count $=\max_{p\in\mathcal{B}}(\ell_p+1)$; padded token length $2L(\mathcal{B})+1$ | §3.4.2 | §3.4.2, §3.5 |
| $\mathbf{M}\in\{0,1\}^{2L(\mathcal{B})+1}$ | padding mask | §3.4.2 | §3.4.2, §3.5 |
| $\mathbf{z}$ | pooled representation (mean pooling) | §3.4.3 | §3.4.3 |
| $\tilde{y}$ | path residual (ReLU MLP, dropout 0.1) | §3.4.3 | §3.4.3 |
| $\hat{y}=\mathcal{S}(s,r,\hat{o})+\tilde{y}$ | score contract | §3.4.3 | §3.4.3, §3.5 |

### Training (§3.5)
| Symbol | Meaning | Defined | Used |
|---|---|---|---|
| $q$, $p_o$, $p_{o^-}$, $S^-(q)$ | query / positive path to gold $o$ / negative paths to sampled objects $o^-$ | §3.5.1 | §3.5.1 |
| $B$ | query batch size (2B samples per batch) | §3.5.1 | §3.5, §4.1 |
| $\mathcal{B}$, $\mathcal{Q}(\mathcal{B})$ | batch / unique queries in batch | §3.5.2 | §3.5.2 |
| $S^{\pm}(q)$ | positive/negative path score sets (contract-linked; roman, unified) | §3.5.2 | §3.5.2 |
| $\mathcal{L}_{\text{rank}}$ | margin ranking loss (Eq. 4) | §3.5.2 | §3.5, §4.1 |
| $m=1.0$ | margin | §3.5.2 | §3.5, §4.1 |
| $y_i\in\{0,1\}$ | labels | §3.5.2 | §3.5.2 |

### Experiment-side symbols (§4)
| Symbol | Meaning | Defined | Used |
|---|---|---|---|
| $\eta=0.01$, $\lambda=0.01$ | AdamW lr / weight decay (λ = weight decay ONLY) | §4.1 | §4.1 |
| $\beta_1=0.9$, $\beta_2=0.999$ | AdamW moments | §4.1 | §4.1 |
| $p=0.1$ | dropout (encoder + readout MLP) | §3.4.2/§3.4.3 | §3.4, §4.1 |
| $D=768$ | embedding dimension (instance, TransE backbone) | §4.1 | §4.1, §4.4 |
| $d=768$, $d'=2048$ | token width / feed-forward inner width (instance values; Method defines $d$, not $d'$) | §4.1 | §4.1, §4.4 |
| $F=100$ | BFS base fanout | §4.2/§4.5 | §4.2, §4.5 |
| $\delta=0.5$ | BFS decay factor | §4.2/§4.5 | §4.2, §4.5 |
| $k$ | Hits@$k$ cutoff | §4.3 | §4.3 |
| MRR, Hits@$k$ | ranking metrics, reported overall and per relation | §4.3 | §4.3, §4.4, abstract |
| $O(\ell^{2}h)$ | per-candidate attention cost over the $2\ell+3$ tokens | §4.5 | §4.5 |
| $\pm$ (Table 1) | mean $\pm$ standard deviation across relations; unweighted mean, count-weighted std — definition deliberately withheld from the caption per wip §8 neutrality call | §4.4 caption | Table 1 |

### Retired / disambiguated (do not reintroduce)
| Symbol | Status |
|---|---|
| $\chi$ | RETIRED (§15 round) |
| $\mathcal{G}^*$, asterisks | RETIRED (notation round 2026-09-20): the N-MMKG setting is assumed directly; $\mathcal{G}$ IS the multimodal graph |
| $\mathcal{V}_{\text{concept}}$ / $\mathcal{V}_{\text{mod}}$ | RETIRED (notation round): entity designation via $\mathcal{M}(v)=\text{concept}$ |
| $\tau$, $\mu$ | RETIRED (notation round): replaced by the single total mapping $\mathcal{M}:\mathcal{V}\to\mathbb{M}$ |
| $\tilde{v}$ | RETIRED (notation round): multimodal entities are $\{v:\mathcal{M}(v)\neq\text{concept}\}$ |
| $\mathbb{H}$ | RETIRED (notation round): embedding space is now $\mathbb{E}$ |
| $\mathbf{E}$, $\mathbf{R}$ | RETIRED (notation round): replaced by embedding functions $g_V$, $g_R$ |
| $g$ | RETIRED (notation round): triple scorer is now $\mathcal{S}$ |
| $\mathbf{v}_i$ / $\mathbf{v}$ (unified) | RETIRED (notation round): fusion/averaging omitted — upstream learner detail |
| old $\mathcal{M}(v)\subseteq\mathbb{M}$ | RETIRED (notation round): $\mathcal{M}$ redefined as the total mapping $\mathcal{V}\to\mathbb{M}$ |
| $d_{\mathcal{V}}$, $d_{\mathcal{R}}$, $d=\max(\cdot)$ | RETIRED (notation round): dimension bookkeeping removed from §3.2 |
| $d'$ in Method | RETIRED from §3.4.2 (notation round); survives only as the §4.1 instance $d'=2048$ |
| $\mathcal{N}(0,0.02^2)$ | RETIRED (notation round): init distribution omitted from the paper |
| $\mathbf{T}$, $\mathbf{T}^{\text{in}}$ | RETIRED (notation round): replaced by $\tilde{\mathbf{X}}$ / $\mathbf{X}$ |
| $\mathbf{p}_i$ | RETIRED (notation round): positional embedding now $\mathbf{e}_i$ |
| $\Delta$ | RETIRED (notation round): path residual now $\tilde{y}$ |
| $N_{\max}(\mathcal{B})$ | RETIRED (notation round): replaced by $L(\mathcal{B})$ |
| $p^+_q$ / $p^-_q$ | RETIRED (notation round): replaced by $p_o$ / $p_{o^-}$ |
| $\mathcal{S}^{-}(q)$ (script S for scores) | RETIRED (notation round): unified to roman $S^{\pm}(q)$; script $\mathcal{S}$ reserved for the triple scorer |
| $\mathcal{R}_{\text{med}}$ / $\mathcal{R}_{\text{concept}}$ | RETIRED 2026-09-20 (bib round): single/near-single use — verbalized ("dedicated mediation relations", "r a relation joining conceptual entities") |
| $L$ vs $N$ | L is now the batch length function $L(\mathcal{B})$; N = encoder depth |
| λ | AdamW weight decay ONLY (never BFS decay — that is δ) |

Maintenance: update this table at the end of every round; final review
after all rounds complete. Granularity note: pure operator names
(MHA, LayerNorm, GELU, ReLU, EncLayer) and local loop indices (i, j)
are deliberately NOT tracked — standard notation without signatures;
PE is tracked because it carries an explicit signature.
