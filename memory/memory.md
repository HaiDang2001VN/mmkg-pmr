# Memory — PAMR Conference Paper

## Paper File
- **Main file**: `assets/paper.tex` (~631 lines)
- **Method reference**: `assets/method.tex` (~518 lines)

## Paper Structure
1. **Introduction** — placeholder, needs expansion
2. **Related Work** — 4 subsections (Path-Based Reasoning, Multimodal KGE, Transformers for KG, Evaluation Approaches)
3. **Method**
   - Overview (4 paragraphs + figure)
   - Embedding Module (frozen KGE, VL-KGE)
   - Path Extraction (multimodal/concept configs, $\ell$ bounded by $K$)
   - PAMR Architecture (Token Interleaving, Transformer Encoder, Readout and Scoring)
   - Training (Paired Sampling, Loss Function)
4. **Experiment**
   - Implementation Details
   - Datasets (WikiArt-MKG v2)
   - Evaluation Metrics
   - Results (Comparative Study, Per-Relation Analysis)

## Hard LaTeX Rules
- No em-dashes (—)
- No ad hoc hyphens (established compounds like `breadth-first` OK)
- No semicolons in prose
- No parenthetical explanations (use verbal transitions)
- No double quotes
- `\emph{}` not `\textbf{}` for emphasis (except named methods)

## Notation Conventions
- $(s, r, o)$ for triples (plain, not bold)
- $\mathbf{s}, \mathbf{o}$ for embeddings
- $\mathbb{M}$ for modality set
- $\mathcal{M}: \mathcal{V} \to \mathbb{M}$ for modality map
- $\mathbb{H}$ for embedding space
- $\mathcal{G}_{\text{multimodal}}$ and $\mathcal{G}_{\text{concept}}$ for graph configs
- $d$ not $d_{\text{model}}$
- $L$ not $L_{\text{tr}}$
- $d'$ not $d_{\text{ff}}$
- $K$ for max hop count
- $\ell$ for hop count (bounded by $K$)
- $N^-$ for number of negative paths
- $\hat{y}$ for scores (not $s$ to avoid conflict with subject entity)
- $\mathcal{Q}(\mathcal{B})$ for set of queries in batch
- $S^+(q)$ and $S^-(q)$ for positive/negative score lists (parentheses, not subscript)

## Key Dataset Stats (WikiArt-MKG v2)
- 224,166 entities (216,564 artworks, 7,602 attributes)
- 22 relations
- 7,877,220 / 208,513 / 208,368 train/val/test triples

## Key Results
- **PAMR**: MRR 0.379, Hits@1 0.345, Hits@3 0.576, Hits@5 0.782
- **ComplEx**: MRR 0.363, Hits@1 0.387, Hits@3 0.625, Hits@5 0.772
- **PAR**: MRR 0.221, Hits@1 0.058, Hits@3 0.199, Hits@5 0.331
- **Runtime**: PAMR ~543s/epoch, ComplEx ~577s/epoch

## Per-Relation Analysis
- **isPupilOf**: PAMR strongest (0.390 MRR, 0.909 Hits@5)
- **isInfluencedOn**: PAMR strongest (0.415 MRR)
- **isTeacherOf**: ComplEx strongest (0.422 MRR)
- **isInfluencedBy**: ComplEx strongest (0.354 MRR)

## Configuration
- **ComplEx backbone**: $D = 768$, $d = 1536$
- **PAMR**: 8 layers, 8 heads, $d' = 2048$, dropout 0.1
- **AdamW**: $\eta = 0.01$, $\lambda = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$
- **BFS**: max_hops = 5, max_fanout = 100, decay_factor = 0.5

## Figure Files
- `assets/overview.svg` — PAMR architecture overview (included in paper)
- `assets/case.svg` — Case study visualization (6.6MB draw.io diagram)
- `assets/MRR.svg`, `assets/Hits1.svg`, `assets/Hits3.svg`, `assets/Hits5.svg` — Per-relation bar charts

## Case Study Diagram (case.svg)
- Shows query (Picasso, isInfluenceOn, ?) reasoning paths
- Artwork nodes with embedded images (nodes 4, 20, 23)
- Artist nodes: Picasso, Juan Gris, Dali, Braque, Matisse, Lee Krasner
- Relations: hasArtist, isRelatedArtist, isInfluenceBy, isRelatedArtwork
- Demonstrates multimodal path reasoning on art domain graph

## Remaining TODOs
- Introduction section needs expansion (L76)
- Overview subsection may need motivation-first framing
- Final style verification pass
