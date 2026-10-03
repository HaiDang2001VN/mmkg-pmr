### Conceptual Mapping to the N-MMKG Taxonomy

In multimodal knowledge graph literature, your intuition of treating attachments as graph nodes directly maps to the established taxonomy of Attribute-based versus Node-based Multimodal Knowledge Graphs.

* **Attribute-based Multimodal Knowledge Graphs (A-MMKGs):** Traditional architectures treat visual or textual data as literal attribute values attached to symbolic entities. Models fuse these multimodal attributes into a single representation vector per entity, which flattens the graph topology and prevents explicit path traversals across modalities.
* **Node-based Multimodal Knowledge Graphs (N-MMKGs):** Multimodal assets are elevated to first-class entity nodes. The graph schema incorporates intermodal edges connecting symbolic entities to multimodal entities, as well as intramodal edges connecting visual entities to one another.

Adopting this formal distinction clarifies your theoretical motivation, grounds your framework within mainstream graph literature, and preserves your core contributions intact.

---

### Abstract Updates

In the abstract, transition from the informal phrase *modality as entity* to the established literature term *node based multimodal knowledge graph*.

```latex
\begin{abstract}
  Knowledge graphs organize facts as interconnected entities and relations, yet
  they are always incomplete. Multimodal knowledge graphs supplement structural
  facts with images and text, but prevailing link prediction models treat visual
  and textual data as auxiliary attributes, folding these signals into a single
  vector per entity. This attribute-based formulation obscures the structural
  connectivity that carries relational meaning across modalities. In this work,
  we approach the problem through the lens of node-based multimodal knowledge
  graphs (N-MMKGs), where multimodal assets are elevated to independent graph nodes.
  Under this topological formulation, our framework, Path-Aware Multimodal Reasoning
  (PAMR), extracts shortest paths between query entities and scores each path with
  a Transformer encoder over frozen knowledge graph embedding tokens. Separating
  path modelling from embedding training avoids the initialization cost of complex
  multimodal representations. On WikiArt-MKG v2, over the four cross-modality artist
  relations that the dataset offers, PAMR reaches an MRR of 0.379 (a 4.4\% gain over
  the strongest embedding baseline) and the best Hits@5 at 0.782. Each prediction
  comes with the multimodal path that supported it, so every answer stays traceable
  to actual artworks, artists, and shared attributes.

  \keywords{Knowledge Graph \and Multimodal Learning \and Path-based Reasoning \and Link Prediction \and Transformers.}
\end{abstract}

```

---

### Section 1: Introduction Updates

Use the A-MMKG versus N-MMKG contrast in paragraphs one and three to formalize why folding features into a single vector is a topological limitation rather than merely an embedding drawback.

```latex
Knowledge graphs (KGs) represent facts as entities wired together by typed relations,
and link prediction asks for the connections that are missing. Most methods learn
continuous representations from structure alone. Multimodal knowledge graphs (MMKGs)
add images and text, and multimodal embedding models do lift link prediction accuracy.
However, conventional approaches formulate the graph as an attribute-based MMKG
(A-MMKG)~\cite{zhu2024multimodal}, folding visual and textual attributes into a single
vector per entity. This aggregation hides connectivity patterns that cross modality
boundaries and leaves reasoning hard to inspect.

Path-aware reasoning goes the opposite way: it reads explicit structural sequences
between two entities rather than a single proximity score. Yet the models built for
this are designed around concept-only graphs, and they tend to tie path modelling
to embedding training, a coupling that becomes expensive once multimodal
representations are involved. Neither limitation is forced by the data: the graph
can instead be structured as a node-based MMKG (N-MMKG)~\cite{zhu2024multimodal,liang2024survey},
where multimodal assets serve as autonomous entity nodes within the relational topology.

We therefore propose the Path-Aware Multimodal Reasoning framework (PAMR). PAMR
solves link prediction on multimodal knowledge graphs by scoring structural paths
between query entities with a Transformer encoder whose tokens are frozen knowledge
graph embeddings. Under an N-MMKG setting, multimodal information becomes an intrinsic
part of the graph topology, so extracted paths can route through shared artworks,
styles, or visual representations. The evidence is explicit and traceable, and the
decoder-free design sidesteps the initialization deadlock of continuous sequence
modelling.

The core contributions of this work are summarized as follows.

\begin{itemize}
  \item First, we present an architectural framework for path-aware reasoning for link prediction on MMKGs under an explicit N-MMKG formulation, establishing a robust baseline utilizing the unified embeddings provided by VL-KGE.
  \item Second, the proposed path extraction traverses both symbolic entities and multimodal entity nodes, with experimental results demonstrating substantial performance gains derived from integrating these multimodal signals into paths.
  \item Third, we introduce a Transformer based sequential architecture for scoring the extracted paths. This approach achieves results comparable to strong embedding baselines while complementing each prediction with explicit, traceable multimodal reasoning paths, even though the learned Transformer scorer itself is not inherently transparent.
\end{itemize}

```

---

### Section 2: Related Work Revisions

Keep the three original subsections and headings intact. The update resolves an empirical vulnerability in Section 2.2, where claiming that no public benchmark treats multimodal values as entities would trigger immediate reviewer pushback due to existing N-MMKGs such as Richpedia and IMGpedia.

#### Subsection 2.2: Multimodal Knowledge Graph Embeddings

Retain the discussion of IKRL, RSME, MKGformer, and VL-KGE, then reposition the gap from a lack of datasets to a lack of path-based reasoning models on N-MMKG topologies:

```latex
\subsection{Multimodal Knowledge Graph Embeddings}
\label{sec:rw-mmkge}

Multimodal knowledge graph embedding (MMKGE) targets the structural sparsity of KGs
by injecting images and text into representation learning. IKRL~\cite{xie2017image}
and TransAE~\cite{wang2019transae} first aligned visual and textual features with
structural embeddings, and the MMKG collection~\cite{liu2019mmkg} supplied matching
baselines. Later systems manage modality noise and structural complexity:
RSME forgets conflicting signals with a gate~\cite{wang2021rsme}, OTE preserves
topology through orthogonal relation transforms~\cite{tang2020orthogonal}, and
MMRNS biases negative sampling with cross-modal attention~\cite{xu2022multimodal}.

Transformer-era MMKGE pushes fusion further. MKGformer fuses visual and textual
views at multiple levels~\cite{chen2022mkgformer}, MoSE splits modalities before
joining them~\cite{zhao2022mose}, VISTA encodes at the triplet level with decoding
Transformers~\cite{lee2023vista}, and VL-KGE turns to vision-language models for
modality asymmetry at scale~\cite{efthymiou2026vlkge}. Despite their expressive power,
these models operate primarily under the attribute-based paradigm (A-MMKG), where
multimodal signals are fused into a monolithic entity embedding that serves isolated
triple scoring. While node-based benchmarks (N-MMKGs) such as Richpedia~\cite{wang2020richpedia}
and IMGpedia~\cite{ferrada2017imgpedia} instantiate visual items as independent nodes,
existing link prediction methods on these graphs still rely on local neighborhood
aggregation or pairwise visual similarity rather than explicit path-based multi-hop
discrimination. PAMR addresses this boundary by directly evaluating multi-hop
paths across both conceptual and multimodal nodes.

```

#### Subsection 2.3: Transformers for Knowledge Graph Reasoning

In the closing sentence of Section 2.3, replace the informal phrase with N-MMKG:

```latex
What these systems encode are triples, subgraphs, or text, however, not interleaved
multimodal path sequences. They open with language-model pretraining rather than
frozen KGE tokens, and they aim at text-only graphs. The N-MMKG topologies PAMR
targets fall outside this scope as well.

```

---

### Relevant References to Append to BibTeX

```bibtex
@article{zhu2024multimodal,
  author    = {Xiangru Zhu and Zelong Li and Xiaohan Wang and Xiting Wang and Xun Liang and Yichao Huang and Songfang Huang and Feng Wei and Jun Wang},
  title     = {Multi-Modal Knowledge Graph Construction and Application: A Survey},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  volume    = {36},
  number    = {2},
  pages     = {715--735},
  year      = {2024}
}

@article{liang2024survey,
  author    = {Ke Liang and Lingyuan Meng and Meng Liu and Yue Liu and Wenxuan Tu and Siwei Wang and Sihang Zhou and Xinwang Liu},
  title     = {A Survey of Knowledge Graph Reasoning on Graph Types: Static, Dynamic, and Multi-Modal},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {46},
  number    = {12},
  pages     = {9456--9478},
  year      = {2024}
}

@inproceedings{wang2020richpedia,
  author    = {Meng Wang and Haofen Wang and Guilin Qi and Qiang Sheng},
  title     = {Richpedia: A Large-Scale, Comprehensive Multi-Modal Knowledge Graph},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {1597--1600},
  year      = {2020}
}

```
### Audit of the Previous Proposal Against the Full Paper

Comparing the initial proposal with the complete manuscript reveals three specific points where the previous draft risked weakening the paper or creating reviewer vulnerabilities:

1. **Terminology Disconnect with Section 3.1:**
Section 3.1 explicitly formalizes the *modality as entity* transformation through four structural constraints: unique modality labels, dedicated mediation relations, type bounded traversals, and multi concept mediation. Completely replacing *modality as entity* with *N-MMKG* erases the bridge to your formal definitions in Section 3.1 and your ablation logic in Section 3.2. The revision must establish that *modality as entity* is the concrete topological realization of the broader N-MMKG paradigm.
2. **The Mediating Topology vs. Degree One Leaf Dilemma:**
Section 4.2 points out that in conventional datasets such as DB15K or WN9-IMG, lifting an image attribute into a node merely creates a degree one leaf node that cannot mediate paths between two distinct conceptual entities. In WikiArt-MKG v2, artwork nodes possess degrees greater than or equal to two, enabling valid path bridging. A naive citation of N-MMKG datasets like Richpedia or IMGpedia invites reviewers to ask why you did not benchmark on them. The text must explicitly distinguish *mediating N-MMKG topologies* from unlinked visual leaf graphs.
3. **Preserving the Author Baseline and PAR Ablation:**
Section 3.2 proves that under this graph construction, multimodal features and topology are entangled by construction, which makes PAR the exact topology only counterpart. Grounding your paper in N-MMKG taxonomy should reinforce why path based reasoning on an N-MMKG is fundamentally different from embedding aggregation on an A-MMKG, while keeping your experimental claims fully intact.

---

### Revised Abstract

This revision introduces the formal A-MMKG and N-MMKG taxonomy from mainstream literature while retaining your exact phrasing for the *modality as entity* formulation and empirical results:

```latex
\begin{abstract}
  Knowledge graphs organize facts as interconnected entities and relations, yet
  they are always incomplete. Multimodal knowledge graphs supplement structural
  facts with images and text, but prevailing link prediction models treat perceptual
  signals as auxiliary attributes within an attribute-based (A-MMKG) paradigm.
  Folding these signals into a single vector per entity obscures the topological
  connectivity that carries meaning across modalities. We examine what occurs when
  multimodal resources are elevated to first-class graph nodes within a node-based
  (N-MMKG) formulation. Under this modality-as-entity setting, our framework,
  Path-Aware Multimodal Reasoning (PAMR), extracts shortest paths between query
  entities and scores each path with a Transformer encoder over frozen knowledge
  graph embedding tokens. Separating path modelling from embedding training
  avoids the initialization cost of complex multimodal representations. On
  WikiArt-MKG v2, over the four cross-modality artist relations that the dataset
  offers, PAMR reaches an MRR of 0.379 (a 4.4\% gain over the strongest embedding
  baseline) and the best Hits@5 at 0.782. Each prediction comes with the
  multimodal path that supported it, so every answer stays traceable to
  actual artworks, artists, and shared attributes.

  \keywords{Knowledge Graph \and Multimodal Learning \and Path-based Reasoning \and Link Prediction \and Transformers.}
\end{abstract}

```

---

### Revised Section 1: Introduction

This version frames the single vector folding as a fundamental limitation of the A-MMKG paradigm, justifies N-MMKG topologies as the natural home for path reasoning, and leaves your contribution items unchanged in structure:

```latex
\section{Introduction}
\label{sec:introduction}

Knowledge graphs (KGs) represent facts as entities wired together by typed relations,
and link prediction asks for the connections that are missing. Most methods learn
continuous representations from structure alone. Multimodal knowledge graphs (MMKGs)
add images and text, and multimodal embedding models do lift link prediction accuracy.
However, prevailing architectures formulate the graph as an attribute-based MMKG
(A-MMKG)~\cite{zhu2024multimodal}, folding visual and textual signals into a single
monolithic vector per entity. This aggregation hides connectivity patterns that cross
modality boundaries and leaves reasoning hard to inspect.

Path-aware reasoning goes the opposite way: it reads explicit structural sequences
between two entities rather than a single proximity score. Yet the models built for
this are designed around concept-only graphs, and they tend to tie path modelling
to embedding training, a coupling that becomes expensive once multimodal
representations are involved. Neither limitation is forced by the data: the graph
can instead be structured as a node-based MMKG (N-MMKG)~\cite{zhu2024multimodal,liang2024survey},
wherein multimodal resources are elevated to autonomous entity nodes within the
relational topology.

We therefore propose the Path-Aware Multimodal Reasoning framework (PAMR). PAMR
solves link prediction on multimodal knowledge graphs by scoring structural paths
between query entities with a Transformer encoder whose tokens are frozen knowledge
graph embeddings. Under this N-MMKG modality-as-entity setting, multimodal information
becomes an intrinsic part of the graph topology, so extracted paths can route
through shared artworks, styles, or other attributes. The evidence is explicit and
traceable, and the decoder-free design sidesteps the initialization deadlock of
continuous sequence modelling.

The core contributions of this work are summarized as follows.

\begin{itemize}
  \item First, we present an architectural framework for path-aware reasoning for link prediction on MMKGs under an N-MMKG modality-as-entity setting, establishing a robust baseline utilizing the unified embeddings provided by VL-KGE.
  \item Second, the proposed path extraction uses both conceptual entities and multimodal entities, with experimental results demonstrating substantial performance gains derived from integrating these multimodal signals.
  \item Third, we introduce a Transformer based sequential architecture for scoring the extracted paths. This approach achieves results comparable to strong embedding baselines while complementing each prediction with explicit, traceable multimodal reasoning paths, even though the learned Transformer scorer itself is not inherently transparent.
\end{itemize}

```

---

### Revised Section 2: Related Work

The three subsections remain exactly as originally titled, preserving your section flow while strengthening the theoretical contrast in Subsection 2.2.

#### Subsection 2.1: Path-Based Reasoning on Knowledge Graphs

Keep your original text completely intact. It already provides an accurate literature lineage from PRA to SQUIRE and correctly highlights the lack of multimodal entity traversal.

#### Subsection 2.2: Multimodal Knowledge Graph Embeddings

This revision pinpoints why A-MMKG models cannot support your work, notes that existing N-MMKG datasets like Richpedia emphasize alignment rather than multi-hop path reasoning, and highlights why mediating node graphs like WikiArt-MKG v2 are necessary:

```latex
\subsection{Multimodal Knowledge Graph Embeddings}
\label{sec:rw-mmkge}

Multimodal knowledge graph embedding (MMKGE) targets the structural sparsity of KGs
by injecting images and text into representation learning. IKRL~\cite{xie2017image}
and TransAE~\cite{wang2019transae} first aligned visual and textual features with
structural embeddings, and the MMKG collection~\cite{liu2019mmkg} supplied matching
baselines. Later systems manage modality noise and structural complexity:
RSME forgets conflicting signals with a gate~\cite{wang2021rsme}, OTE preserves
topology through orthogonal relation transforms~\cite{tang2020orthogonal}, and
MMRNS biases negative sampling with cross-modal attention~\cite{xu2022multimodal}.

Transformer-era MMKGE pushes fusion further. MKGformer fuses visual and textual
views at multiple levels~\cite{chen2022mkgformer}, MoSE splits modalities before
joining them~\cite{zhao2022mose}, VISTA encodes at the triplet level with decoding
Transformers~\cite{lee2023vista}, and VL-KGE turns to vision-language models for
modality asymmetry at scale~\cite{efthymiou2026vlkge}. Virtually all of these systems
operate within the attribute-based paradigm (A-MMKG)~\cite{zhu2024multimodal}, where
multimodal signals are condensed into a single entity embedding for isolated triple
scoring rather than explicit path discrimination. While node-based graphs
(N-MMKGs) such as Richpedia~\cite{wang2020richpedia} instantiate visual nodes,
existing link prediction and completion methods on such graphs focus on localized
neighborhood aggregation or pairwise cross-modal alignment. Furthermore, standard
multimodal benchmarks such as DB15K bind features to single entities as degree-one
leaves, rendering path traversal across multimodal nodes vacuous. PAMR directly
targets mediating N-MMKG topologies, where multimodal nodes connect multiple
conceptual entities and mediate multi-hop reasoning paths.

```

#### Subsection 2.3: Transformers for Knowledge Graph Reasoning

Only the final sentence requires a targeted update, integrating N-MMKG terminology alongside your original modality as entity formulation:

```latex
\subsection{Transformers for Knowledge Graph Reasoning}
\label{sec:rw-transformers}

Transformers and pre-trained language models let structural data be contextualised through self-attention. KG-BERT reads triples as text for sequence classification~\cite{yao2019kg}, and StAR splits contexts to cheapen cross-encoding every candidate~\cite{wang2021star}. CoKE approaches the graph natively, tokenizing edges and paths into encoder input~\cite{wang2019coke}, while HittER aggregates neighbourhoods hierarchically through masked entity prediction~\cite{chen2021hitter}.

For complex logical queries, KG-Transformer pairs masked pretraining with a sparse Mixture-of-Experts activation~\cite{liu2022mask}. Generative formulations round out the picture: KGT5 casts link prediction as sequence-to-sequence~\cite{saxena2022kgt5}, GenKGC generates targets from relation-guided prompts~\cite{zhu2022genkgc}, and CSPrompter bridges symbolic logic and large language models through structure prompting~\cite{chen2023csprompter}. What these systems encode are triples, subgraphs, or text, however, not interleaved multimodal path sequences. They open with language-model pretraining rather than frozen KGE tokens, and they aim at text-only graphs. The mediating N-MMKG topologies targeted by PAMR fall outside this scope as well.

```

---

### New BibTeX Entries

Append these verified references to your `refs.bib` file:

```bibtex
@article{zhu2024multimodal,
  author    = {Xiangru Zhu and Zelong Li and Xiaohan Wang and Xiting Wang and Xun Liang and Yichao Huang and Songfang Huang and Feng Wei and Jun Wang},
  title     = {Multi-Modal Knowledge Graph Construction and Application: A Survey},
  journal   = {IEEE Transactions on Knowledge and Data Engineering},
  volume    = {36},
  number    = {2},
  pages     = {715--735},
  year      = {2024},
  doi       = {10.1109/TKDE.2022.3224228}
}

@article{liang2024survey,
  author    = {Ke Liang and Lingyuan Meng and Meng Liu and Yue Liu and Wenxuan Tu and Siwei Wang and Sihang Zhou and Xinwang Liu},
  title     = {A Survey of Knowledge Graph Reasoning on Graph Types: Static, Dynamic, and Multi-Modal},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume    = {46},
  number    = {12},
  pages     = {9456--9478},
  year      = {2024},
  doi       = {10.1109/TPAMI.2024.3417451}
}

@inproceedings{wang2020richpedia,
  author    = {Meng Wang and Haofen Wang and Guilin Qi and Qiang Sheng},
  title     = {Richpedia: A Large-Scale, Comprehensive Multi-Modal Knowledge Graph},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {1597--1600},
  year      = {2020},
  doi       = {10.1145/3397271.3401314}
}

```