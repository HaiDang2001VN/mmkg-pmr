### Structural and Methodological Diagnostics

A rigorous review of Section 3.1 reveals four specific issues that make the text appear stitched together and cause it to read like an informal dataset description rather than a general methodological framework:

1. **Premature Dataset Leakage:**
Paragraph 3 and Paragraph 4 mention fine-art specifics: *artworks are inherently visual, while artists, styles, genres, and locations are textual*, and cite `hasArtist` in the formal structural constraints. Specific entity types and relations belong exclusively to Section 4.2 (Datasets). Section 3.1 must define the transformation over generic conceptual entities and perceptual entities.
2. **Notation Redundancy and Clashing Functions:**
Paragraph 3 introduces a mapping $\chi: \mathcal{V} \to \mathbb{M}$ to record which modality produced each node. Immediately in Paragraph 4, the text introduces a second mapping $\mu(v) \in \mathbb{M}$ defined on $\mathcal{T}_{\text{mod}} \subseteq \mathcal{T}$. Defining two conflicting functions for the same modality assignment is a classic sign of disparate text merges. You only need $\mu$.
3. **Inverted Logical Flow:**
The section currently defines standard symbolic path reasoning first, abruptly interrupts itself with multimodal definitions, drops into formal graph transformation constraints, and then jumps to the architecture overview. The natural mathematical order is:
* Define the base graph and the topological transformation from an attribute-based graph (A-MMKG) to a node-based graph (N-MMKG) $\mathcal{G}^*$.
* State the four governing structural constraints in domain-agnostic terms.
* Define path reasoning directly over this transformed multimodal topology $\mathcal{G}^*$.
* Present the four-stage execution pipeline of PAMR.


4. **Conversational Phrasing:**
Phrases such as *We ask what happens*, *differ in nature*, and *a sharing that per-entity attachment cannot express and that turns multimodal values into first-class bridges* read colloquially. Replacing these with precise topological terminology (intermodal mediation, hypergraph reduction, shared perceptual anchors) elevates the academic rigor.

---

### Step-by-Step Revision Blueprint

#### 1. Consolidate Graph and Modality Notation

Maintain your core symbols $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$ and transformed graph $\mathcal{G}^* = (\mathcal{V}^*, \mathcal{E}^*, \mathcal{R}^*)$. Partition the entity set into symbolic conceptual entities and externalized multimodal entities:


$$\mathcal{V}^* = \mathcal{V}_{\text{concept}} \cup \mathcal{V}_{\text{mod}}$$


Eliminate $\chi$. Retain entity type mapping $\tau: \mathcal{V}^* \to \mathcal{T}$ and modality mapping $\mu: \mathcal{V}_{\text{mod}} \to \mathbb{M}$.

#### 2. Generalize the Four Structural Constraints

Rewrite the constraints without referencing artworks or artists:

* Constraint 1: Each multimodal node $\tilde{v} \in \mathcal{V}_{\text{mod}}$ holds exactly one modality label $\mu(\tilde{v}) \in \mathbb{M}$.
* Constraint 2: Multimodal nodes interface with conceptual entities exclusively through dedicated mediation relations $\mathcal{R}_{\text{med}} \subset \mathcal{R}^*$.
* Constraint 3: Direct relational edges $\mathcal{R}_{\text{concept}}$ link conceptual entities, ensuring cross-entity path traversals across disparate conceptual domains are mediated through intermediate multimodal nodes.
* Constraint 4: A single multimodal node can connect to multiple conceptual entities, turning perceptual assets into shared topological bridges between concepts.

#### 3. Anchor Path Reasoning to the Transformed Graph $\mathcal{G}^*$

State the objective function directly on $\mathcal{G}^*$:
Given query $(s, r, ?)$ where $s, \hat{o} \in \mathcal{V}_{\text{concept}}$ and $r \in \mathcal{R}_{\text{concept}}$, path reasoning estimates $P(\hat{o} \mid s, r, \mathcal{G}^*)$ via shortest paths $p_o$ that traverse both conceptual and multimodal entities.

---

### Revised LaTeX Implementation for `\subsection{Overview}`

```latex
\subsection{Overview}
\label{sec:overview}

A standard knowledge graph is defined as a directed multi-relational graph
$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R})$, where $\mathcal{V}$
is the set of entities, $\mathcal{R}$ the set of relation types, and
$\mathcal{E} \subseteq \mathcal{V} \times \mathcal{R} \times \mathcal{V}$ the
set of labelled edges denoting factual triples $(s, r, o)$. In multimodal
knowledge graphs, entities are associated with perceptual data such as images
and natural language text. The dominant attribute-based formulation
(A-MMKG) treats these signals as literal attribute values attached to
individual entities, aggregating modality-specific embeddings
$\mathbf{v}_i$ ($i \in \mathbb{M}$) into a single vector $\mathbf{v} \in \mathbb{H}$
per node (detailed in Section~\ref{sec:embedding-module}).

In contrast, our framework operates over a node-based multimodal knowledge
graph (N-MMKG), externalizing modality-bearing attributes into first-class
graph entities. Each entity carries a semantic type $\tau: \mathcal{V}^* \to \mathcal{T}$.
The transformed graph $\mathcal{G}^* = (\mathcal{V}^*, \mathcal{E}^*, \mathcal{R}^*)$
partitions entities into conceptual entities $\mathcal{V}_{\text{concept}}$ and
multimodal entities $\mathcal{V}_{\text{mod}}$, with a modality mapping
$\mu: \mathcal{V}_{\text{mod}} \to \mathbb{M}$ defined over perceptual nodes.
This transformation enforces four structural properties:
\begin{enumerate}
  \item Unimodal assignment: each multimodal entity $\tilde{v} \in \mathcal{V}_{\text{mod}}$ carries exactly one modality designation $\mu(\tilde{v}) \in \mathbb{M}$;
  \item Intermodal mediation: multimodal nodes interface with conceptual entities exclusively through dedicated mediation relations $\mathcal{R}_{\text{med}} \subset \mathcal{R}^*$;
  \item Domain separation: conceptual relations $\mathcal{R}_{\text{concept}} \subset \mathcal{R}^*$ connect conceptual entities directly, such that cross-domain traversals between distinct conceptual classes are mediated through shared multimodal hops;
  \item Multi-concept incidence: an individual multimodal entity can connect to multiple conceptual entities simultaneously, functioning as an explicit topological bridge across entities that conventional per-entity attachments isolate.
\end{enumerate}

Path-based reasoning on $\mathcal{G}^*$ leverages these topological sequences
to infer unobserved relations. Given a query $(s, r, ?)$ with subject
$s \in \mathcal{V}_{\text{concept}}$ and query relation $r \in \mathcal{R}_{\text{concept}}$,
the goal is to predict target object $\hat{o} \in \mathcal{V}_{\text{concept}}$
by approximating the conditional distribution $P(\hat{o} \mid s, r, \mathcal{G}^*)$.
Assuming the structural paths between query arguments provide the principal
evidence for the relation, the objective reduces to evaluating $P(p_o \mid s, r)$,
where $p_o$ denotes the shortest path connecting $s$ and $\hat{o}$ in $\mathcal{G}^*$.
A learned scoring function $\hat{f}_\theta(p_o, r)$ assigns a real-valued score
to candidate paths, and the entity associated with the highest-scoring path
is selected as the predicted answer.

The proposed PAMR architecture realizes this formulation across four sequential
stages, as depicted in Figure~\ref{fig:overview}. First, a frozen multimodal
embedding model provides fixed token representations for all entities and
relations. Second, bidirectional breadth-first search extracts shortest paths
connecting candidate query pairs over the N-MMKG topology. Third, a Transformer
encoder scores each path by contextualizing the interleaved sequence of entity
and relation tokens. Fourth, a margin-based ranking loss trains the sequence
scorer using balanced pairs of positive and negative paths. The following
subsections detail each module in sequence.

```

---

### Changes Made to Strengthen the Manuscript

* **Mathematical Precision:** Replaced vague mentions of *differ in nature* with formal definitions for $\mathcal{V}_{\text{concept}}$, $\mathcal{V}_{\text{mod}}$, $\mathcal{R}_{\text{concept}}$, and $\mathcal{R}_{\text{med}}$.
* **Notation Cleanup:** Removed the duplicate $\chi$ function, retaining only $\mu$ over the explicit domain $\mathcal{V}_{\text{mod}}$.
* **Generalization:** Removed dataset-specific terms (`artwork`, `artist`, `hasArtist`, `style`, `genre`), moving all domain-specific grounding to Section 4.2 where it belongs.
* **Cohesive Narrative:** Formulated the section as an orderly transition: standard KG definition $\to$ limitation of A-MMKG $\to$ mathematical definition of N-MMKG $\to$ path reasoning objective on $\mathcal{G}^*$ $\to$ four-stage pipeline roadmap.