## Implementation Related Concerns and Solutions

* **Target Edge Leakage in Breadth First Search**
* *Concern:* The path extraction procedure does not verify whether query triples, inverse relations, or evaluation edges are removed from the graph before running the path search algorithm. Retaining these edges allows the search to reach target nodes trivially, which invalidates reported ranking performance.


* *Fix:* Construct the graph adjacency list strictly using training split triples. During path extraction for any sample, explicitly mask out the query edge along with its reciprocal or inverse counterpart before executing bidirectional search. Document this edge removal procedure comprehensively in the methodology section.


* **Confounded Topology and Modality Contributions**
* *Concern:* The comparison between Path Aware Reasoning and Path Aware Multimodal Reasoning alters both graph topology and feature representations simultaneously. Path Aware Reasoning traverses a concept only subgraph while Path Aware Multimodal Reasoning utilizes the full graph containing artwork entities, leaving it ambiguous whether performance gains stem from multimodal feature vectors or additional structural connectivity.


* *Fix:* Conduct controlled ablation experiments using the exact same full graph topology. Benchmark the full graph using structural only initializations, the full graph using randomly shuffled or zeroed out visual and textual embeddings, and the full graph using unified multimodal embeddings.


* **Missing Architectural and Hyperparameter Ablations**
* *Concern:* The paper omits experimental evaluations comparing classification token readout against mean pooling, assessing Transformer encoder layer depth, varying path length bounds, and comparing frozen against unfrozen representation layers.


* *Fix:* Implement an ablation study reporting performance across classification token pooling versus mean pooling, encoder depths spanning 4, 8, and 12 layers, maximum path length thresholds from 3 to 5 hops, and frozen representations versus fine tuned linear projections.


* **Performance Drop on Stricter Top Rank Metrics**
* *Concern:* The model achieves strong Mean Reciprocal Rank and Hits at 5 but drops noticeably below ComplEx on Hits at 1 and Hits at 3. This degradation limits reliability for top candidate selection.


* *Fix:* Integrate a hard negative contrastive loss or temperature scaled cross entropy objective into training. This will force the Transformer scoring function to penalize plausible false paths more heavily and sharpen score separation among top candidates.


* **Narrow Evaluation Scope and Missing Baseline Paradigms**
* *Concern:* Evaluation is limited to WikiArt MKG v2 across four artist relations, without benchmarking against modern graph neural networks or path based reasoning architectures adapted to the multimodal graph.


* *Fix:* Add an evaluation on an additional benchmark such as WN9 IMG or DB15K, evaluate across all available relation types, and adapt a representative path based graph neural network such as NBFNet or RED GNN to the modality as entity graph for direct comparison.


* **Sampling Rigor and Metric Uncertainty**
* *Concern:* Testing uses ten sampled negatives per query rather than full candidate ranking, omits repeated seed runs, provides no confidence intervals, and does not specify path selection when multiple shortest paths connect two entities.


* *Fix:* Evaluate models using filtered full candidate ranking or explicitly formalized negative candidate sets, execute all runs across five random seeds to report mean and standard deviation, and implement deterministic path handling by scoring all shortest paths and aggregating candidate scores via mean or maximum pooling.


* **Empty Code Repository**
* *Concern:* The linked GitHub repository contains only boilerplate files, preventing reproducibility.


* *Fix:* Upload complete Python and C++ source code, graph preprocessing scripts, environment configuration files, training logs, and model checkpoints to ensure end to end reproducibility.



---

## Text Related Concerns and Solutions

* **Baseline Data Inconsistencies**
* *Concern:* Numerical results for baselines in Table 1 do not match the values published in Table 3 of the original VL KGE reference, and the text contradicts Table 1 by stating that all methods share frozen ComplEx embeddings while reporting TransE, RotatE, and DistMult baselines.


* *Fix:* Verify every baseline number against the official VL KGE publication. Correct the text to clarify whether baseline models were evaluated locally under identical sampling conditions or taken directly from external literature, and resolve the contradiction by specifying which embedding backbones correspond to each column.


* **Conflicting Candidate Scoring Descriptions**
* *Concern:* Section 4.3 states that the model scores every candidate object, whereas Section 4.2 states that evaluation uses ten sampled negatives per query.


* *Fix:* Reconcile the evaluation descriptions across Section 4.2 and Section 4.3 so that both accurately describe the exact candidate pool evaluated during testing.


* **Duplicate Sentences in Section 4.2**
* *Concern:* Section 4.2 contains a verbatim repetition of the sentence describing the evaluation relations and negative candidate counts.


* *Fix:* Proofread Section 4.2 and excise the repeated sentence to restore clean paragraph flow.


* **Overstated Novelty and Interpretability Claims**
* *Concern:* Reviewers criticized describing the work as pioneering given that it combines existing techniques, and noted that claiming interpretability from a single qualitative case study is insufficient because a multi layer Transformer remains a non transparent black box.


* *Fix:* Tone down rhetoric by replacing pioneering with terminology describing an architectural framework for path reasoning under the modality as entity setting. Temper interpretability claims by acknowledging that the Transformer sequence model introduces a non transparent layer, and support the case study by visualizing attention weights across path tokens.


* **Incomplete Formal Semantic Grounding**
* *Concern:* The modality as entity formulation lacks formal semantic characterization, explicit structural constraints, and grounding in established knowledge representation standards.


* *Fix:* Expand Section 3.1 to formalize the modality as entity transformation mathematically, specifying exact entity typing rules, schema mappings, and how multimodal attribute nodes align with standard graph schemas.


* **Figure Sizing and Readability**
* *Concern:* Text labels and axis markers across Figures 2, 3, 4, and 5 are too small to inspect clearly.


* *Fix:* Replot all charts with larger typography, thicker line weights, and combine the individual hit rate charts into a single compact multi panel figure.


* **Incomplete Runtime and Scalability Analysis**
* *Concern:* The reported training duration omits path extraction and preprocessing overhead, and the paper lacks discussion regarding computational complexity and inference latency as the graph scales.


* *Fix:* Document total end to end wall clock times including graph construction and breadth first search traversal, and expand the discussion section to address asymptotic complexity, memory footprints, and inference efficiency optimizations.