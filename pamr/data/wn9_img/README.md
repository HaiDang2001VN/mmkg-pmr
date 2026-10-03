# WN9-IMG Dataset

## Overview

The **WN9-IMG** dataset is a multimodal knowledge graph that augments the WordNet subset **WN9** with image representations.

It was originally introduced in *"Image-embodied Knowledge Representation Learning"* (IKRL) and is widely used as a benchmark for multimodal knowledge graph completion and link prediction tasks.

This version is the one used in the [VL-KGE paper](https://doi.org/10.1145/3774904.3792677).  
The associated image data has been re-downloaded and organized to ensure consistency and accessibility.

---

## Dataset Structure

| File | Description |
|------|--------------|
| **`wn9_img_triples.csv`**  | Knowledge graph triples derived from the WN9 dataset, with visual entities linked to WordNet synsets. |

---

## Data Format

**Total triples:** 14,397  
**Columns (4):**

| Column | Description |
|---------|-------------|
| `head` | Entity identifier (WordNet synset ID, e.g., `n02103406`). |
| `relation` | Relation type between entities (e.g., `_hypernym`, `_hyponym`). |
| `tail` | Linked entity identifier (WordNet synset ID). |
| `mode` | Dataset split (`train`, `val`, or `test`). |

---

## Data Origin

The **WN9-IMG** dataset extends the **WN9** subset of WordNet with visual data.  
Each synset is associated with one or more representative images sourced from publicly available datasets.

> This dataset configuration follows the setup introduced in the paper:  
> **"Image-embodied Knowledge Representation Learning" (IKRL)**  
> *Ruobing Xie, Zhiyuan Liu, Huanbo Luan, and Maosong Sun.*  
> *In Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI), 2017.*

---

## Usage

You can easily load the dataset with **pandas**:

```python
import pandas as pd

wn9_img = pd.read_csv('data/wn9_img/wn9_img_triples.csv')
print(wn9_img.head())
```

## License

The dataset metadata is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license.

---

## Citation

If you use this WN9-IMG instance, please cite both the original WN9-IMG paper and the VL-KGE paper:
```bibtex
@inproceedings{10.5555/3172077.3172327,
author = {Xie, Ruobing and Liu, Zhiyuan and Luan, Huanbo and Sun, Maosong},
title = {Image-embodied knowledge representation learning},
year = {2017},
publisher = {AAAI Press},
booktitle = {Proceedings of the 26th International Joint Conference on Artificial Intelligence},
pages = {3140–3146}
}

@inproceedings{efthymiou2026vlkge,
  title     = {{VL-KGE}: Vision--Language Models Meet Knowledge Graph Embeddings},
  author    = {Efthymiou, Athanasios and Rudinac, Stevan and Kackovic, Monika and Wijnberg, Nachoem and Worring, Marcel},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3774904.3792677},
  doi       = {10.1145/3774904.3792677}
}
```