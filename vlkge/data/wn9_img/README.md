# WN9-IMG Dataset

## Overview

The **WN9-IMG** is a multimodal knowledge graph that augments the WordNet subset **WN9** with image representations.  
It was originally introduced in the paper *"Image-embodied Knowledge Representation Learning"* (IKRL) and is widely used as a benchmark for multimodal knowledge graph completion and link prediction tasks.

This version follows the same structure as the original dataset, while we re-downloaded and organized the associated image data to ensure consistency and accessibility.

---

## Dataset Structure

| File | Description |
|------|--------------|
| **`wn9_img_triples.csv`** | Knowledge graph triples derived from the WN9 dataset, with visual entities linked to WordNet synsets. |

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
> *Shizhen Xu, Kang Liu, Siwei Lai, Yubo Chen, and Jun Zhao.*  
> *In Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI), 2017.*

For reproducibility, the image data has been **re-downloaded and restructured**, while maintaining compatibility with the original entity and relation identifiers used in IKRL.

---

## Usage

You can easily load the dataset with **pandas**:

```python
import pandas as pd

wn9_img = pd.read_csv('wn9_img_triples.csv')
print(wn9_img.head())
```

## License

The dataset metadata is released under the
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
license.