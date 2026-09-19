# WikiArt-MKG-v1 Dataset

## Overview

**WikiArt-MKG-v1** is a fine-art multimodal knowledge graph that models artworks and their categorical attributes (artists, styles, creation years, and tags). It was introduced in the paper **["VL-KGE: Vision–Language Models Meet Knowledge Graph Embeddings" (WWW '26)](https://doi.org/10.1145/3774904.3792677)**, and is designed to evaluate multimodal knowledge graph embedding methods under modality asymmetry, where different entity types naturally possess different modalities.

---

## Dataset Structure

| File | Description |
|------|--------------|
| **`wikiart_mkg_v1_triples.csv`** | Knowledge graph triples connecting artworks (visual entities) to their attributes (textual entities). |
| **`wikiart_v1.csv`** | Metadata for artworks including titles, URLs, and attribute information. |
| **`features/`** | Directory containing precomputed visual, textual, and relation features. |

---

## Statistics

- **Total Entities:** 76,758
  - Visual entities (artworks): 75,921
  - Textual entities (attributes): 837
- **Relations:** 4
- **Total Triples:** 353,683
  - Train: 299,968
  - Validation: 34,020
  - Test: 19,695

---

## Data Format

**`wikiart_mkg_v1_triples.csv`** contains the following columns:

| Column | Description |
|---------|-------------|
| `head` | Artwork identifier (e.g., `artwork_12345`). |
| `relation` | Relation type connecting artwork to attribute. |
| `tail` | Attribute identifier (artist, style, year, or tag). |
| `mode` | Dataset split (`train`, `val`, or `test`). |

---

## Relations

The dataset contains 4 relation types:

1. **`isCreatedByArtist`** - Links artwork to its creator.
2. **`hasStyle`** - Links artwork to its artistic style/movement.
3. **`isCreatedInYear`** - Links artwork to its creation year.
4. **`isAssociatedWithTag`** - Links artwork to descriptive tags.

---

## Modality Asymmetry

Unlike WN9-IMG where all entities have both visual and textual features, WikiArt-MKG-v1 exhibits **modality asymmetry**:

- **Artworks (heads):** Represented with visual features (images).
- **Attributes (tails):** Represented with textual features (names/descriptions).

---

## Evaluation Protocol

- **Task:** Tail prediction only (predict artwork attributes).
- **Metrics:** MRR, Hits@1, Hits@3, Hits@10.
- **Filtering:** Filtered ranking with per-relation candidate pools.
- **Inductive Setting:** Test artworks are unseen during training.

---

## Features

Precomputed features are available in the `features/` directory:

| File | Description | Dimensions |
|------|-------------|------------|
| `wikiart_mkg_v1_vf_clip.pkl` | CLIP visual features | 75,921 × 768 |
| `wikiart_mkg_v1_tf_clip.pkl` | CLIP textual features | 837 × 768 |
| `wikiart_mkg_v1_vf_blip.pkl` | BLIP visual features | 75,921 × 256 |
| `wikiart_mkg_v1_tf_blip.pkl` | BLIP textual features | 837 × 256 |
| `wikiart_mkg_v1_vf_vit_b_16.pkl` | ViT-B/16 visual features | 75,921 × 768 |
| `wikiart_mkg_v1_tf_bert.pkl` | BERT textual features | 837 × 768 |
| `wikiart_mkg_v1_rf_clip.pkl` | CLIP relation features | 4 × 768 |
| `wikiart_mkg_v1_rf_blip.pkl` | BLIP relation features | 4 × 256 |
| `wikiart_mkg_v1_rf_bert.pkl` | BERT relation features | 4 × 768 |

---

## Usage Example
```python
from vlkge.dataloader import KnowledgeGraphDataLoader

loader = KnowledgeGraphDataLoader(
    data_path='data/wikiart_mkg_v1/wikiart_mkg_v1_triples.csv',
    dataset_name='wikiart_mkg_v1',
    use_per_relation_candidates=True,
    bidirectional_eval=False,
    inductive=True,
    modality_asymmetry=True
)

train_data, val_data, test_data = loader.split_data()
```

---

## Data Origin

The WikiArt-MKG-v1 is constructed from the WikiArt-v1 artwork collection originally introduced in:

> **"Graph Neural Networks for Knowledge Enhanced Visual Representation of Paintings" (ArtSAGENet)**  
> *Athanasios Efthymiou, Stevan Rudinac, Monika Kackovic, Marcel Worring, and Nachoem Wijnberg.*  
> *In Proceedings of the 29th ACM International Conference on Multimedia (MM '21), 2021.*

The WikiArt-v1 collection provides artwork images and metadata (artists, stylistic movements, years, tags). In the VL-KGE work, this data was transformed into an explicit multimodal knowledge graph with typed relations, inductive splits, and precomputed visual–textual features, yielding WikiArt-MKG-v1.

---

## Citation

If you use this dataset in your research, please cite:
```bibtex
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

If you use the underlying WikiArt-v1 artwork collection, please also cite:
```bibtex
@inproceedings{10.1145/3474085.3475586,
author = {Efthymiou, Athanasios and Rudinac, Stevan and Kackovic, Monika and Worring, Marcel and Wijnberg, Nachoem},
title = {Graph Neural Networks for Knowledge Enhanced Visual Representation of Paintings},
year = {2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3474085.3475586},
doi = {10.1145/3474085.3475586},
pages = {3710–3719},
}
```

---

## License

The dataset metadata is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Images and artwork information are subject to [WikiArt Terms of Use](https://www.wikiart.org/en/terms-of-use).