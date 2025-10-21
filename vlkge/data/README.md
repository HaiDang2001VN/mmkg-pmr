# VL-KGE Datasets

This directory contains three multimodal knowledge graph datasets used in the VL-KGE paper.

## Overview

| Dataset | Entities | Relations | Train | Val | Test | Modality Coverage |
|---------|----------|-----------|-------|-----|------|-------------------|
| [WN9-IMG](wn9_img/) | 6,555 | 9 | 11,741 | 1,337 | 1,319 | Complete |
| [WikiArt-MKG-v1](wikiart_mkg_v1/) | 76,758 | 4 | 299,968 | 34,020 | 19,695 | Asymmetric |
| [WikiArt-MKG-v2](wikiart_mkg_v2/) | 224,166 | 22 | 7,877,220 | 208,513 | 208,368 | Asymmetric |

---

## Dataset Descriptions

### [WN9-IMG](wn9_img/)

WordNet-based benchmark where all entities possess both visual and textual modalities.

**Key Characteristics:**
- Entities are ImageNet synsets with images and WordNet definitions
- Complete modality coverage (all entities have visual + textual features)
- Evaluates multimodal integration under uniform modality distribution
- Standard benchmark for multimodal KGE methods

**Use Case:** Evaluating how well models integrate visual and textual information when both modalities are always available.

---

### [WikiArt-MKG-v1](wikiart_mkg_v1/)

Fine-art knowledge graph focused on artwork attribute prediction.

**Key Characteristics:**
- Artworks (visual entities) connected to attributes (textual entities)
- 4 relation types: artist, style, year, tags
- Modality asymmetry: artworks have visual features, attributes have textual features
- Inductive setting: test artworks are unseen during training

**Use Case:** Evaluating modality asymmetry handling in a controlled setting with clear entity type separation.

---

### [WikiArt-MKG-v2](wikiart_mkg_v2/)

Large-scale fine-art knowledge graph with rich relational structure.

**Key Characteristics:**
- 3x more entities and 22 relation types
- Includes artist-to-artist relations (influences, pupils, teachers)
- Complex modality asymmetry across multiple entity types
- Multiple inductive evaluation settings
- Relation downsampling and filtering for specific evaluation protocols

**Use Case:** Evaluating real-world multimodal KGE scenarios with heterogeneous entities, complex relations, and modality sparsity.

---

## Data Format

All datasets use the same CSV format for triples:
```csv
head,relation,tail,split
entity_1,relation_name,entity_2,train
entity_3,relation_name,entity_4,val
entity_5,relation_name,entity_6,test
```

**Columns:**
- `head`: Head entity identifier
- `relation`: Relation type
- `tail`: Tail entity identifier
- `split`: Dataset split (`train`, `val`, or `test`)

---

## Features

Precomputed features are provided in each dataset's `features/` subdirectory:

### Visual Features

- **`*_vf_clip.pkl`** - CLIP ViT-B/16 visual features (512-dim)
- **`*_vf_blip.pkl`** - BLIP visual features (768-dim)
- **`*_vf_vit_b_16.pkl`** - ViT-B/16 visual features (768-dim)

### Textual Features

- **`*_tf_clip.pkl`** - CLIP text encoder features (512-dim)
- **`*_tf_blip.pkl`** - BLIP text encoder features (768-dim)
- **`*_tf_bert.pkl`** - BERT text encoder features (768-dim)

### Relation Features

- **`*_rf_clip.pkl`** - CLIP relation features
- **`*_rf_blip.pkl`** - BLIP relation features
- **`*_rf_bert.pkl`** - BERT relation features

All features are stored as PyTorch tensors in pickle format.

---

## Feature Extraction

Visual and textual features were extracted using the following models:

| Model | Visual Encoder | Textual Encoder | Dimensions |
|-------|----------------|-----------------|------------|
| CLIP | ViT-B/16 | Transformer | 512 |
| BLIP | ViT-B/16 | BERT-base | 768 |
| ViT-B/16 | ViT-B/16 | - | 768 |
| BERT | - | BERT-base | 768 |

---

## Usage Example
```python
from vlkge.dataloader import KnowledgeGraphDataLoader

# Load WN9-IMG (complete modalities)
loader_wn9 = KnowledgeGraphDataLoader(
    data_path='data/wn9_img/wn9_img_triples.csv',
    dataset_name='wn9_img',
    bidirectional_eval=True
)

# Load WikiArt-MKG-v1 (modality asymmetry)
loader_wa1 = KnowledgeGraphDataLoader(
    data_path='data/wikiart_mkg_v1/wikiart_mkg_v1_triples.csv',
    dataset_name='wikiart_mkg_v1',
    use_per_relation_candidates=True,
    inductive=True,
    modality_asymmetry=True
)

# Load WikiArt-MKG-v2 (complex modality asymmetry)
loader_wa2 = KnowledgeGraphDataLoader(
    data_path='data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv',
    dataset_name='wikiart_mkg_v2',
    use_per_relation_candidates=True,
    inductive=True,
    modality_asymmetry=True,
    exclude_relations_eval=['isRelatedToArtwork', 'isRelatedToArtist', 'hasCreatedArtwork']
)

# Get splits
train_data, val_data, test_data = loader_wn9.split_data()
```

---

## Download

Dataset files and precomputed features are available at:
- **GitHub Repository:** [https://github.com/thefth/vlkge](https://github.com/thefth/vlkge)

Total download size: ~4 GB (including all features)

---

## Comparison with Other Datasets

| Dataset | Source | Entities | Relations | Modality Coverage | Inductive |
|---------|--------|----------|-----------|-------------------|-----------|
| WN9-IMG | WordNet + ImageNet | 6.6K | 9 | Complete | No |
| FB-IMG | Freebase + ImageNet | 14.9K | 1,345 | Complete | No |
| YAGO15K | YAGO | 15K | 34 | Partial | No |
| WikiArt-MKG-v1 | WikiArt | 76.8K | 4 | Asymmetric | Yes |
| WikiArt-MKG-v2 | WikiArt | 224K | 22 | Asymmetric | Yes |

WikiArt-MKG datasets are distinguished by their explicit modality asymmetry and inductive evaluation settings.

---

## Citation
```bibtex
@inproceedings{efthymiou2025vlkge,
  title={VL-KGE: Vision-Language Models Meet Knowledge Graph Embeddings},
  author={Efthymiou, Athanasios and Rudinac, Stevan and Kackovic, Monika and Wijnberg, Nachoem and Worring, Marcel},
  year={2025},
}
```

---

## License

- **WN9-IMG:** Derived from WordNet and ImageNet (see respective licenses)
- **WikiArt-MKG-v1/v2:** Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Images subject to [WikiArt Terms of Use](https://www.wikiart.org/en/terms-of-use).

---

## References

**Original Datasets:**
- WN9-IMG: Xie et al. "Image-embodied Knowledge Representation Learning" (IJCAI 2017)
- WordNet: Miller, G. A. (1995). "WordNet: A lexical database for English"
- ImageNet: Deng et al. (2009). "ImageNet: A large-scale hierarchical image database"
- WikiArt: [https://www.wikiart.org/](https://www.wikiart.org/)

**Pretrained Models:**
- CLIP: Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (ICML 2021)
- BLIP: Li et al. "BLIP: Bootstrapping Language-Image Pre-training" (ICML 2022)
- ViT: Dosovitskiy et al. "An Image is Worth 16x16 Words" (ICLR 2021)
- BERT: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)