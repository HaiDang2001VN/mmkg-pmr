# WikiArt-MKG-v2 Dataset

## Overview

**WikiArt-MKG-v2** is a large-scale fine-art multimodal knowledge graph that substantially extends WikiArt-MKG-v1 with richer metadata, broader coverage, and more complex relational structure. It was introduced in the paper *"VL-KGE: Vision-Language Models Meet Knowledge Graph Embeddings"* and is designed to evaluate multimodal knowledge graph embedding methods under realistic conditions including modality asymmetry, inductive learning, and heterogeneous relation types.

WikiArt-MKG-v2 features 3x more entities than v1, with 22 relation types spanning artwork-to-attribute relations, artist-to-attribute relations, and artist-to-artist relations (influences, pupils, teachers).

---

## Dataset Structure

| File | Description |
|------|--------------|
| **`wikiart_mkg_v2_triples.csv`** | Knowledge graph triples with diverse relation types. |
| **`wikiart_v2_artworks.csv`** | Metadata for artworks including titles, URLs, and detailed attributes. |
| **`wikiart_v2_artists.csv`** | Metadata for artists including biographical information and relationships. |
| **`features/`** | Directory containing precomputed visual, textual, and relation features. |

---

## Statistics

- **Total Entities:** 224,166
  - Visual entities (artworks): 216,564
  - Textual entities (attributes): 7,602
- **Relations:** 22
- **Total Triples:** 8,294,101
  - Train: 7,877,220
  - Validation: 208,513
  - Test: 208,368

---

## Data Format

**`wikiart_mkg_v2_triples.csv`** contains the following columns:

| Column | Description |
|---------|-------------|
| `head` | Head entity identifier (artwork, artist, or attribute). |
| `relation` | Relation type connecting head to tail entity. |
| `tail` | Tail entity identifier. |
| `split` | Dataset split (`train`, `val`, or `test`). |

---

## Relations

The dataset contains 22 relation types organized into three categories:

### Artwork-to-Attribute Relations (10)

1. **`isCreatedByArtist`** - Links artwork to its creator
2. **`hasStyle`** - Links artwork to artistic style/movement
3. **`belongsToGenre`** - Links artwork to genre (portrait, landscape, etc.)
4. **`isCreatedInYear`** - Links artwork to creation year
5. **`isCreatedWithMedium`** - Links artwork to medium (oil, watercolor, etc.)
6. **`isAssociatedWithTag`** - Links artwork to descriptive tags
7. **`isLocatedIn`** - Links artwork to exhibition location/museum

### Artist-to-Attribute Relations (10)

8. **`hasNationality`** - Links artist to nationality
9. **`isAssociatedWithField`** - Links artist to field of work
10. **`isAssociatedWithArtMovement`** - Links artist to art movement
11. **`wasBornOnDate`** - Links artist to birth date
12. **`wasBornIn`** - Links artist to birthplace
13. **`diedOnDate`** - Links artist to death date
14. **`diedIn`** - Links artist to place of death
15. **`isMemberOfPaintingSchool`** - Links artist to painting school
16. **`isAffiliatedWithArtInstitution`** - Links artist to institution

### Artist-to-Artist Relations (4)

17. **`isInfluencedBy`** - Artist influenced by another artist
18. **`isInfluencedOn`** - Artist influenced another artist (inverse)
19. **`isPupilOf`** - Student-teacher relationship
20. **`isTeacherOf`** - Teacher-student relationship (inverse)

### Similarity Relations (2)

21. **`isRelatedToArtwork`** - Semantic similarity between artworks
22. **`isRelatedToArtist`** - Semantic similarity between artists

**Note:** Relations 21-22 are evaluated separately using retrieval metrics (see paper Appendix A.2).

---

## Modality Asymmetry

WikiArt-MKG-v2 exhibits complex **modality asymmetry** across entity types:

- **Artworks:** Visual features (images)
- **Artists, Styles, Movements, etc.:** Textual features (names/descriptions)
- **Some entities:** May lack one or both modalities

This heterogeneous modality distribution reflects real-world knowledge graphs where modality availability depends on entity type and data completeness.

---

## Evaluation Protocol

- **Task:** Tail prediction (predict attributes, related entities)
- **Metrics:** MRR, Hits@1, Hits@3, Hits@10
- **Filtering:** Filtered ranking with per-relation candidate pools
- **Inductive Setting:** 
  - All test artworks are unseen during training
  - Artist-to-artist relations use disjoint artist subsets
  - Artwork-to-artist relations: artists appear in training but connections to test artworks are unseen

### Special Handling

- **Excluded from evaluation:** `isRelatedToArtwork`, `isRelatedToArtist`, `hasCreatedArtwork` (inverse of `isCreatedByArtist`)
- **Downsampling:** `isRelatedToArtwork` triples are downsampled to 0.1% per epoch during training

---

## Features

Precomputed features are available in the `features/` directory:

| File | Description | Dimensions |
|------|-------------|------------|
| `wikiart_mkg_v2_vf_clip.pkl` | CLIP visual features | 216,564 × 512 |
| `wikiart_mkg_v2_tf_clip.pkl` | CLIP textual features | 7,602 × 512 |
| `wikiart_mkg_v2_vf_blip.pkl` | BLIP visual features | 216,564 × 768 |
| `wikiart_mkg_v2_tf_blip.pkl` | BLIP textual features | 7,602 × 768 |
| `wikiart_mkg_v2_vf_vit_b_16.pkl` | ViT-B/16 visual features | 216,564 × 768 |
| `wikiart_mkg_v2_tf_bert.pkl` | BERT textual features | 7,602 × 768 |
| `wikiart_mkg_v2_rf_clip.pkl` | CLIP relation features | 22 × 512 |
| `wikiart_mkg_v2_rf_blip.pkl` | BLIP relation features | 22 × 768 |
| `wikiart_mkg_v2_rf_bert.pkl` | BERT relation features | 22 × 768 |

---

## Usage Example
```python
from vlkge.dataloader import KnowledgeGraphDataLoader

loader = KnowledgeGraphDataLoader(
    data_path='data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv',
    dataset_name='wikiart_mkg_v2',
    use_per_relation_candidates=True,
    bidirectional_eval=False,
    inductive=True,
    modality_asymmetry=True,
    exclude_relations_eval=['isRelatedToArtwork', 'isRelatedToArtist', 'hasCreatedArtwork'],
    add_inverse_relations={'isCreatedByArtist': 'hasCreatedArtwork'},
    artist2artist_relations=['isInfluencedBy', 'isInfluencedOn', 'isPupilOf', 'isTeacherOf']
)

train_data, val_data, test_data = loader.split_data()
```

---

## Training Configuration

See `configs/wikiart_mkg_v2/` for complete training configurations.

Typical training settings:
- **Epochs:** 20
- **Batch size:** 512
- **Learning rate:** 0.1
- **Negative samples:** 100
- **Runtime:** ~4 hours per model (A100 GPU)

---

## Data Origin

The WikiArt-MKG-v2 dataset is constructed through large-scale web scraping from [WikiArt.org](https://www.wikiart.org/), substantially extending WikiArt-MKG-v1 with broader coverage of artworks, artists, and enriched metadata including exhibition locations, artistic movements, and artist relationships.

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

The dataset metadata is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Images and artwork information are subject to [WikiArt Terms of Use](https://www.wikiart.org/en/terms-of-use).