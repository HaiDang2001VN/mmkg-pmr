# WikiArt-MKG-v2 Dataset

## Overview

**WikiArt-MKG-v2** is a large-scale fine-art multimodal knowledge graph that substantially extends WikiArt-MKG-v1 with richer metadata, broader coverage, and more complex relational structure. It was introduced in the paper **["VL-KGE: Vision–Language Models Meet Knowledge Graph Embeddings" (WWW '26)](https://doi.org/10.1145/3774904.3792677)**.

In the same work, we also introduce **WikiArt-v2**, a large-scale artwork and artist collection constructed from WikiArt.org, which serves as the underlying data source for WikiArt-MKG-v2.

WikiArt-MKG-v2 is designed to evaluate multimodal knowledge graph embedding methods under realistic conditions including modality asymmetry, inductive learning, and heterogeneous relation types.


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
| `mode` | Dataset split (`train`, `val`, or `test`). |

---

## Relations

The dataset contains 22 relation types organized into three categories:

### Artwork-to-Attribute Relations (7)

1. **`isCreatedByArtist`** - Links artwork to its creator.
2. **`hasStyle`** - Links artwork to artistic style/movement.
3. **`belongsToGenre`** - Links artwork to genre (portrait, landscape, etc.).
4. **`isCreatedInYear`** - Links artwork to creation year.
5. **`isCreatedWithMedium`** - Links artwork to medium (oil, watercolor, etc.).
6. **`isAssociatedWithTag`** - Links artwork to descriptive tags.
7. **`isLocatedIn`** - Links artwork to exhibition location/museum.

### Artist-to-Attribute Relations (9)

8. **`hasNationality`** - Links artist to nationality.
9. **`isAssociatedWithField`** - Links artist to field of work.
10. **`isAssociatedWithArtMovement`** - Links artist to art movement.
11. **`wasBornOnDate`** - Links artist to birth date.
12. **`wasBornIn`** - Links artist to birthplace.
13. **`diedOnDate`** - Links artist to death date.
14. **`diedIn`** - Links artist to place of death.
15. **`isMemberOfPaintingSchool`** - Links artist to painting school.
16. **`isAffiliatedWithArtInstitution`** - Links artist to institution.

### Artist-to-Artist Relations (4)

17. **`isInfluencedBy`** - Artist influenced by another artist.
18. **`isInfluencedOn`** - Artist influenced another artist (inverse).
19. **`isPupilOf`** - Student-teacher relationship.
20. **`isTeacherOf`** - Teacher-student relationship (inverse).

### Similarity Relations (2)

21. **`isRelatedToArtwork`** - Semantic similarity between artworks.
22. **`isRelatedToArtist`** - Semantic similarity between artists.

**Note:** Relations 21-22 are evaluated separately using retrieval metrics (see paper Appendix A.1).

---

## Modality Asymmetry

WikiArt-MKG-v2 exhibits complex **modality asymmetry** across entity types:

- **Artworks:** Visual features (images).
- **Artists, Styles, Movements, etc.:** Textual features (names/descriptions).
- **Some entities:** May lack one or both modalities.

---

## Evaluation Protocol

- **Task:** Tail prediction (predict attributes, related entities).
- **Metrics:** MRR, Hits@1, Hits@3, Hits@10.
- **Filtering:** Filtered ranking with per-relation candidate pools.
- **Inductive Setting:** 
  - All test artworks are unseen during training.
  - Artist-to-artist relations use disjoint artist subsets.
  - Artwork-to-artist relations: artists appear in training but connections to test artworks are unseen.

### Special Handling

- **Excluded from evaluation:** `isRelatedToArtwork`, `isRelatedToArtist`, `hasCreatedArtwork` (inverse of `isCreatedByArtist`).
- **Downsampling:** `isRelatedToArtwork` triples are downsampled to 0.1% per epoch during training.

---

## Features

Precomputed features are available in the `features/` directory:

| File | Description | Dimensions |
|------|-------------|------------|
| `wikiart_mkg_v2_vf_clip.pkl` | CLIP visual features | 216,564 × 768 |
| `wikiart_mkg_v2_tf_clip.pkl` | CLIP textual features | 7,602 × 768 |
| `wikiart_mkg_v2_vf_blip.pkl` | BLIP visual features | 216,564 × 256 |
| `wikiart_mkg_v2_tf_blip.pkl` | BLIP textual features | 7,602 × 256 |
| `wikiart_mkg_v2_vf_vit_b_16.pkl` | ViT-B/16 visual features | 216,564 × 768 |
| `wikiart_mkg_v2_tf_bert.pkl` | BERT textual features | 7,602 × 768 |
| `wikiart_mkg_v2_rf_clip.pkl` | CLIP relation features | 21 × 768 |
| `wikiart_mkg_v2_rf_blip.pkl` | BLIP relation features | 21 × 256 |
| `wikiart_mkg_v2_rf_bert.pkl` | BERT relation features | 21 × 768 |

##### Note: Relation features are provided for 21 relations (excluding `isRelatedToArtwork`, which is visual-visual). Pre-computed relation features are never used in VL-KGE experiments but are provided for potential future work.
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

## WikiArt-v2 Dataset Curation and Construction

The WikiArt-v2 was constructed through large-scale web scraping of the WikiArt online collection (https://www.wikiart.org), a comprehensive repository of fine art spanning multiple centuries, artistic movements, and geographic regions. The WikiArt-v2 extends WikiArt-v1 in both scale and semantic richness, increasing coverage from approximately 76K to over 216K artworks and from 750 to over 4K artists, alongside thousands of additional attribute entities (styles, movements, genres, years, locations, etc.).

### Data Collection and Curation

We implemented a multi-stage scraping pipeline to collect artwork images, artist profiles, and associated metadata. For each artwork, we extracted high-resolution images, basic metadata (title, year, dimensions), categorical attributes (style, genre, medium), and relational metadata. For artists, we collected biographical information, movement affiliations, and inter-artist relationships.

To ensure data quality, we applied several filtering and normalization steps:

- Artworks with missing images or critical metadata (artist, creation year) were excluded.
- Entity names were normalized to resolve aliases and spelling variations, particularly for artists and locations with multiple language representations.
- Artist relationships were retained only when explicitly documented in the source metadata.
- Attributes occurring fewer than 10 times were removed to reduce noise from rare or potentially erroneous labels.
- Artist-to-artist relations were exempt from frequency filtering due to their inherently sparse nature.
- Birth and death dates were mapped to half-century time bins (e.g., 1600–1650).
- Fine-grained locations were mapped to countries to reduce geographic fragmentation.

Unlike curated benchmarks such as WN9-IMG where all entities possess complete modalities by design, WikiArt-MKG-v2 exhibits natural modality sparsity reflecting real-world data characteristics.

### WikiArt-MKG-v2 Knowledge Graph Construction

From the curated WikiArt-v2 collection, we constructed a multimodal knowledge graph with 22 relation types spanning:

- artwork-to-attribute relations (e.g., `isCreatedByArtist`, `hasStyle`, `belongsToGenre`),
- artist-to-attribute relations (e.g., `hasNationality`, `wasBornIn`, `isAssociatedWithArtMovement`),
- artist-to-artist relations (e.g., `isInfluencedBy`, `isPupilOf`),
- similarity relations (`isRelatedToArtwork`, `isRelatedToArtist`).

For each artwork–attribute pair in the metadata, we generated a corresponding triple. Artist-to-artist relations were extracted from structured relationship fields in artist profiles. Inverse relations were validated, and contradictory or duplicate triples were removed.

Similarity relations were constructed using WikiArt’s own relatedness annotations, which capture stylistic, temporal, and thematic similarity.

### Inductive Split Construction

To enable systematic evaluation of inductive inference:

- All test artworks are unseen during training.
- Artist-to-artist relations use disjoint artist subsets.
- For artwork-to-artist relations, artists appear in training through other artworks, but connections to test artworks are unseen.

This setup reflects realistic deployment scenarios in which new artworks are continuously added and their attributes must be predicted without retraining.

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

---

## License

The dataset metadata is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Images and artwork information are subject to [WikiArt Terms of Use](https://www.wikiart.org/en/terms-of-use).