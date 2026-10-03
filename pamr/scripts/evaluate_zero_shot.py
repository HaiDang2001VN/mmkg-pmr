"""
Zero-shot evaluation for WikiArt-MKG datasets using CLIP or BLIP text encoders.
Uses pre-extracted visual embeddings; extracts text embeddings on the fly.

Supports:
- WikiArt-MKG-v1
- WikiArt-MKG-v2

Usage:
# Run as a module (recommended)
    python3 -m vlkge.scripts.evaluate_zero_shot --dataset wikiart_mkg_v1 --model clip
    python3 -m vlkge.scripts.evaluate_zero_shot --dataset wikiart_mkg_v2 --model blip --filtering test
"""

import argparse
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import os

import numpy as np
import pandas as pd
import pickle
import torch
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from lavis.models import load_model_and_preprocess

import vlkge.utils as utils


# ============================================================================
# Dataset-specific configurations
# ============================================================================

DATASET_CONFIGS = {
    'wikiart_mkg_v1': {
        'relation_templates': {
            'hasStyle': "This is a painting in the style of {}.",
            'isCreatedByArtist': "This painting was created by {}.",
            'belongsToTimeframe': "This painting was created during {}.",
            'isAssociatedWithTag': "This painting is associated with the tag {}.",
        },
        'multi_label_relations': ['isAssociatedWithTag'],
        'visual_visual_relations': [],
        'inductive_relations': [],
        'exclude_from_metrics': [],
        'exclude_from_data': [],
    },
    
    'wikiart_mkg_v2': {
        'relation_templates': {
            # Artwork-head
            "isCreatedByArtist": "This painting was created by {}.",
            "belongsToGenre": "This painting belongs to the genre {}.",
            "isAssociatedWithTag": "This painting is associated with the tag {}.",
            "isCreatedWithMedium": "This painting was created using the medium {}.",
            "isLocatedIn": "This painting is located in {}.",
            "hasStyle": "This painting has the style {}.",
            "isCreatedInYear": "This painting was created in the year {}.",
            # Artist-head
            "hasNationality": "This artist has nationality {}.",
            "isAssociatedWithArtMovement": "This artist is associated with the art movement {}.",
            "isAssociatedWithField": "This artist is associated with the field {}.",
            "wasBornIn": "This artist was born in {}.",
            "diedIn": "This artist died in {}.",
            "isAffiliatedWithArtInstitution": "This artist is affiliated with the art institution {}.",
            "isMemberOfPaintingSchool": "This artist is a member of the painting school {}.",
            "wasBornOnDate": "This artist was born during the period {}.",
            "diedOnDate": "This artist died during the period {}.",
            # Artist ↔ Artist
            "isInfluencedBy": "This artist was influenced by {}.",
            "isInfluencedOn": "This artist influenced {}.",
            "isPupilOf": "This artist was a pupil of {}.",
            "isTeacherOf": "This artist was a teacher of {}.",
            "isRelatedTo": "This artist is related to {}.",
            # Artwork ↔ Artwork (visual-visual, no text template)
            "isRelatedToArtwork": None,
            # Inverse
            "hasCreatedArtwork": "This artist created {}.",
        },
        'multi_label_relations': ['isAssociatedWithTag'],
        'visual_visual_relations': ['isRelatedToArtwork'],
        'inductive_relations': ['isInfluencedBy', 'isInfluencedOn', 'isTeacherOf', 'isPupilOf'],
        'exclude_from_metrics': ['isRelatedToArtist', 'isRelatedToArtwork'],
        'exclude_from_data': ['hasCreatedArtwork', 'isRelatedToArtwork', 'isRelatedToArtist'],
    }
}


# ============================================================================
# Argument parsing
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent

parser = argparse.ArgumentParser(description='Zero-shot evaluation for WikiArt-MKG datasets')

parser.add_argument('--dataset', type=str, default='wikiart_mkg_v1',
                    choices=['wikiart_mkg_v1', 'wikiart_mkg_v2'],
                    help='Dataset name')

parser.add_argument('--data_path', type=str,
                    default='vlkge/data/wikiart_mkg_v1/wikiart_mkg_v1_triples.csv',
                    help='Path to triples CSV file')

parser.add_argument('--visual_features_path', type=str,
                    default='vlkge/data/wikiart_mkg_v1/features/wikiart_mkg_v1_vf_clip.pkl',
                    help='Path to visual features pickle file')

parser.add_argument('--model', type=str, default='clip',
                    choices=['clip', 'blip'], help='Text encoder model')

parser.add_argument('--filtering', type=str, default='all',
                    choices=['none', 'test', 'all'],
                    help='Filtered evaluation policy')

parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

# Expand & validate paths (use same helpers as train.py)
args.data_path = utils.validate_path(utils.expand_path(args.data_path), "Triples CSV")
args.visual_features_path = utils.validate_path(utils.expand_path(args.visual_features_path), "Visual features")

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
utils.set_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
utils.set_seed(args.seed)

print("\n" + "="*80)
print(f"Zero-shot Evaluation: {args.dataset.upper()} with {args.model.upper()}")
print("="*80)
print(f"Device: {device}")
print(f"Filtering: {args.filtering}")
print(f"Seed: {args.seed}")
print("="*80 + "\n")

# ============================================================================
# Verify paths
# ============================================================================

print(f"Data path: {args.data_path}")
print(f"Visual features path: {args.visual_features_path}")

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Data file not found: {args.data_path}")
if not os.path.exists(args.visual_features_path):
    raise FileNotFoundError(f"Visual features file not found: {args.visual_features_path}")


# ============================================================================
# Load dataset configuration
# ============================================================================

dataset_config = DATASET_CONFIGS[args.dataset]


# ============================================================================
# Load and preprocess data
# ============================================================================

print("\nLoading data...")
df = pd.read_csv(args.data_path)

# Exclude relations from data if specified
if 'exclude_from_data' in dataset_config and dataset_config['exclude_from_data']:
    exclude_rels = dataset_config['exclude_from_data']
    print(f"Excluding relations from data: {exclude_rels}")
    df = df[~df['relation'].isin(exclude_rels)]

# Create splits
splits = {
    'all': df,
    'test': df[df['mode'] == 'test'].copy()
}
test_df = splits['test']

print(f"\nDataset statistics:")
print(f"  Total triples: {len(df)}")
print(f"  Test triples: {len(test_df)}")
print(f"  Unique relations: {df['relation'].nunique()}")
print(f"  Relations in test: {sorted(test_df['relation'].unique().tolist())}")


# ============================================================================
# Build candidate tails per relation
# ============================================================================

print("\nBuilding candidate sets per relation...")

# For inductive relations, use TEST split candidates
# For transductive relations, use ALL split candidates
inductive_relations = set(dataset_config.get('inductive_relations', []))

tails_all = df.groupby('relation')['tail'].unique()
tails_test = test_df.groupby('relation')['tail'].unique()

relation_tails = {}
all_rels = set(tails_all.index).union(tails_test.index)

for rel in all_rels:
    if rel in inductive_relations:
        # Inductive: use only TEST split candidates
        relation_tails[rel] = tails_test.get(rel, np.array([], dtype=object))
    else:
        # Transductive: use ALL split candidates
        relation_tails[rel] = tails_all.get(rel, np.array([], dtype=object))

print("\nCandidate counts per relation:")
for rel in sorted(relation_tails.keys()):
    n = len(relation_tails[rel]) if relation_tails[rel] is not None else 0
    policy = "INDUCTIVE (test)" if rel in inductive_relations else "TRANSDUCTIVE (all)"
    print(f"  {rel}: {n} candidates [{policy}]")


# ============================================================================
# Build filtered evaluation truth maps
# ============================================================================

print(f"\nBuilding filtered evaluation truth maps (filtering={args.filtering})...")

true_tails_all = defaultdict(set)
true_tails_test = defaultdict(set)

for _, r in df.iterrows():
    true_tails_all[(r['head'], r['relation'])].add(r['tail'])
for _, r in test_df.iterrows():
    true_tails_test[(r['head'], r['relation'])].add(r['tail'])

if args.filtering == 'all':
    true_tails = true_tails_all
    print("  Using ALL splits for filtering")
elif args.filtering == 'test':
    true_tails = true_tails_test
    print("  Using TEST split only for filtering")
else:
    true_tails = defaultdict(set)
    print("  No filtering applied")


# ============================================================================
# Load text encoder model
# ============================================================================

print(f"\nLoading {args.model.upper()} text encoder...")

def load_text_model(which):
    if which == "clip":
        return {
            "tokenizer": CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"),
            "model": CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
        }
    elif which == "blip":
        model, _, _ = load_model_and_preprocess(
            "blip_feature_extractor", model_type="base", is_eval=True, device=device.type
        )
        return {"model": model}
    else:
        raise ValueError("Unsupported text model.")


def encode_text(text, which, mdl):
    if which == "clip":
        inputs = mdl["tokenizer"](text=[text], return_tensors="pt",
                                  padding=True, truncation=True).to(device)
        with torch.no_grad():
            feat = mdl["model"].get_text_features(**inputs).squeeze()
        return feat
    elif which == "blip":
        with torch.no_grad():
            sample = {"text_input": text}
            feat = mdl["model"].extract_features(sample, mode="text").text_embeds_proj[:, 0].squeeze()
        return feat
    else:
        raise ValueError("Unsupported text model.")


text_model = load_text_model(args.model)
print(f"  {args.model.upper()} model loaded successfully")


# ============================================================================
# Encode text candidates for each relation
# ============================================================================

print(f"\nEncoding text candidates with {args.model.upper()}...")

def tail_text_for_prompt(dataset_name: str, raw_tail: str) -> str:
    # Clean tail text (replace hyphens with spaces) - only for WikiArt-MKG-v2
    if dataset_name == 'wikiart_mkg_v2':
        return str(raw_tail).replace('-', ' ')
    else:
        return str(raw_tail)
        
text_embeddings = {}      # relation -> (N, d) tensor on device
relation_tail_map = {}    # relation -> {tail_str: index}

relation_templates = dataset_config['relation_templates']

for relation, tails in relation_tails.items():
    template = relation_templates.get(relation, None)
    if template is None or tails is None or len(tails) == 0:
        continue

    emb_list = []
    tail_map = {}

    for idx, raw_tail in enumerate(tqdm(tails, desc=f"  [{relation}]", leave=False)):
        tail_key  = str(raw_tail)                               # <-- keep original for lookups
        tail_text = tail_text_for_prompt(args.dataset, raw_tail) # <-- normalized for prompt only
        prompt_text = template.format(tail_text)

        emb = encode_text(prompt_text, args.model, text_model).float().to(device)
        emb = emb / (emb.norm() + 1e-12)
        emb_list.append(emb.unsqueeze(0))
        tail_map[tail_key] = idx                                # <-- map by ORIGINAL tail

    if emb_list:
        text_embeddings[relation] = torch.cat(emb_list, dim=0)
        relation_tail_map[relation] = tail_map

print(f"\nEncoded text candidates for {len(text_embeddings)} relations")


# ============================================================================
# Load visual embeddings
# ============================================================================

print(f"\nLoading visual embeddings from {args.visual_features_path}...")

with open(args.visual_features_path, 'rb') as f:
    visual_embeddings = pickle.load(f)

print(f"  Loaded {len(visual_embeddings)} visual embeddings")

# Check dimensionality consistency
text_dim = None
for rel in text_embeddings:
    if text_embeddings[rel].numel() > 0:
        text_dim = text_embeddings[rel].shape[-1]
        break

visual_dim = None
if len(visual_embeddings) > 0:
    some_key = next(iter(visual_embeddings))
    visual_dim = visual_embeddings[some_key].shape[-1] if hasattr(visual_embeddings[some_key], 'shape') else None

if text_dim is not None and visual_dim is not None:
    if text_dim != visual_dim:
        print(f"\n  WARNING: Text dim ({text_dim}) != Visual dim ({visual_dim})")
        print(f"    Ensure visual features match the '{args.model}' backbone!")
    else:
        print(f"  Dimensionality check: text={text_dim}, visual={visual_dim}")


# ============================================================================
# Prepare visual-visual relation candidates (if applicable)
# ============================================================================

visual_visual_relations = dataset_config.get('visual_visual_relations', [])

if len(visual_visual_relations) > 0:
    print(f"\nPreparing visual-visual relation candidates...")
    artwork_candidates = sorted(set(visual_embeddings.keys()))
    artwork_to_idx = {a: i for i, a in enumerate(artwork_candidates)}
    
    if len(artwork_candidates) > 0:
        cand_art_mat = torch.tensor(
            np.stack([visual_embeddings[a] for a in artwork_candidates]),
            dtype=torch.float32, device=device
        )
        cand_art_mat = cand_art_mat / (cand_art_mat.norm(dim=-1, keepdim=True) + 1e-12)
        print(f"  Built visual candidate matrix: {cand_art_mat.shape}")
    else:
        cand_art_mat = None
        artwork_to_idx = {}
else:
    cand_art_mat = None
    artwork_to_idx = {}


# ============================================================================
# Head embedding helper
# ============================================================================

@lru_cache(maxsize=100_000)
def encode_head_text_cached(head_text: str):
    """Encode head as text (cached for efficiency)."""
    t = encode_text(head_text, args.model, text_model).float().to(device)
    return (t / (t.norm() + 1e-12)).detach()


def get_head_embedding(head_str, relation):
    """
    Get normalized head embedding.
    - If head is in visual_embeddings, use visual embedding
    - Otherwise, encode as text
    """
    if head_str in visual_embeddings:
        v = torch.tensor(visual_embeddings[head_str], dtype=torch.float32, device=device)
        return v / (v.norm() + 1e-12)
    
    # Encode as text
    raw = head_str.replace('-', ' ')
    return encode_head_text_cached(raw)


# ============================================================================
# Evaluation function
# ============================================================================

def evaluate_zero_shot(test_data: pd.DataFrame):
    """
    Perform zero-shot evaluation on test data.
    Returns relation-specific and overall metrics.
    """
    relation_ranks = defaultdict(list)
    exclude_from_metrics = set(dataset_config.get('exclude_from_metrics', []))
    
    print("\nEvaluating zero-shot predictions...")
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), 
                       desc=f"  Zero-shot eval"):
        head = row['head']
        relation = row['relation']
        tail = row['tail']
        
        # Skip relations excluded from metrics
        if relation in exclude_from_metrics:
            continue
        
        # ---- Visual-visual relations ----
        if relation in visual_visual_relations:
            if cand_art_mat is None:
                continue
            if head not in visual_embeddings:
                continue
            if tail not in artwork_to_idx:
                continue
            
            head_emb = torch.tensor(visual_embeddings[head], dtype=torch.float32, device=device)
            head_emb = head_emb / (head_emb.norm() + 1e-12)
            
            scores = head_emb @ cand_art_mat.T  # (N_art,)
            
            # Apply filtering
            if args.filtering != 'none':
                mask = torch.ones_like(scores, dtype=torch.bool, device=device)
                for t in true_tails.get((head, relation), set()):
                    j = artwork_to_idx.get(t)
                    if j is not None:
                        mask[j] = False
                focal_pos = artwork_to_idx.get(tail, None)
                if focal_pos is None:
                    continue
                mask[focal_pos] = True
                scores = torch.where(mask, scores, torch.tensor(float('-inf'), device=device))
            
            sorted_idx = torch.argsort(scores, descending=True)
            true_index = artwork_to_idx[tail]
            rank = (sorted_idx == true_index).nonzero(as_tuple=True)[0].item() + 1
            relation_ranks[relation].append(rank)
            continue
        
        # ---- Text-based relations ----
        if relation not in text_embeddings:
            continue
        
        focal_idx = relation_tail_map[relation].get(tail, None)
        if focal_idx is None:
            continue
        
        head_emb = get_head_embedding(head, relation)      # (d,)
        cand_embs = text_embeddings[relation]              # (N, d)
        scores = head_emb @ cand_embs.T                    # (N,)
        
        # Apply filtering
        if args.filtering != 'none':
            mask = torch.ones_like(scores, dtype=torch.bool, device=device)
            for t in true_tails.get((head, relation), set()):
                j = relation_tail_map[relation].get(t, None)
                if j is not None:
                    mask[j] = False
            mask[focal_idx] = True
            scores = torch.where(mask, scores, torch.tensor(float('-inf'), device=device))
        
        sorted_idx = torch.argsort(scores, descending=True)
        rank = (sorted_idx == focal_idx).nonzero(as_tuple=True)[0].item() + 1
        relation_ranks[relation].append(rank)
    
    return relation_ranks


# ============================================================================
# Compute mAP for multi-label relations
# ============================================================================

def compute_map_for_multi_label(test_df: pd.DataFrame, relation: str):
    """
    Compute mAP for multi-label relations (e.g., tags).
    """
    if relation not in text_embeddings or relation not in relation_tail_map:
        print(f"\n[mAP] No candidates/embeddings for {relation}")
        return None
    
    # Build reverse map: index -> tail
    tag_embs = text_embeddings[relation]  # (N, d) normalized
    idx2tag = [None] * tag_embs.shape[0]
    for t, i in relation_tail_map[relation].items():
        if 0 <= i < len(idx2tag):
            idx2tag[i] = t
    
    # Filter test rows with this relation
    eval_rows = test_df[test_df['relation'] == relation]
    if eval_rows.empty:
        print(f"\n[mAP] No samples for {relation} in test set")
        return None
    
    # Group by head
    groups = eval_rows.groupby('head')
    
    def average_precision(y_true_binary: np.ndarray, scores: np.ndarray) -> float:
        """Compute Average Precision."""
        order = np.argsort(-scores)  # descending
        y_sorted = y_true_binary[order]
        hits = 0
        prec_sum = 0.0
        for k in range(1, len(y_sorted) + 1):
            if y_sorted[k - 1] == 1:
                hits += 1
                prec_sum += hits / k
        pos = int(y_true_binary.sum())
        return (prec_sum / pos) if pos > 0 else 0.0
    
    ap_scores = []
    skipped = 0
    
    for head, g in tqdm(groups, desc=f"  Computing mAP [{relation}]", leave=False):
        try:
            head_emb = get_head_embedding(head, relation)  # normalized
        except Exception:
            skipped += 1
            continue
        
        # Compute scores
        with torch.no_grad():
            scores = (head_emb @ tag_embs.T).detach().cpu().numpy()
        
        # Ground truth positives (from filtered truth map)
        gt = true_tails.get((head, relation), set())
        y_true = np.array([1 if (t is not None and t in gt) else 0 for t in idx2tag], dtype=np.int32)
        
        if y_true.sum() == 0:
            skipped += 1
            continue
        
        ap = average_precision(y_true, scores)
        ap_scores.append(ap)
    
    if len(ap_scores) == 0:
        print(f"\n[mAP] No valid samples for {relation} (skipped: {skipped})")
        return None
    
    mAP = float(np.mean(ap_scores))
    print(f"\n mAP ({relation}): {mAP:.4f}  (images: {len(ap_scores)}, skipped: {skipped})")
    return mAP


# ============================================================================
# Run evaluation and report results
# ============================================================================

relation_ranks = evaluate_zero_shot(test_df)


# ============================================================================
# Report metrics
# ============================================================================

def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else float('nan')


print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

# Per-relation metrics
print("\n--- Per-Relation Metrics ---")
for rel in sorted(relation_ranks.keys()):
    ranks = np.array(relation_ranks[rel], dtype=np.int32)
    if len(ranks) == 0:
        continue
    
    mrr = safe_mean(1.0 / ranks)
    h1 = safe_mean(ranks <= 1)
    h3 = safe_mean(ranks <= 3)
    h10 = safe_mean(ranks <= 10)
    
    print(f"\n{rel}:")
    print(f"  Samples: {len(ranks)}")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Hits@1:  {h1:.4f}")
    print(f"  Hits@3:  {h3:.4f}")
    print(f"  Hits@10: {h10:.4f}")


# Overall (micro-averaged) metrics
all_ranks = np.array([r for rs in relation_ranks.values() for r in rs], dtype=np.int32)

if len(all_ranks) == 0:
    print("\n  No evaluable samples found!")
else:
    overall_mrr = safe_mean(1.0 / all_ranks)
    overall_h1 = safe_mean(all_ranks <= 1)
    overall_h3 = safe_mean(all_ranks <= 3)
    overall_h10 = safe_mean(all_ranks <= 10)
    
    print("\n" + "="*80)
    print("--- Overall (Micro-Averaged) Metrics ---")
    print(f"Total Samples: {len(all_ranks)}")
    print(f"MRR:           {overall_mrr:.4f}")
    print(f"Hits@1:        {overall_h1:.4f}")
    print(f"Hits@3:        {overall_h3:.4f}")
    print(f"Hits@10:       {overall_h10:.4f}")


# Macro-averaged metrics
if len(relation_ranks) > 0:
    macro_mrr = safe_mean([np.mean(1.0 / np.array(rs)) for rs in relation_ranks.values() if len(rs) > 0])
    macro_h1 = safe_mean([np.mean(np.array(rs) <= 1) for rs in relation_ranks.values() if len(rs) > 0])
    macro_h3 = safe_mean([np.mean(np.array(rs) <= 3) for rs in relation_ranks.values() if len(rs) > 0])
    macro_h10 = safe_mean([np.mean(np.array(rs) <= 10) for rs in relation_ranks.values() if len(rs) > 0])
    
    print("\n" + "="*80)
    print("--- Macro-Averaged Metrics ---")
    print(f"Macro MRR:     {macro_mrr:.4f}")
    print(f"Macro Hits@1:  {macro_h1:.4f}")
    print(f"Macro Hits@3:  {macro_h3:.4f}")
    print(f"Macro Hits@10: {macro_h10:.4f}")


# Compute mAP for multi-label relations
multi_label_relations = dataset_config.get('multi_label_relations', [])
if len(multi_label_relations) > 0:
    print("\n" + "="*80)
    print("--- Multi-Label Evaluation (mAP) ---")
    for rel in multi_label_relations:
        if rel in test_df['relation'].values:
            compute_map_for_multi_label(test_df, rel)


print("\n" + "="*80)
print("Evaluation Complete!")
print("="*80 + "\n")