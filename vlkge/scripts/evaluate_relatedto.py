"""
Semi-supervised IR evaluation for relatedTo relations in WikiArt-MKG-v2.

Evaluates retrieval performance for:
- isRelatedToArtwork (artwork → artwork)
- isRelatedToArtist (artist → artist)

Uses attribute-based relevance (no ground-truth labels) and diversity metrics.

Metrics reported:
- AP@k per attribute (style, period, genre, tag)
- mAP (mean Average Precision)
- ILD (Intra-List Diversity):
  - ILD_props: attribute-based diversity
  - ILD_mod: modality feature diversity

Usage:
    # CLIP retriever for artwork→artwork
    python3 -m vlkge.scripts.evaluate_relatedto \
        --relation isRelatedToArtwork \
        --eval_props hasStyle isCreatedInYear belongsToGenre isAssociatedWithTag \
        --model clip \
        --data_path vlkge/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
        --visual_features_path vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_vf_clip.pkl \
        --ild_features_path vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_vf_vit_b_16.pkl \
        --topk 50

    # KGE retriever for artist→artist
    python3 -m vlkge.scripts.evaluate_relatedto \
        --relation isRelatedToArtist \
        --eval_props isAssociatedWithArtMovement isMemberOfPaintingSchool wasBornOnDate \
        --model kge \
        --kge_model ComplEx \
        --kge_ckpt checkpoints/complex_model.pt \
        --data_path vlkge/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
        --visual_features_path vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_vf_clip.pkl \
        --ild_features_path vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_tf_bert.pkl \
        --topk 20
"""

import argparse
import json
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import vlkge.utils as utils


try:
    from argparse import BooleanOptionalAction
except ImportError:
    import argparse
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, required=False, help=None):
            opts = []
            for s in option_strings:
                opts.append(s)
                if s.startswith("--"):
                    opts.append("--no-" + s[2:])
            super().__init__(option_strings=opts, dest=dest, nargs=0,
                             default=default, required=required, help=help)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))


# ============================================================================
# Helper functions
# ============================================================================

def bin_year(y, step=50):
    """Bin years into intervals for period attribute."""
    try:
        y = int(y)
    except:
        return None
    lo = (y // step) * step
    return f"{lo}-{lo+step}"


def load_data(path, bin_step=50):
    """Load triples CSV and bin years for period attribute."""
    df = pd.read_csv(path)
    m = df["relation"].eq("isCreatedInYear")
    df.loc[m, "tail"] = df.loc[m, "tail"].map(lambda x: bin_year(x, bin_step))
    return df


def build_entity_properties(df, entity_ids, eval_props):
    """
    Build property table for entities.
    Returns: (properties_wide, entity_mode)
    """
    df_rel = df.loc[df["relation"].isin(eval_props), ["head", "relation", "tail", "mode"]].copy()
    df_rel = df_rel[df_rel["head"].isin(set(entity_ids))]
    
    wide = (df_rel.drop_duplicates(["head", "relation"])
            .pivot(index="head", columns="relation", values="tail")
            .reindex(columns=eval_props))
    
    return wide


def get_entities_with_all_props(props_wide):
    """Return list of entity IDs that have all required properties."""
    if props_wide.shape[1] == 0:
        return []
    return list(props_wide.index[props_wide.notna().all(axis=1)])


def build_vector_getter(ild_dict):
    """
    Pre-normalize modality features for fast cosine similarity computation.
    Returns: get_vec(entity_id) -> normalized np.array or None
    """
    cache = {}
    for eid, v in ild_dict.items():
        x = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(x)
        if n > 0:
            cache[eid] = x / n
    
    def get_vec(eid):
        return cache.get(eid, None)
    
    return get_vec


# ============================================================================
# Diversity metrics
# ============================================================================

def compute_diversity(entity_ids, get_vec):
    """
    Compute mean pairwise cosine distance (1 - cosine similarity) for modality features.
    Returns: ILD_mod (mean cosine distance)
    """
    vecs = [get_vec(e) for e in entity_ids]
    vecs = [v for v in vecs if v is not None]
    m = len(vecs)
    
    if m < 2:
        return np.nan
    
    X = np.stack(vecs)  # already unit-normalized
    S = X @ X.T  # cosine similarity matrix
    
    # Mean pairwise cosine similarity (exclude diagonal)
    num = (S.sum() - m)
    den = m * (m - 1)
    mean_sim = float(num / den)
    
    # Convert to distance
    return 1.0 - mean_sim


def compute_attribute_diversity(entity_ids, get_prop_value, eval_props):
    """
    Compute mean pairwise attribute dissimilarity.
    Returns: ILD_props (fraction of differing attributes per pair)
    """
    entity_ids = [e for e in entity_ids if e is not None]
    n = len(entity_ids)
    
    if n < 2:
        return np.nan
    
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs = 0
            count = 0
            for p in eval_props:
                vi = get_prop_value(entity_ids[i], p)
                vj = get_prop_value(entity_ids[j], p)
                if vi is not None and vj is not None:
                    count += 1
                    if vi != vj:
                        diffs += 1
            if count > 0:
                distances.append(diffs / count)
    
    return float(np.mean(distances)) if distances else np.nan


# ============================================================================
# Average Precision (AP) computation
# ============================================================================

def compute_ap_at_k(binary_rels, k=None):
    """
    Compute Average Precision at k.
    Returns: (AP@k, R, p) where R = #positives, p = prevalence
    """
    if k is None:
        k = len(binary_rels)
    
    rels = np.asarray(binary_rels[:k], dtype=int)
    
    if len(rels) == 0:
        return 0.0, 0, np.nan
    
    R = int(rels.sum())
    p = float(R) / float(len(rels))
    
    if R == 0:
        return 0.0, 0, p
    
    cumsum = np.cumsum(rels)
    ranks = np.arange(1, len(rels) + 1)
    precisions = cumsum / ranks
    ap = float((precisions * rels).sum() / R)
    
    return ap, R, p


def build_binary_relevance(retrieved_ids, get_prop_value, query_id, attr):
    """
    Build binary relevance vector: 1 if retrieved item matches query's attribute value.
    """
    q_attr = get_prop_value(query_id, attr)
    rels = []
    
    for rid in retrieved_ids:
        r_attr = get_prop_value(rid, attr)
        is_relevant = (r_attr is not None and q_attr is not None and r_attr == q_attr)
        rels.append(1 if is_relevant else 0)
    
    return rels


# ============================================================================
# Main evaluation function
# ============================================================================

def evaluate(retrieve_fn, get_prop_value, candidate_set, query_set, 
             eval_props, k=50, hide_self=True, get_vec=None):
    """
    Evaluate retrieval performance using attribute-based relevance.
    
    Returns: (per_query_df, summary)
    """
    rows = []
    
    for qid in query_set:
        # Retrieve top-k
        top = retrieve_fn(qid, k) or []
        top_ids = [eid for (eid, _score) in top 
                   if (not hide_self or eid != qid) and eid in candidate_set]
        k_eff = min(k, len(top_ids))
        
        # Compute diversity metrics
        ILD_mod = compute_diversity(top_ids[:k_eff], get_vec) if get_vec else np.nan
        ild_props = compute_attribute_diversity(top_ids[:k_eff], get_prop_value, eval_props)
        
        # Compute AP per attribute
        ap_results = {}
        for prop in eval_props:
            rels = build_binary_relevance(top_ids[:k_eff], get_prop_value, qid, prop)
            ap, R, p = compute_ap_at_k(rels, k=k_eff)
            ap_results[prop] = ap
        
        # Build row
        row = {"query": qid, "k_eff": k_eff}
        
        # AP per attribute
        for prop in eval_props:
            row[f"AP_{prop}"] = ap_results.get(prop, np.nan)
        
        # Diversity
        row["ILD_mod"] = ILD_mod
        row["ILD_props"] = ild_props
        
        rows.append(row)
    
    per_query_df = pd.DataFrame(rows)
    
    # Compute summary statistics
    summary = {}
    
    # mAP per attribute
    for prop in eval_props:
        col = f"AP_{prop}"
        summary[f"mAP_{prop}"] = per_query_df[col].mean()
    
    # Overall mAP (mean across attributes)
    ap_cols = [f"AP_{prop}" for prop in eval_props]
    summary["mAP_overall"] = per_query_df[ap_cols].mean().mean()
    
    # Diversity metrics
    summary["mILD_mod"] = per_query_df["ILD_mod"].mean()
    summary["mILD_props"] = per_query_df["ILD_props"].mean()
    
    return per_query_df, summary


# ============================================================================
# Retriever builders
# ============================================================================

def build_clip_retriever(vis_path, device, allowed_ids=None):
    """Build CLIP-based retriever using cosine similarity."""
    with open(vis_path, "rb") as f:
        vis = pickle.load(f)
    
    ids = sorted((set(vis.keys()) & set(allowed_ids)) if allowed_ids else vis.keys())
    
    if not ids:
        def _empty(*_args, **_kwargs):
            return []
        return _empty, []
    
    mat = torch.tensor(np.stack([vis[a] for a in ids]), dtype=torch.float32, device=device)
    mat = mat / (mat.norm(dim=-1, keepdim=True) + 1e-12)
    idx = {a: i for i, a in enumerate(ids)}
    
    @torch.no_grad()
    def retrieve(head, k):
        if head not in vis:
            return []
        h = torch.tensor(vis[head], dtype=torch.float32, device=device)
        h = h / (h.norm() + 1e-12)
        scores = h @ mat.T
        
        # Hide self
        if head in idx:
            scores[idx[head]] = -1e9
        
        k_eff = min(k, scores.numel())
        vals, inds = torch.topk(scores, k=k_eff)
        return [(ids[i.item()], float(vals[j])) for j, i in enumerate(inds)]
    
    return retrieve, ids


def build_kge_retriever(args, device, emb_dim=768, fusion_mode="average"):
    """Build KGE-based retriever."""
    from vlkge.dataloader import KnowledgeGraphDataLoader
    
    # Import model class
    name = args.kge_model.lower()
    if name == "transe":
        from vlkge.models.transe import TransE as KGE
    elif name == "distmult":
        from vlkge.models.distmult import DistMult as KGE
    elif name == "complex":
        from vlkge.models.complex import ComplEx as KGE
    elif name == "rotate":
        from vlkge.models.rotate import RotatE as KGE
    else:
        raise ValueError(f"Unsupported KGE: {model_name}")
    
    dl = KnowledgeGraphDataLoader(args.data_path, 'wikiart_mkg_v2')
    entity_to_id, relation_to_id = dl.get_entities_and_relations()
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    
    # Load features
    visual_features, textual_features, relation_features, visual_entity_to_index, textual_entity_to_index = \
        utils.load_features(args, entity_to_id, relation_to_id)
    
    # Build model
    model = KGE(
        num_entities=len(entity_to_id),
        num_relations=len(relation_to_id)+1,
        embedding_dim=emb_dim,
        visual_features=visual_features if args.use_visual else None,
        textual_features=textual_features if args.use_textual else None,
        relation_features=relation_features,
        visual_entity_to_index=visual_entity_to_index,
        textual_entity_to_index=textual_entity_to_index,
        fusion_mode=fusion_mode,
        use_structural=args.use_structural,
        use_visual=args.use_visual,
        use_textual=args.use_textual,
        freeze_visual=True,
        freeze_textual=True,
        visual_proj=False,
        textual_proj=False,
        device=device,
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.kge_ckpt, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    
    def make_retriever(pool_tensor):
        @torch.no_grad()
        def retrieve(head, k):
            if head not in entity_to_id or args.relation not in relation_to_id or pool_tensor.numel() == 0:
                return []
            
            hid = entity_to_id[head]
            rid = relation_to_id[args.relation]
            H = torch.full((len(pool_tensor),), hid, dtype=torch.long, device=device)
            R = torch.full((len(pool_tensor),), rid, dtype=torch.long, device=device)
            T = pool_tensor
            
            scores = model(H, R, T)
            
            # Hide self
            self_idx = entity_to_id.get(head, None)
            if self_idx is not None:
                pos = (pool_tensor == self_idx).nonzero(as_tuple=False)
                if pos.numel():
                    scores[pos.view(-1)] = -1e9
            
            k_eff = min(k, scores.numel())
            vals, idxs = torch.topk(scores, k=k_eff)
            return [(id_to_entity[int(T[i])], float(vals[j])) for j, i in enumerate(idxs.tolist())]
        
        return retrieve, entity_to_id
    
    return make_retriever, entity_to_id


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate relatedTo relations')

    # What to evaluate
    parser.add_argument('--relation', type=str, required=True,
                        choices=['isRelatedToArtwork', 'isRelatedToArtist'],
                        help='Relation to evaluate')
    parser.add_argument('--eval_props', nargs='+', required=True,
                        help='Evaluation properties (e.g., hasStyle isCreatedInYear belongsToGenre isAssociatedWithTag)')

    # Retriever choice
    parser.add_argument('--model', type=str, required=True,
                        choices=['clip', 'kge'],
                        help='Retriever type')

    parser.add_argument('--data_path', type=str,
                        default='vlkge/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv',
                        help='Path to triples CSV')
    parser.add_argument('--visual_features_path', type=str,
                        default='vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_vf_clip.pkl',
                        help='Path to visual features (for CLIP retriever and/or KGE visual embeddings)')
    parser.add_argument('--textual_features_path', type=str,
                        default='vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_tf_clip.pkl',
                        help='Path to textual features (used if --use_textual)')
    parser.add_argument('--relation_features_path', type=str,
                        default=None,
                        help='Path to relation features (used if --use_relation_features)')
    parser.add_argument('--ild_features_path', type=str,
                        default='vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_vf_vit_b_16.pkl',
                        help='Path to modality features for ILD computation')

    # KGE specifics
    parser.add_argument('--kge_model', type=str, default='ComplEx',
                        choices=['TransE', 'DistMult', 'ComplEx', 'RotatE'],
                        help='KGE model name (when --model kge)')
    parser.add_argument('--kge_ckpt', type=str, default='',
                        help='Path to KGE checkpoint (required when --model kge)')

    # Feature toggles
    parser.add_argument('--use_structural', action=BooleanOptionalAction, default=True,
                        help='Use structural embeddings (KGE)')
    parser.add_argument('--use_visual', action=BooleanOptionalAction, default=True,
                        help='Use visual features (KGE)')
    parser.add_argument('--use_textual', action=BooleanOptionalAction, default=True,
                        help='Use textual features (KGE)')
    parser.add_argument('--use_relation_features', action=BooleanOptionalAction, default=False,
                        help='Use pretrained relation features (KGE)')

    # Normalization toggles
    parser.add_argument('--normalize_visual', action=BooleanOptionalAction, default=False,
                        help='L2 normalize visual features at load time')
    parser.add_argument('--normalize_textual', action=BooleanOptionalAction, default=False,
                        help='L2 normalize textual features at load time')
    parser.add_argument('--normalize_relation', action=BooleanOptionalAction, default=False,
                        help='L2 normalize relation features at load time')

    # Eval settings
    parser.add_argument('--pool', type=str, default='test', choices=['test', 'all'],
                        help='Candidate pool: test split only or all entities')
    parser.add_argument('--topk', type=int, default=50, help='Number of items to retrieve')
    parser.add_argument('--bin_step', type=int, default=50, help='Year bin size for period binning')

    # Output options
    parser.add_argument('--per_query_output', type=str, default=None,
                        help='Path to save per-query results CSV (optional)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # ---- Expand & validate (respects defaults or CLI overrides)
    args.data_path = utils.validate_path(utils.expand_path(args.data_path), "Triples CSV")
    args.visual_features_path = utils.validate_path(utils.expand_path(args.visual_features_path), "Visual features")
    args.ild_features_path = utils.validate_path(utils.expand_path(args.ild_features_path), "ILD features")

    if args.model == 'kge':
        if not args.kge_ckpt:
            raise ValueError("--kge_ckpt is required when --model kge")

        args.kge_ckpt = utils.validate_path(utils.expand_path(args.kge_ckpt), "KGE checkpoint")

        if args.use_textual:
            if not args.textual_features_path:
                raise ValueError("--textual_features_path is required when --model kge and --use_textual")
            args.textual_features_path = utils.validate_path(
                utils.expand_path(args.textual_features_path), "Textual features"
            )
        if args.use_relation_features:
            if not args.relation_features_path:
                raise ValueError("--relation_features_path is required when --model kge and --use_relation_features")
            args.relation_features_path = utils.validate_path(
                utils.expand_path(args.relation_features_path), "Relation features"
            )

    
    # Setup
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    utils.set_seed(args.seed)

    
    eval_props = args.eval_props
    
    print("\n" + "="*80)
    print(f"RelatedTo Evaluation: {args.relation}")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Pool: {args.pool}")
    print(f"Top-k: {args.topk}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    df = load_data(args.data_path, args.bin_step)
    
    print("Loading modality features for ILD computation...")
    with open(args.ild_features_path, "rb") as f:
        ild_dict = pickle.load(f)
    get_vec = build_vector_getter(ild_dict)

    # Determine candidate pool
    if args.pool == 'test':
        base_pool = set(df.loc[df['mode'].str.lower().eq('test'), 'head'])
    else:
        base_pool = set(df['head'].unique())
    
    print(f"Base pool size: {len(base_pool)}")
    
    # Build entity properties
    print("Building entity properties...")
    props_wide = build_entity_properties(df, sorted(base_pool), eval_props)
    entities_with_props = set(get_entities_with_all_props(props_wide))
    
    # Test entities (queries are always from test set)
    test_entities = set(df.loc[df['mode'].str.lower().eq('test'), 'head'])
    
    # Build retriever and determine final candidate set
    print("Building retriever...")
    if args.model == 'clip':
        candidate_set = entities_with_props
        query_set = sorted(candidate_set & test_entities)
        
        retrieve_fn, _ = build_clip_retriever(
            args.visual_features_path, device,
            allowed_ids=sorted(candidate_set)
        )
    
    else:  # KGE
        if not args.kge_ckpt:
            print("ERROR: --kge_ckpt is required for --model kge", file=sys.stderr)
            sys.exit(1)
        
        make_retriever, entity_to_id = build_kge_retriever(args, device)
        
        # KGE: restrict to entities in knowledge graph
        candidate_set = entities_with_props & set(entity_to_id.keys())
        query_set = sorted(candidate_set & test_entities)
        
        # Build pool tensor
        kge_ids = sorted(entity_to_id[e] for e in candidate_set)
        pool_tensor = torch.tensor(kge_ids, dtype=torch.long, device=device)
        retrieve_fn, _ = make_retriever(pool_tensor)
    
    print(f"Candidate set size: {len(candidate_set)}")
    print(f"Query set size: {len(query_set)}")
    
    # Property value getter
    def get_prop_value(eid, prop):
        try:
            v = props_wide.at[eid, prop]
            return None if pd.isna(v) else v
        except Exception:
            return None
    
    # Run evaluation
    print("\nRunning evaluation...")
    per_query_df, summary = evaluate(
        retrieve_fn, get_prop_value, candidate_set, query_set,
        eval_props, k=args.topk, hide_self=True, get_vec=get_vec
    )
    
    # Print results
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    for key, value in sorted(summary.items()):
        print(f"{key:40s} {value:8.4f}")
    
    if args.per_query_output:
        print(f"\nSaving per-query results to: {args.per_query_output}")
        per_query_df.to_csv(args.per_query_output, index=False)
        print(f"Saved {len(per_query_df)} query results")
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()