"""
Unified helper functions for VL-KGE training and evaluation
Supports: WN9-IMG, WikiArt-v1, WikiArt-v2
"""

import numpy as np
import pandas as pd
import os
import datetime
import torch
import torch.nn.functional as F
import pamr.models as models
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from pamr.dataloader import KGDataset
from pamr import utils


# ==================== Loss Functions ====================

def compute_logistic_loss(positive_scores, negative_scores):
    """
    Computes the logistic loss for KGE training.
    
    Args:
        positive_scores: Scores for positive triples, shape (batch_size,)
        negative_scores: Scores for negative triples, shape (batch_size, num_negatives)
    
    Returns:
        Scalar loss
    """
    positive_labels = torch.ones_like(positive_scores.unsqueeze(1))
    negative_labels = -torch.ones_like(negative_scores)
    
    all_scores = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
    all_labels = torch.cat([positive_labels, negative_labels], dim=1)
    
    loss = F.softplus(-all_labels * all_scores)
    return torch.mean(loss)


# ==================== Negative Sampling ====================

def negative_sampling_uniform(heads, relations, tails, num_entities, num_neg_samples, 
                              filter_map, relation_probs, generator, 
                              use_bernoulli=False):
    """
    Uniform negative sampling with optional Bernoulli corruption strategy.
    Used for WN9-IMG (bidirectional corruption).
    """
    batch_size = heads.size(0)
    device = heads.device
    
    # Legacy behavior corrupts with a fixed 50/50 batch-level mask. With
    # use_bernoulli=True the corruption direction is drawn per triple from the
    # relation-specific probability p(corrupt head | r).
    corrupt_head_mask = torch.rand(batch_size, generator=generator) > 0.5  # Boolean tensor
    if use_bernoulli and relation_probs is not None:
        corrupt_head_mask = torch.zeros(batch_size, dtype=torch.bool)
        for i in range(batch_size):
            relation = int(relations[i].item())
            p_head = float(relation_probs.get(relation, 0.5))
            corrupt_head_mask[i] = torch.rand(
                (1,), generator=generator).item() < p_head
    
    neg_heads_list, neg_tails_list = [], []
    
    for i in range(batch_size):
        head, relation, tail = heads[i].item(), relations[i].item(), tails[i].item()
        
        if corrupt_head_mask[i]:  # Use pre-computed mask
            # Corrupt head
            valid_heads = filter_map.get_heads((tail, relation), set())
            sampled_heads = set()
            attempts = 0
            max_attempts = max(num_neg_samples * 100, 10000)
            while len(sampled_heads) < num_neg_samples and attempts < max_attempts:
                neg_head = torch.randint(0, num_entities, (1,), generator=generator).item()
                if neg_head not in valid_heads:
                    sampled_heads.add(neg_head)
                attempts += 1
            if len(sampled_heads) < num_neg_samples:
                raise RuntimeError(
                    "Could not sample head negative for relation "
                    f"{relation}: no eligible candidates remain"
                )
            neg_heads_list.extend(sampled_heads)
            neg_tails_list.extend([tail] * num_neg_samples)
        else:
            # Corrupt tail
            valid_tails = filter_map.get_tails((head, relation), set())
            sampled_tails = set()
            attempts = 0
            max_attempts = max(num_neg_samples * 100, 10000)
            while len(sampled_tails) < num_neg_samples and attempts < max_attempts:
                neg_tail = torch.randint(0, num_entities, (1,), generator=generator).item()
                if neg_tail not in valid_tails:
                    sampled_tails.add(neg_tail)
                attempts += 1
            if len(sampled_tails) < num_neg_samples:
                raise RuntimeError(
                    "Could not sample tail negative for relation "
                    f"{relation}: no eligible candidates remain"
                )
            neg_heads_list.extend([head] * num_neg_samples)
            neg_tails_list.extend(sampled_tails)
    
    neg_heads = torch.tensor(neg_heads_list, dtype=torch.long, device=device)
    neg_tails = torch.tensor(neg_tails_list, dtype=torch.long, device=device)
    neg_relations = relations.repeat_interleave(num_neg_samples)
    
    return neg_heads, neg_relations, neg_tails


def negative_sampling_per_relation(heads, relations, tails, num_neg_samples,
                                   filter_map, relation_to_valid_tails, generator):
    """
    Per-relation negative sampling WITHOUT replacement (WikiArt-v1).
    
    Samples negatives without replacement using torch.randperm. Filters all valid
    triples before sampling. May return fewer than num_neg_samples negatives if
    insufficient candidates are available after filtering.
    
    Used in WikiArt-v1 experiments for reproducibility.
    
    Characteristics:
    - Sampling: WITHOUT replacement (each negative at most once per triple)
    - Batch size: Variable (may be < batch_size * num_neg_samples)
    - Method: Random permutation of filtered candidates
    
    For larger datasets or faster training, use negative_sampling_per_relation_fast.
    
    Args:
        heads: Head entity IDs (batch_size,)
        relations: Relation IDs (batch_size,)
        tails: Tail entity IDs (batch_size,)
        num_neg_samples: Maximum number of negative samples per positive triple
        filter_map: Dict mapping (head_id, relation_id) -> set of valid tail_ids
        relation_to_valid_tails: Dict mapping relation_id -> list of valid tail candidates
        generator: Random generator for reproducibility
    
    Returns:
        neg_heads, neg_relations, neg_tails
        (Output size may be < batch_size * num_neg_samples)
    """
    batch_size = heads.size(0)
    device = heads.device
    
    neg_heads_list = []
    neg_relations_list = []
    neg_tails_list = []
    
    for i in range(batch_size):
        h = heads[i].item()
        r = relations[i].item()
        t = tails[i].item()
        
        # Get all valid negative candidates for this relation
        if r not in relation_to_valid_tails:
            # Skip if relation not in pool
            continue
        
        possible_neg_tails = list(relation_to_valid_tails[r])
        
        # Remove valid tails from negative candidates
        valid_tails = filter_map.get_tails((h, r), set())
        candidate_neg_tails = list(set(possible_neg_tails) - valid_tails)
        
        if len(candidate_neg_tails) == 0:
            # No valid negatives available, skip this triple
            continue
        
        # Sample negatives WITHOUT replacement using randperm
        num_to_sample = min(num_neg_samples, len(candidate_neg_tails))
        candidate_tensor = torch.tensor(candidate_neg_tails, dtype=torch.long, device=device)
        
        # Generate permutation on CPU (where generator lives), then move to device
        perm = torch.randperm(len(candidate_neg_tails), generator=generator)
        sampled_neg_tails = candidate_tensor[perm[:num_to_sample].to(device)]
        
        # Append negatives
        neg_heads_list.extend([h] * num_to_sample)
        neg_relations_list.extend([r] * num_to_sample)
        neg_tails_list.extend(sampled_neg_tails.tolist())
    
    # Convert lists to tensors
    neg_heads = torch.tensor(neg_heads_list, dtype=torch.long, device=device)
    neg_relations = torch.tensor(neg_relations_list, dtype=torch.long, device=device)
    neg_tails = torch.tensor(neg_tails_list, dtype=torch.long, device=device)
    
    return neg_heads, neg_relations, neg_tails


def negative_sampling_per_relation_fast(heads, relations, tails, num_neg_samples,
                                        filter_map, relation_to_valid_tails, generator):
    """
    Per-relation negative sampling WITH replacement (WikiArt-v2) —
    grouped-vectorized.

    Distribution-faithful rejection sampling against the filtered truth
    set: negatives are uniform over the relation's candidate pool minus
    the (h, r) truths; hub rows whose pool is fully covered by known
    truths contribute no negative (whole-triple drop; the caller aligns
    the positive side via the returned keep mask), preserving the 4-tuple
    contract (neg_heads, neg_rels, neg_tails, keep_rows) — kept in order,
    neg tensors pair 1:1 with pos_scores[keep_rows].

    The RNG stream differs from the earlier per-row loop (one pre-batched
    randint draw per relation group covers the 16-attempt budget, instead
    of lazy per-attempt draws): the uniform-over-eligible distribution is
    unchanged, and the June bit stream was already retired together with
    the hub-drop fix. Inputs may be CPU tensors (train loop passes
    .cpu() copies; a CUDA input costs one sync in tolist).
    """
    batch_size = heads.size(0)
    device = heads.device

    heads_list = heads.tolist()
    rels_list = relations.tolist()

    rows_by_rel = defaultdict(list)
    for i, r in enumerate(rels_list):
        rows_by_rel[r].append(i)

    neg_heads_l, neg_rels_l, neg_tails_l = [], [], []
    drop_rows = []

    for r, rows in rows_by_rel.items():
        pool = relation_to_valid_tails.get(int(r))
        if pool is None:
            # A relation present in a training batch must have a candidate
            # pool. Substituting the positive tail as a "negative" would
            # corrupt the loss signal, so fail loudly instead.
            raise RuntimeError(
                "No candidate pool registered for training relation "
                f"{r}; regenerate the candidate pools"
            )
        n_pool = len(pool)
        # One C-level randint per group pre-draws the full attempt budget:
        # per-row rejection then walks plain python ints (no per-attempt
        # tensor/`.item()` crossings).
        draws = torch.randint(
            n_pool, (len(rows), num_neg_samples, 16),
            generator=generator).tolist()
        for j, i in enumerate(rows):
            h = heads_list[i]
            truths = filter_map.get_tails((h, int(r)), set())
            row_negs = []
            exhausted = False
            for k in range(num_neg_samples):
                neg_t = None
                for a in range(16):
                    cand = pool[draws[j][k][a]]
                    if cand not in truths:
                        neg_t = cand
                        break
                if neg_t is None:
                    # Rare fallback: masked view of the eligible tails
                    if truths:
                        avail = [t for t in pool if t not in truths]
                    else:
                        avail = pool
                    if avail:
                        pick = torch.randint(
                            len(avail), (1,), generator=generator)
                        neg_t = int(pick.item())
                        neg_t = avail[neg_t]
                    else:
                        # Hub exhaustion: every pool tail is a known truth.
                        # Protocol-safe drop for this triple (whole row, so
                        # num_neg_samples>1 pairing stays coherent).
                        exhausted = True
                        break
                row_negs.append(neg_t)

            if exhausted:
                drop_rows.append(i)
                continue
            for neg_t in row_negs:
                neg_heads_l.append(h)
                neg_rels_l.append(int(r))
                neg_tails_l.append(neg_t)

    keep = torch.ones(batch_size, dtype=torch.bool, device=device)
    if drop_rows:
        keep[torch.tensor(drop_rows, dtype=torch.long, device=device)] = False

    neg_heads = torch.tensor(neg_heads_l, dtype=torch.long, device=device)
    neg_rels = torch.tensor(neg_rels_l, dtype=torch.long, device=device)
    neg_tails = torch.tensor(neg_tails_l, dtype=torch.long, device=device)
    return neg_heads, neg_rels, neg_tails, keep


# ==================== Evaluation ====================

def evaluate_kge(model, dataloader, filter_map, ks=[1, 3, 10], 
                bidirectional=True, relation_to_valid_tails=None,
                device=torch.device('cpu')):
    """
    Unified evaluation function for both WN9-IMG and WikiArt.
    
    Args:
        model: The KGE model
        dataloader: DataLoader for evaluation data
        filter_map: Dict for filtered ranking
        ks: List of k values for Hits@k
        bidirectional: If True, evaluate both (h,r,?) and (?,r,t). If False, only (h,r,?)
        relation_to_valid_tails: If provided, use per-relation candidates (WikiArt style)
        device: Device for computation
    
    Returns:
        overall_mrr, overall_hits_at_k, relation_metrics

    Dispatch: the tail-only WikiArt protocol (bidirectional=False with
    candidate pools) runs through _evaluate_kge_pooled — a batched,
    protocol-identical path. Everything else keeps the per-triple
    reference loop (_evaluate_kge_per_triple).
    """
    if relation_to_valid_tails is not None and not bidirectional:
        return _evaluate_kge_pooled(model, dataloader, filter_map, ks,
                                    relation_to_valid_tails, device)
    return _evaluate_kge_per_triple(model, dataloader, filter_map, ks,
                                    bidirectional, relation_to_valid_tails,
                                    device)


def _evaluate_kge_per_triple(model, dataloader, filter_map, ks,
                             bidirectional, relation_to_valid_tails, device):
    """
    Reference (per-triple) evaluation loop for both WN9-IMG and WikiArt.

    Args:
        model: The KGE model
        dataloader: DataLoader for evaluation data
        filter_map: Dict for filtered ranking
        ks: List of k values for Hits@k
        bidirectional: If True, evaluate both (h,r,?) and (?,r,t). If False, only (h,r,?)
        relation_to_valid_tails: If provided, use per-relation candidates (WikiArt style)
        device: Device for computation
    
    Returns:
        overall_mrr, overall_hits_at_k, relation_metrics
    """
    model.eval()
    
    with torch.no_grad():
        ranks_head = [] if bidirectional else None
        ranks_tail = []
        
        hits_at_ks_head = {k: 0 for k in ks} if bidirectional else None
        hits_at_ks_tail = {k: 0 for k in ks}
        
        relation_ranks_head = defaultdict(list) if bidirectional else None
        relation_ranks_tail = defaultdict(list)
        
        relation_hits_at_ks = {k: defaultdict(int) for k in ks}
        relation_counts = defaultdict(int)
        
        for batch in dataloader:
            heads = batch["head_id"].to(device)
            relations = batch["relation_id"].to(device)
            tails = batch["tail_id"].to(device)
            
            for i in range(len(heads)):
                head = heads[i]
                relation = relations[i]
                tail = tails[i]
                
                # ========== Tail Prediction: (h, r, ?) ==========
                if relation_to_valid_tails is not None:
                    # WikiArt: per-relation candidates
                    valid_tails_for_relation = set(relation_to_valid_tails.get(relation.item(), []))
                    correct_tails = filter_map.get((head.item(), relation.item()), set())
                    filtered_tails = (valid_tails_for_relation - correct_tails) | {tail.item()}
                    all_tail_entities = torch.tensor(sorted(filtered_tails), device=device, dtype=torch.long)
                else:
                    # WN9-IMG: all entities
                    all_tail_entities = torch.arange(model.num_entities, device=device)
                
                all_scores_tail = model(
                    head=head.unsqueeze(0).expand(len(all_tail_entities)),
                    relation=relation.unsqueeze(0).expand(len(all_tail_entities)),
                    tail=all_tail_entities
                )
                
                # Filtered ranking for WN9-IMG
                if relation_to_valid_tails is None:
                    valid_tails = filter_map.get((head.item(), relation.item()), set())
                    for valid_tail in valid_tails:
                        if valid_tail != tail.item():
                            all_scores_tail[valid_tail] = float('-inf')
                
                sorted_indices_tail = torch.argsort(all_scores_tail, descending=True, stable=True)
                
                if relation_to_valid_tails is not None:
                    # WikiArt: find correct tail in candidate list
                    correct_tail_idx = (all_tail_entities == tail).nonzero(as_tuple=False)[0].item()
                    rank_tail = (sorted_indices_tail == correct_tail_idx).nonzero(as_tuple=False)[0].item() + 1
                else:
                    # WN9-IMG: direct lookup
                    rank_tail = (sorted_indices_tail == tail).nonzero(as_tuple=False).item() + 1
                
                ranks_tail.append(rank_tail)
                relation_ranks_tail[relation.item()].append(rank_tail)
                relation_counts[relation.item()] += 1
                
                for k in ks:
                    if rank_tail <= k:
                        hits_at_ks_tail[k] += 1
                        relation_hits_at_ks[k][relation.item()] += 1
                
                # ========== Head Prediction: (?, r, t) ==========
                if bidirectional:
                    all_head_entities = torch.arange(model.num_entities, device=device)
                    
                    all_scores_head = model(
                        head=all_head_entities,
                        relation=relation.unsqueeze(0).expand(model.num_entities),
                        tail=tail.unsqueeze(0).expand(model.num_entities)
                    )
                    
                    # Filtered ranking
                    valid_heads = filter_map.get_heads(
                        (tail.item(), relation.item()), set())
                    for valid_head in valid_heads:
                        if valid_head != head.item():
                            all_scores_head[valid_head] = float('-inf')
                    
                    sorted_indices_head = torch.argsort(all_scores_head, descending=True, stable=True)
                    rank_head = (sorted_indices_head == head).nonzero(as_tuple=False).item() + 1
                    
                    ranks_head.append(rank_head)
                    relation_ranks_head[relation.item()].append(rank_head)
                    
                    for k in ks:
                        if rank_head <= k:
                            hits_at_ks_head[k] += 1
                            relation_hits_at_ks[k][relation.item()] += 1
        
        # Compute overall metrics
        if bidirectional:
            total_predictions = len(ranks_tail) + len(ranks_head)
            overall_mrr = (sum(1.0 / r for r in ranks_tail) + sum(1.0 / r for r in ranks_head)) / total_predictions
            overall_hits_at_k = {k: (hits_at_ks_tail[k] + hits_at_ks_head[k]) / total_predictions for k in ks}
        else:
            overall_mrr = sum(1.0 / r for r in ranks_tail) / len(ranks_tail)
            overall_hits_at_k = {k: hits_at_ks_tail[k] / len(ranks_tail) for k in ks}
        
        # Per-relation metrics
        relation_metrics = {}
        for relation, count in relation_counts.items():
            if count == 0:
                continue
            
            if bidirectional:
                ranks = relation_ranks_tail[relation] + relation_ranks_head[relation]
            else:
                ranks = relation_ranks_tail[relation]
            
            mrr = sum(1.0 / r for r in ranks) / len(ranks)
            # Uncomment if you want to compute MR
            # mean_rank = sum(ranks) / len(ranks)
            
            relation_metrics[relation] = {
                'MRR': mrr,
                # Uncomment if you want to report MR
                # 'Mean Rank': mean_rank,
                **{f'Hits@{k}': relation_hits_at_ks[k][relation] / len(ranks) for k in ks}
            }
    
    return overall_mrr, overall_hits_at_k, relation_metrics


def _evaluate_kge_pooled(model, dataloader, filter_map, ks,
                         relation_to_valid_tails, device):
    """
    Batched evaluation for the WikiArt-style tail-only protocol.

    Semantics contract (identical to _evaluate_kge_per_triple with
    bidirectional=False and candidate pools): every triple (h, r, t) is
    ranked against sorted(pool[r] - truths(h, r) | {t}), scored by the
    model per candidate. Grouping triples per relation and scoring the
    shared pre-sorted pool in chunked forwards preserves that candidate
    index space exactly: removing known truths and re-adding the true
    tail keeps the sorted order, a stable descending argsort breaks score
    ties by pool position in both constructions, and -inf masked truths
    sort strictly last in both (they are absent in the reference list).

    Model forwards are flattened to 1D (the fusion/inductive paths read
    batch_size from entity_ids.shape[0] and assign via 1D boolean masks),
    matching the reference call shape row-for-row.
    """
    model.eval()

    # GPU-memory budget: the fused multimodal forward materializes ~a dozen
    # (rows x dim) fp32 tensors per call (structural + visual/textual zeros
    # + masked assigns + fusion result + seen masks), so the binding limit
    # is forward ROWS per call, not the score-matrix element count. Keep
    # rows <= ~128K => ~6-8 GB transient on A100-40 next to the model,
    # optimizer states and cached blocks. First GPU exercise proved the
    # score-based bound (16M elements => up to ~1.9M rows/forward) OOMs.
    scores_budget = 16_000_000
    rows_budget = 131_072
    triple_cap = 256
    cand_chunk = 8192

    heads_l, rels_l, tails_l = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            heads_l.append(batch["head_id"])
            rels_l.append(batch["relation_id"])
            tails_l.append(batch["tail_id"])
        if not heads_l:
            raise RuntimeError("evaluate_kge received an empty eval loader")
        heads_all = torch.cat(heads_l)
        rels_all = torch.cat(rels_l)
        tails_all = torch.cat(tails_l)

        ranks_tail = []
        hits_at_ks_tail = {k: 0 for k in ks}

        relation_ranks_tail = defaultdict(list)
        relation_hits_at_ks = {k: defaultdict(int) for k in ks}
        relation_counts = defaultdict(int)

        for rel_id in rels_all.unique().tolist():
            r = int(rel_id)
            rows = (rels_all == r).nonzero(as_tuple=False).flatten()
            pool = relation_to_valid_tails.get(r)
            if not pool:
                # Degenerate candidate universe {t}: per-triple rank 1.
                n = rows.numel()
                ranks_tail.extend([1] * n)
                relation_ranks_tail[r].extend([1] * n)
                relation_counts[r] += n
                for k in ks:
                    hits_at_ks_tail[k] += n
                    relation_hits_at_ks[k][r] += n
                continue

            pool_cpu = torch.tensor(pool, dtype=torch.long)
            pool_dev = pool_cpu.to(device)
            m = pool_cpu.numel()
            pool_np = pool_cpu.numpy()
            chunk = max(1, min(triple_cap, scores_budget // m,
                               rows_budget // cand_chunk))

            for t0 in range(0, rows.numel(), chunk):
                idx = rows[t0:t0 + chunk]
                h_cpu = heads_all[idx]
                t_cpu = tails_all[idx]
                b = idx.numel()
                rel_col = torch.full((b,), r, dtype=torch.long,
                                     device=device)

                # Known-truth mask + true-tail positions (CPU bookkeeping;
                # no CUDA sync — searchsorted runs on a CPU pool copy).
                mask = torch.zeros(b, m, dtype=torch.bool)
                pos_t = torch.full((b,), -1, dtype=torch.long)
                for j in range(b):
                    hi = int(h_cpu[j].item())
                    ti = int(t_cpu[j].item())
                    truths = filter_map.get_tails((hi, r), set())
                    if truths:
                        t_np = np.fromiter(sorted(truths), dtype=np.int64)
                        pos = np.searchsorted(pool_np, t_np)
                        clipped = np.minimum(pos, m - 1)
                        in_pool = pool_np[clipped] == t_np
                        if in_pool.any():
                            mask[j, torch.from_numpy(
                                pos[in_pool]).long()] = True
                    pt = int(np.searchsorted(pool_np, ti))
                    if pt < m and pool_np[pt] == ti:
                        # The true tail re-enters the candidate universe.
                        mask[j, pt] = False
                        pos_t[j] = pt
                    else:
                        raise RuntimeError(
                            "Eval tail absent from its relation candidate "
                            f"pool (relation {r}, tail {ti}); pools are "
                            "supersets of loader facts — regenerate the "
                            "candidate pools"
                        )

                h_dev = h_cpu.to(device)
                scores = torch.empty(b, m, device=device)
                for c0 in range(0, m, cand_chunk):
                    c1 = min(c0 + cand_chunk, m)
                    w = c1 - c0
                    hf = h_dev.unsqueeze(1).expand(b, w).reshape(-1)
                    rf = rel_col.unsqueeze(1).expand(b, w).reshape(-1)
                    tf = pool_dev[c0:c1].unsqueeze(0).expand(b, w).reshape(-1)
                    scores[:, c0:c1] = model(hf, rf, tf).view(b, w)
                scores = scores.masked_fill(mask.to(device), float("-inf"))

                order = torch.argsort(scores, dim=1, descending=True,
                                      stable=True)
                rank = (order == pos_t.to(device).unsqueeze(1)).int()
                rank = rank.argmax(dim=1) + 1
                rk = rank.tolist()
                for j in range(b):
                    rj = rk[j]
                    ranks_tail.append(rj)
                    relation_ranks_tail[r].append(rj)
                    relation_counts[r] += 1
                    for k in ks:
                        if rj <= k:
                            hits_at_ks_tail[k] += 1
                            relation_hits_at_ks[k][r] += 1

        overall_mrr = sum(1.0 / r for r in ranks_tail) / len(ranks_tail)
        overall_hits_at_k = {k: hits_at_ks_tail[k] / len(ranks_tail)
                             for k in ks}

        relation_metrics = {}
        for relation, count in relation_counts.items():
            if count == 0:
                continue
            ranks = relation_ranks_tail[relation]
            mrr = sum(1.0 / r for r in ranks) / len(ranks)
            relation_metrics[relation] = {
                'MRR': mrr,
                **{f'Hits@{k}': relation_hits_at_ks[k][relation]
                   / len(ranks) for k in ks}
            }

    return overall_mrr, overall_hits_at_k, relation_metrics


# ==================== Prediction Functions ====================

def predict_tails(model, entity_id, relation_ids, entity_to_id, relation_to_id,
                 filter_map=None, relation_to_valid_tails=None, k=5, 
                 device=torch.device('cpu')):
    """
    Predict top-k tails for a given head entity and list of relations.
    Works for both WN9-IMG and WikiArt.
    
    Args:
        model: Trained KGE model
        entity_id: Head entity ID (integer)
        relation_ids: List of relation IDs (integers)
        entity_to_id: Entity name to ID mapping (for reverse lookup)
        relation_to_id: Relation name to ID mapping (for reverse lookup)
        filter_map: Optional filter map for excluding known triples
        relation_to_valid_tails: Optional per-relation candidate pools (WikiArt)
        k: Number of top predictions to return
        device: Computation device
    
    Returns:
        Dictionary mapping relation_name -> list of (tail_entity_name, score) tuples
    """
    model.eval()
    
    # Reverse mappings
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    predictions = {}
    
    with torch.no_grad():
        for rel_id in relation_ids:
            # Get candidate tails
            if relation_to_valid_tails is not None:
                # WikiArt: per-relation candidates
                candidate_tails = torch.tensor(
                    relation_to_valid_tails.get(rel_id, []),
                    device=device, dtype=torch.long
                )
            else:
                # WN9-IMG: all entities
                candidate_tails = torch.arange(model.num_entities, device=device)
            
            if len(candidate_tails) == 0:
                predictions[id_to_relation.get(rel_id, f"relation_{rel_id}")] = []
                continue
            
            # Compute scores for all candidates
            head = torch.tensor([entity_id], device=device).expand(len(candidate_tails))
            relation = torch.tensor([rel_id], device=device).expand(len(candidate_tails))
            
            scores = model(head, relation, candidate_tails)
            
            # Filter out known triples if filter_map provided
            if filter_map is not None:
                known_tails = filter_map.get((entity_id, rel_id), set())
                for known_tail in known_tails:
                    if relation_to_valid_tails is not None:
                        # WikiArt: find position in candidate list
                        mask = (candidate_tails == known_tail)
                        if mask.any():
                            scores[mask] = float('-inf')
                    else:
                        # WN9-IMG: direct indexing
                        scores[known_tail] = float('-inf')
            
            # Get top-k
            top_k_values, top_k_indices = torch.topk(scores, min(k, len(scores)))
            
            # Convert to entity names
            top_predictions = []
            for idx, score in zip(top_k_indices, top_k_values):
                if relation_to_valid_tails is not None:
                    tail_id = candidate_tails[idx].item()
                else:
                    tail_id = idx.item()
                tail_name = id_to_entity.get(tail_id, f"entity_{tail_id}")
                top_predictions.append((tail_name, score.item()))
            
            rel_name = id_to_relation.get(rel_id, f"relation_{rel_id}")
            predictions[rel_name] = top_predictions
    
    return predictions


def predict_heads(model, entity_id, relation_ids, entity_to_id, relation_to_id,
                 filter_map=None, k=5, device=torch.device('cpu')):
    """
    Predict top-k heads for a given tail entity and list of relations.
    Used for WN9-IMG bidirectional evaluation.
    
    Args:
        model: Trained KGE model
        entity_id: Tail entity ID (integer)
        relation_ids: List of relation IDs (integers)
        entity_to_id: Entity name to ID mapping (for reverse lookup)
        relation_to_id: Relation name to ID mapping (for reverse lookup)
        filter_map: Optional filter map for excluding known triples
        k: Number of top predictions to return
        device: Computation device
    
    Returns:
        Dictionary mapping relation_name -> list of (head_entity_name, score) tuples
    """
    model.eval()
    
    # Reverse mappings
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    predictions = {}
    
    with torch.no_grad():
        for rel_id in relation_ids:
            # All entities as candidates (head prediction always uses all entities)
            candidate_heads = torch.arange(model.num_entities, device=device)
            
            # Compute scores for all candidates
            relation = torch.tensor([rel_id], device=device).expand(model.num_entities)
            tail = torch.tensor([entity_id], device=device).expand(model.num_entities)
            
            scores = model(candidate_heads, relation, tail)
            
            # Filter out known triples if filter_map provided
            if filter_map is not None:
                known_heads = filter_map.get_heads((entity_id, rel_id), set())
                for known_head in known_heads:
                    scores[known_head] = float('-inf')
            
            # Get top-k
            top_k_values, top_k_indices = torch.topk(scores, min(k, len(scores)))
            
            # Convert to entity names
            top_predictions = []
            for idx, score in zip(top_k_indices, top_k_values):
                head_id = idx.item()
                head_name = id_to_entity.get(head_id, f"entity_{head_id}")
                top_predictions.append((head_name, score.item()))
            
            rel_name = id_to_relation.get(rel_id, f"relation_{rel_id}")
            predictions[rel_name] = top_predictions
    
    return predictions


def make_predictions(model, targets, entity_to_id, relation_to_id,
                    filter_map=None, relation_to_valid_tails=None,
                    device=torch.device('cpu')):
    """
    Make predictions for a list of target entities and relations.
    Supports both tail prediction and head prediction.
    
    Args:
        model: Trained KGE model
        targets: List of dicts with keys:
                 - 'entity': entity name
                 - 'relations': list of relation names
                 - 'k': number of predictions (default: 5)
                 - 'direction': 'tail' or 'head' (default: 'tail')
        entity_to_id: Entity name to ID mapping
        relation_to_id: Relation name to ID mapping
        filter_map: Optional filter map
        relation_to_valid_tails: Optional per-relation candidates (WikiArt)
        device: Computation device
    
    Returns:
        List of prediction results
    """
    results = []
    
    for target in targets:
        entity_name = target['entity']
        relation_names = target['relations']
        k = target.get('k', 5)
        direction = target.get('direction', 'tail')
        
        if entity_name not in entity_to_id:
            print(f"Warning: Entity '{entity_name}' not found in dataset")
            continue
        
        entity_id = entity_to_id[entity_name]
        relation_ids = [relation_to_id[r] for r in relation_names if r in relation_to_id]
        
        if not relation_ids:
            print(f"Warning: No valid relations found for entity '{entity_name}'")
            continue
        
        if direction == 'tail':
            predictions = predict_tails(
                model, entity_id, relation_ids,
                entity_to_id, relation_to_id,
                filter_map=filter_map,
                relation_to_valid_tails=relation_to_valid_tails,
                k=k,
                device=device
            )
        elif direction == 'head':
            predictions = predict_heads(
                model, entity_id, relation_ids,
                entity_to_id, relation_to_id,
                filter_map=filter_map,
                k=k,
                device=device
            )
        else:
            print(f"Warning: Invalid direction '{direction}'. Use 'tail' or 'head'")
            continue
        
        results.append({
            'entity': entity_name,
            'direction': direction,
            'predictions': predictions
        })
        
        print_predictions(predictions, 
                         title=f"Top-{k} {direction} predictions for '{entity_name}'")
    
    return results


def print_predictions(predictions, title="Predictions"):
    """Pretty print predictions."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    
    for relation, preds in predictions.items():
        print(f"\n{relation}:")
        if not preds:
            print("  (no predictions)")
        else:
            for i, (entity, score) in enumerate(preds, 1):
                print(f"  {i}. {entity:50s} (score: {score:.4f})")
    
    print(f"{'='*80}\n")


# ==================== Model Initialization ====================

def get_model(model_name, num_entities, num_relations,
              visual_features=None, textual_features=None, relation_features=None,
              visual_entity_to_index=None, textual_entity_to_index=None,
              embedding_dim=768, fusion_mode="average",
              use_structural=True, use_visual=False, use_textual=False,
              freeze_visual=True, freeze_textual=True,
              visual_proj=False, textual_proj=False,
              shared_projection=None,
              inductive=False,
              modality_asymmetry=False,
              normalize_before_fusion=False,
              # Model-specific arguments
              p_norm=1,
              normalize_relations=True,
              margin=12.0,
              lr=0.1, use_scheduler=False, device=torch.device('cpu')):
    """
    Initialize KGE model and optimizer.
    """
    model_class = getattr(models, model_name)
    
    # Common arguments for VLKge models
    model_kwargs = {
        'num_entities': num_entities,
        'num_relations': num_relations,
        'embedding_dim': embedding_dim,
        'visual_features': visual_features,
        'textual_features': textual_features,
        'relation_features': relation_features,
        'visual_entity_to_index': visual_entity_to_index,
        'textual_entity_to_index': textual_entity_to_index,
        'fusion_mode': fusion_mode,
        'use_structural': use_structural,
        'use_visual': use_visual,
        'use_textual': use_textual,
        'freeze_visual': freeze_visual,
        'freeze_textual': freeze_textual,
        'visual_proj': visual_proj,
        'textual_proj': textual_proj,
        'shared_projection': shared_projection,
        'inductive': inductive,
        'modality_asymmetry': modality_asymmetry,
        'normalize_before_fusion': normalize_before_fusion,
        'device': device
    }
    
    # Add model-specific arguments
    if model_name == 'TransE':
        model_kwargs['p_norm'] = p_norm
        model_kwargs['normalize_relations'] = normalize_relations
        model_kwargs['raw_margin'] = margin
    elif model_name == 'RotatE':
        model_kwargs['raw_margin'] = margin
    # DistMult and ComplEx don't need extra args
    
    model = model_class(**model_kwargs)
    model = model.to(device)
    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    
    return model, optimizer, scheduler


# ==================== Training ====================

def train(model, train_loader, val_loader, test_loader, sampled_train_loader,
          optimizer, scheduler, filter_map, relation_probs,
          relation_to_valid_tails_train=None, relation_to_valid_tails_val=None,
          relation_to_valid_tails_test=None, 
          relation_to_id=None,
          num_epochs=200, num_neg_samples=1,
          use_bernoulli=False, 
          # WikiArt-v2 specific parameters
          train_data=None,
          entity_to_id=None,
          downsample_relation=None,
          downsample_fraction=1.0,
          batch_size=512,
          top_k=[1, 3, 10], dataset_name='wn9_img', 
          bidirectional_eval=True, device=torch.device('cpu'), save_path=None, 
          generator=None, evaluate_every=1, patience=None,
          resume_from=None, log_metrics=False, checkpoint_metadata=None):
    """
    Unified training function for all datasets.
    
    Args:
        ... (existing args) ...
        resume_from: Path to checkpoint to resume training from (optional)
    """
    
    # ========== Resume Training if Checkpoint Provided ==========
    start_epoch = 0
    best_val_mrr = 0
    early_stop_counter = 0
    
    if resume_from is not None:
        print(f"\nResuming training from checkpoint: {resume_from}")
        model, optimizer, scheduler, start_epoch, best_val_mrr, additional_info = \
            utils.load_checkpoint(model, resume_from, optimizer, scheduler, device)
        
        # Restore early stopping counter if saved
        early_stop_counter = additional_info.get('early_stop_counter', 0)
        
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation MRR so far: {best_val_mrr:.4f}")
        if patience is not None: 
            print(f"Early stopping counter: {early_stop_counter}/{patience}\n")
    
    # Determine if using per-relation sampling
    use_per_relation_sampling = relation_to_valid_tails_train is not None
    
    # Determine if using relation downsampling (WikiArt-v2)
    use_relation_downsampling = (downsample_relation is not None and 
                                  train_data is not None and 
                                  entity_to_id is not None)
    
    # Create reverse mapping for relation names
    id_to_relation = None
    if relation_to_id is not None:
        id_to_relation = {v: k for k, v in relation_to_id.items()}
    
    print(f"\n{'='*80}")
    print(f"Training Configuration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Epochs: {num_epochs} (starting from {start_epoch})")
    print(f"  Negative samples: {num_neg_samples}")
    print(f"  Bidirectional eval: {bidirectional_eval}")
    print(f"  Per-relation sampling: {use_per_relation_sampling}")
    if use_relation_downsampling:
        print(f"  Relation downsampling: {downsample_relation} (keep {downsample_fraction*100:.2f}%)")
    print(f"{'='*80}\n")
    
    if log_metrics and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(start_epoch, num_epochs):
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch + 1}/{num_epochs}")
        
        # ========== Prepare Training Data for This Epoch ==========
        if use_relation_downsampling:
            related = train_data[train_data['relation'] == downsample_relation]
            not_related = train_data[train_data['relation'] != downsample_relation]
            
            epoch_seed = generator.initial_seed() + epoch
            sampled_related = (related.sample(frac=downsample_fraction, 
                                             random_state=epoch_seed)
                              if len(related) > 0 else related)
            
            epoch_train_df = pd.concat([not_related, sampled_related], ignore_index=True)
            
            print(f"  {downsample_relation}: kept {len(sampled_related)}/{len(related)} triples "
                  f"({len(sampled_related)/len(related)*100:.2f}%)")
            print(f"  Total training triples this epoch: {len(epoch_train_df)}")
            
            epoch_train_dataset = KGDataset(epoch_train_df, entity_to_id, relation_to_id)
            epoch_train_loader = DataLoader(epoch_train_dataset, batch_size=batch_size, 
                                           shuffle=True, generator=generator)
            current_loader = epoch_train_loader
        else:
            current_loader = train_loader
        
        total_batches = len(current_loader)
        heartbeat = max(1, total_batches // 10)
        
        # ========== Training ==========
        model.train()
        total_loss = 0
        n_loss_batches = 0

        for i, batch in enumerate(current_loader):
            heads = batch["head_id"].to(device)
            relations = batch["relation_id"].to(device)
            tails = batch["tail_id"].to(device)
            
            pos_scores = model(heads, relations, tails)

            # Negative sampling. The per-triple samplers read scalars via
            # .item(); on CUDA tensors every .item() is a device sync —
            # 512 syncs per batch dominated the epoch wall. One .cpu()
            # transfer per batch keeps the sampler's semantics and RNG
            # stream identical while removing the per-triple syncs. GPU
            # tensors are still used for the pos/neg forward passes.
            heads_s = heads.detach().cpu()
            relations_s = relations.detach().cpu()
            tails_s = tails.detach().cpu()
            if use_per_relation_sampling:
                if dataset_name == 'wikiart_mkg_v1':
                    neg_heads, neg_relations, neg_tails = negative_sampling_per_relation(
                        heads_s, relations_s, tails_s, num_neg_samples,
                        filter_map, relation_to_valid_tails_train, generator
                    )
                    keep_rows = None
                else:
                    (neg_heads, neg_relations, neg_tails,
                     keep_rows) = negative_sampling_per_relation_fast(
                        heads_s, relations_s, tails_s, num_neg_samples,
                        filter_map, relation_to_valid_tails_train, generator
                    )
            else:
                neg_heads, neg_relations, neg_tails = negative_sampling_uniform(
                    heads_s, relations_s, tails_s, model.num_entities, num_neg_samples,
                    filter_map, relation_probs, generator, use_bernoulli=use_bernoulli
                )
                keep_rows = None

            neg_heads = neg_heads.to(heads.device)
            neg_relations = neg_relations.to(heads.device)
            neg_tails = neg_tails.to(heads.device)
            if keep_rows is not None:
                keep_rows = keep_rows.to(heads.device)

            if keep_rows is not None:
                n_kept = int(keep_rows.sum().item())
                if n_kept == 0:
                    # Entire batch is hub rows of an exhausted relation:
                    # nothing to train on this batch.
                    continue
                if n_kept < heads.shape[0]:
                    # Hub rows dropped by the sampler: kept negatives pair
                    # 1:1 (in order) with the keep_rows positives.
                    pos_kept = pos_scores[keep_rows]
                    neg_scores = model(neg_heads, neg_relations, neg_tails)
                    neg_scores = neg_scores.view(n_kept, num_neg_samples)
                    loss = compute_logistic_loss(pos_kept, neg_scores)
                else:
                    neg_scores = model(neg_heads, neg_relations, neg_tails)
                    neg_scores = neg_scores.view(pos_scores.shape[0],
                                                 num_neg_samples)
                    loss = compute_logistic_loss(pos_scores, neg_scores)
            else:
                neg_scores = model(neg_heads, neg_relations, neg_tails)
                neg_scores = neg_scores.view(pos_scores.shape[0], num_neg_samples)
                loss = compute_logistic_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_loss_batches += 1

            # Progress heartbeat: large epochs are otherwise fully silent.
            if i % heartbeat == 0:
                hb_avg = (total_loss / n_loss_batches
                          if n_loss_batches else 0.0)
                print(f"  Epoch {epoch + 1}/{num_epochs} "
                      f"{i * 100 // total_batches:3d}% ({i}/{total_batches}) "
                      f"loss={hb_avg:.4f}", flush=True)

        avg_loss = total_loss / n_loss_batches if n_loss_batches else 0.0
        
        # ========== Evaluation ==========
        if (epoch + 1) % evaluate_every == 0:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating...")
            
            mrr_train, hits_train, rel_metrics_train = evaluate_kge(
                model, sampled_train_loader, filter_map, ks=top_k,
                bidirectional=bidirectional_eval,
                relation_to_valid_tails=relation_to_valid_tails_train,
                device=device
            )
            
            mrr_val, hits_val, rel_metrics_val = evaluate_kge(
                model, val_loader, filter_map, ks=top_k,
                bidirectional=bidirectional_eval,
                relation_to_valid_tails=relation_to_valid_tails_val,
                device=device
            )
            
            if scheduler is not None:
                scheduler.step(mrr_val)
            
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Loss: {avg_loss:.4f} | Train MRR: {mrr_train:.4f} " +
                  " ".join([f"H@{k}: {hits_train[k]:.4f}" for k in top_k]))
            
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Val MRR: {mrr_val:.4f} " +
                  " ".join([f"H@{k}: {hits_val[k]:.4f}" for k in top_k]))
            
            if log_metrics and device.type == "cuda":
                gpu_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                gpu_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                print(f"  GPU memory: {gpu_alloc:.0f} MB allocated / {gpu_reserved:.0f} MB reserved")
                torch.cuda.reset_peak_memory_stats(device)
            
            # Early stopping with enhanced checkpoint saving
            if mrr_val > best_val_mrr:
                best_val_mrr = mrr_val
                early_stop_counter = 0
                
                if save_path:
                    import pamr.utils as utils
                    # Save complete checkpoint with training state
                    additional_info = {
                        'best_val_mrr': best_val_mrr,
                        'early_stop_counter': early_stop_counter,
                        'dataset_name': dataset_name,
                        'model_name': model.__class__.__name__,
                        'num_neg_samples': num_neg_samples,
                        'batch_size': batch_size,
                    }
                    if checkpoint_metadata:
                        additional_info.update(checkpoint_metadata)
                    utils.save_checkpoint(model, optimizer, scheduler, epoch + 1, 
                                        mrr_val, save_path, additional_info)
                    print(f"  → Saved best model to {save_path}")
            else:
                early_stop_counter += 1
            
            if patience and early_stop_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        else:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loss: {avg_loss:.4f}")
    
    # ========== Final Test Evaluation ==========
    print(f"\n{'='*80}")
    print(f"Training completed. Best validation MRR: {best_val_mrr:.4f}")
    
    if save_path:
        print(f"Loading best model from {save_path}...")
        if not os.path.exists(save_path):
            print(f"Warning: Save path {save_path} does not exist. Skipping test evaluation.")
            print(f"{'='*80}\n")
            return
        
        model.load_state_dict(torch.load(save_path, map_location=device)['model_state_dict'])
        
        print(f"Evaluating on test set...")
        mrr_test, hits_test, rel_metrics_test = evaluate_kge(
            model, test_loader, filter_map, ks=top_k,
            bidirectional=bidirectional_eval,
            relation_to_valid_tails=relation_to_valid_tails_test,
            device=device
        )
        
        print(f"\n{'='*80}")
        print(f"FINAL TEST RESULTS:")
        print(f"  MRR: {mrr_test:.4f}")
        for k in top_k:
            print(f"  Hits@{k}: {hits_test[k]:.4f}")
        print(f"{'='*80}")
        
        print(f"\nPer-Relation Test Metrics:")
        print(f"{'-'*80}")
        for rel_id, metrics in sorted(rel_metrics_test.items()):
            if id_to_relation is not None:
                rel_name = id_to_relation.get(rel_id, f"Relation_{rel_id}")
            else:
                rel_name = f"Relation_{rel_id}"
            
            metrics_str = " | ".join([f"{key}: {val:.4f}" for key, val in metrics.items()])
            print(f"{rel_name:30s}: {metrics_str}")
        print(f"{'-'*80}\n")
    
    print(f"{'='*80}\n")
