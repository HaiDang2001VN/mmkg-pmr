"""Shared ranking evaluator for VL-KGE and PAMR.

Single source of truth for strict / non-strict / length-first composite
ranking semantics:

- strict:   rank = #scores strictly greater than the best positive + 1
- non-strict: rank = #scores greater or equal (the selected positive is
  already counted in the >= sum; no extra +1 is applied)
- length-first composite: candidates with shorter hops rank first, ties
  broken by score; only meaningful when hop metadata is supplied
- multi-positive queries use the best-scoring positive
- optional candidate-tail aggregation: probes are grouped by
  (query_id, candidate_tail_id) with a max reduction before ranking, so the
  ranking unit is the candidate tail rather than the individual probe
"""

from collections import defaultdict

import numpy as np


def compute_rank_metrics(all_scores, all_labels, all_query_ids, top_k,
                         exclude_relations=None, split=None, all_hops=None,
                         expected_queries=None, all_cand_ids=None,
                         rel_for_qid=None, all_has_path=None,
                         lf_no_path_policy="exclude"):
    """Single-pass MRR/Hits@K (strict + non-strict + length-first composite).

    When all_cand_ids is provided, probes are aggregated per
    (query_id, candidate_tail_id) with a max reduction before ranking, so the
    ranking unit is the candidate tail rather than the individual path.

    all_has_path: optional per-probe bool array. Candidate aggregation then
    derives each candidate's path availability and hop count from its path
    probes only (no-path probes never contribute hops).

    lf_no_path_policy controls the length-first composite when no-path
    candidates exist:
      - "exclude": rank only candidates with a real path; queries whose
        positives are all no-path are skipped from lf metrics and counted.
      - "infinite": no-path candidates carry infinite hops, so all
        path candidates rank before unreachable ones and ties among
        unreachable candidates break by score.
    When no hop metadata is supplied, lf metrics are omitted entirely.
    """
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    qids_raw = np.array(all_query_ids)
    hops_raw = np.array(all_hops) if all_hops is not None else None
    has_path_raw = (np.array(all_has_path, dtype=bool)
                    if all_has_path is not None else None)

    if all_cand_ids is not None:
        cands_raw = np.array(all_cand_ids)
        from collections import defaultdict as _dd
        groups = _dd(list)
        for i in range(len(scores)):
            groups[(int(qids_raw[i]), int(cands_raw[i]))].append(i)
        aggregated = []
        for (qid, _tail), idxs in sorted(groups.items()):
            gather = np.array(idxs, dtype=np.int64)
            if hops_raw is not None:
                if has_path_raw is not None:
                    path_idx = gather[has_path_raw[gather]]
                    hops_val = (int(np.min(hops_raw[path_idx]))
                                if len(path_idx) else 0)
                    has_path_val = bool(has_path_raw[gather].max())
                else:
                    hops_val = int(np.min(hops_raw[gather]))
                    has_path_val = True
            else:
                hops_val = 0
                has_path_val = bool(has_path_raw[gather].max()
                                    if has_path_raw is not None else 1)
            aggregated.append({
                "query_id": qid,
                "score": float(np.max(scores[gather])),
                "label": int(np.max(labels[gather])),
                "hops": hops_val,
                "has_path": has_path_val,
            })
        scores = np.array([row["score"] for row in aggregated])
        labels = np.array([row["label"] for row in aggregated], dtype=float)
        qids_raw = np.array([row["query_id"] for row in aggregated])
        if hops_raw is not None:
            hops_raw = np.array([row["hops"] for row in aggregated])
        if has_path_raw is not None:
            has_path_raw = np.array([row["has_path"] for row in aggregated],
                                    dtype=bool)

    candidate_queries = int(len(np.unique(qids_raw))) if len(qids_raw) else 0

    if len(scores) == 0:
        return {
            "MRR": 0.0, "Hits": {k: 0.0 for k in top_k},
            "Accuracy": 0.0, "num_queries": 0,
            "candidate_queries": candidate_queries,
            "expected_queries": expected_queries,
            "coverage": (0.0 / expected_queries
                          if expected_queries else None),
            "skipped_no_positive": 0,
            "skipped_excluded": 0,
        }

    # Sort by query_id to process in blocks
    order = np.argsort(qids_raw, kind="stable")
    s = scores[order]
    l = labels[order]
    q = qids_raw[order]
    h = hops_raw[order] if hops_raw is not None else None
    has_path_arr = (has_path_raw[order]
                    if has_path_raw is not None else None)

    exclude_set = set(exclude_relations or [])

    ranks, mrrs = [], []
    ranks_ns, mrrs_ns = [], []     # non-strict: equal scores count against
    ranks_lf, mrrs_lf = [], []     # length-first strict
    ranks_lfns, mrrs_lfns = [], [] # length-first non-strict
    correct = 0
    correct_ns = 0
    correct_lf = 0
    correct_lfns = 0
    skipped_no_pos = 0
    skipped_excluded = 0
    lf_skipped_no_path_pos = 0

    # Per-relation accumulators
    per_rel_ranks = defaultdict(list)
    per_rel_mrrs = defaultdict(list)
    per_rel_ranks_ns = defaultdict(list)
    per_rel_mrrs_ns = defaultdict(list)
    per_rel_correct = defaultdict(int)
    per_rel_correct_ns = defaultdict(int)
    per_rel_ranks_lf = defaultdict(list)
    per_rel_mrrs_lf = defaultdict(list)
    per_rel_ranks_lfns = defaultdict(list)
    per_rel_mrrs_lfns = defaultdict(list)

    # Per-split relation map (query_ids restart from 0 per split)
    rel_map = None
    if rel_for_qid and split:
        rel_map = rel_for_qid.get(split, {})

    i = 0
    while i < len(q):
        lf_recorded = False
        qid = q[i]
        # Find end of this query block
        j = i + 1
        while j < len(q) and q[j] == qid:
            j += 1

        # Look up relation name (for exclude check + per-rel grouping)
        rel_name = None
        if rel_map is not None:
            rel_name = rel_map.get(qid, "")

        # Check excluded relation
        if exclude_set and rel_name and rel_name in exclude_set:
            skipped_excluded += 1
            i = j
            continue

        block_s = s[i:j]
        block_l = l[i:j]
        pos_idx = np.where(block_l == 1)[0]
        if len(pos_idx) == 0:
            skipped_no_pos += 1
            i = j
            continue

        # Multiple paths can reach the same positive tail. Use the best
        # positive path rather than making other positive paths count as
        # negatives against the first row.
        best_score_pos = pos_idx[np.argmax(block_s[pos_idx])]
        pos_score = block_s[best_score_pos]
        rank = int(np.sum(block_s > pos_score)) + 1
        # The selected positive is already included in >=. The previous +1
        # double-counted it and made Accuracy_ns impossible for one positive.
        rank_ns = int(np.sum(block_s >= pos_score))
        ranks.append(rank)
        mrrs.append(1.0 / rank)
        ranks_ns.append(rank_ns)
        mrrs_ns.append(1.0 / rank_ns)
        if rank == 1:
            correct += 1
        if rank_ns == 1:
            correct_ns += 1

        # Length-first composite ranking (sort by hops, break ties with
        # score). With per-probe path availability the policy decides how
        # no-path candidates enter the length ordering.
        if h is not None:
            block_h = h[i:j].astype(float)
            block_hp = (has_path_arr[i:j]
                        if has_path_arr is not None
                        else np.ones(j - i, dtype=bool))
            if lf_no_path_policy == "infinite":
                eff_h = np.where(block_hp, block_h, np.inf)
                cand_idx = np.arange(j - i)
            else:  # "exclude": only candidates with a real path
                eff_h = block_h
                cand_idx = np.where(block_hp)[0]
            sel_h = eff_h[cand_idx]
            sel_s = block_s[cand_idx]
            sel_l = block_l[cand_idx]
            sel_pos = np.where(sel_l == 1)[0]
            if len(sel_pos) == 0:
                # Every positive of this query is unreachable under the
                # exclude policy; skip it from lf metrics.
                lf_skipped_no_path_pos += 1
            else:
                pos_hops = sel_h[sel_pos].min()
                pos_hop_mask = sel_h[sel_pos] == pos_hops
                lf_pos_score = sel_s[sel_pos[pos_hop_mask]].max()
                # shorter path (smaller hops) + tie-breaking with score
                rank_lf = int(np.sum(
                    (sel_h < pos_hops) |
                    ((sel_h == pos_hops) & (sel_s > lf_pos_score))
                )) + 1
                rank_lfns = int(np.sum(
                    (sel_h < pos_hops) |
                    ((sel_h == pos_hops) & (sel_s >= lf_pos_score))
                ))
                ranks_lf.append(rank_lf)
                mrrs_lf.append(1.0 / rank_lf)
                ranks_lfns.append(rank_lfns)
                mrrs_lfns.append(1.0 / rank_lfns)
                if rank_lf == 1:
                    correct_lf += 1
                if rank_lfns == 1:
                    correct_lfns += 1
                lf_recorded = True

        # Per-relation tracking
        if rel_name:
            per_rel_ranks[rel_name].append(rank)
            per_rel_mrrs[rel_name].append(1.0 / rank)
            per_rel_ranks_ns[rel_name].append(rank_ns)
            per_rel_mrrs_ns[rel_name].append(1.0 / rank_ns)
            if rank == 1:
                per_rel_correct[rel_name] += 1
            if rank_ns == 1:
                per_rel_correct_ns[rel_name] += 1
            # F-9 gate: append per-relation lf entries ONLY when THIS
            # query produced lf ranks — the exclude branch may skip a
            # query whose positives are all no-path; without the gate a
            # stale rank_lf from the previous iteration leaks here.
            if h is not None and lf_recorded:
                per_rel_ranks_lf[rel_name].append(rank_lf)
                per_rel_mrrs_lf[rel_name].append(1.0 / rank_lf)
                per_rel_ranks_lfns[rel_name].append(rank_lfns)
                per_rel_mrrs_lfns[rel_name].append(1.0 / rank_lfns)

        i = j

    n_queries = len(ranks)
    if n_queries == 0:
        return {
            "MRR": 0.0, "Hits": {k: 0.0 for k in top_k},
            "Accuracy": 0.0, "num_queries": 0,
            "candidate_queries": candidate_queries,
            "expected_queries": expected_queries,
            "coverage": (0.0 / expected_queries
                          if expected_queries else None),
            "skipped_no_positive": skipped_no_pos,
            "skipped_excluded": skipped_excluded,
        }

    has_lf = h is not None

    hits = {}
    hits_ns = {}
    hits_lf = {}
    hits_lfns = {}
    for k in top_k:
        hits[k] = float(np.mean(np.array(ranks) <= k))
        hits_ns[k] = float(np.mean(np.array(ranks_ns) <= k))
        if has_lf:
            hits_lf[k] = (float(np.mean(np.array(ranks_lf) <= k))
                          if ranks_lf else 0.0)
            hits_lfns[k] = (float(np.mean(np.array(ranks_lfns) <= k))
                            if ranks_lfns else 0.0)

    result = {
        "MRR": float(np.mean(mrrs)),
        "MRR_ns": float(np.mean(mrrs_ns)),
        "Hits": hits,
        "Hits_ns": hits_ns,
        "Accuracy": correct / n_queries,
        "Accuracy_ns": correct_ns / n_queries,
        "num_queries": n_queries,
        "candidate_queries": candidate_queries,
        "expected_queries": expected_queries,
        "coverage": (n_queries / expected_queries
                      if expected_queries else None),
        "skipped_no_positive": skipped_no_pos,
        "skipped_excluded": skipped_excluded,
    }
    if has_lf:
        result["MRR_lf"] = float(np.mean(mrrs_lf)) if mrrs_lf else 0.0
        result["MRR_lfns"] = float(np.mean(mrrs_lfns)) if mrrs_lfns else 0.0
        result["Hits_lf"] = hits_lf
        result["Hits_lfns"] = hits_lfns
        result["Accuracy_lf"] = (correct_lf / len(ranks_lf)
                                 if ranks_lf else 0.0)
        result["Accuracy_lfns"] = (correct_lfns / len(ranks_lfns)
                                   if ranks_lfns else 0.0)
        result["lf_num_queries"] = len(ranks_lf)
        result["lf_skipped_no_path_positive"] = lf_skipped_no_path_pos
        result["lf_policy"] = lf_no_path_policy

    # Build per-relation summary
    if per_rel_ranks:
        per_rel = {}
        for rel in sorted(per_rel_ranks):
            r = np.array(per_rel_ranks[rel])
            r_ns = np.array(per_rel_ranks_ns[rel])
            entry = {
                "count": int(len(r)),
                "MRR": float(np.mean(per_rel_mrrs[rel])),
                "MRR_ns": float(np.mean(per_rel_mrrs_ns[rel])),
                "Accuracy": per_rel_correct[rel] / len(r),
                "Accuracy_ns": per_rel_correct_ns[rel] / len(r),
                "Hits": {k: float(np.mean(r <= k)) for k in top_k},
                "Hits_ns": {k: float(np.mean(r_ns <= k)) for k in top_k},
            }
            if has_lf:
                r_lf = np.array(per_rel_ranks_lf[rel])
                r_lfns = np.array(per_rel_ranks_lfns[rel])
                entry["MRR_lf"] = float(np.mean(per_rel_mrrs_lf[rel]))
                entry["MRR_lfns"] = float(np.mean(per_rel_mrrs_lfns[rel]))
                entry["Hits_lf"] = {k: float(np.mean(r_lf <= k)) for k in top_k}
                entry["Hits_lfns"] = {k: float(np.mean(r_lfns <= k)) for k in top_k}
            per_rel[rel] = entry
        result["per_relation"] = per_rel

    return result
