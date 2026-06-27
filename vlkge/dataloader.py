import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class KGDataset(Dataset):
    """
    PyTorch Dataset for Knowledge Graph triples.
    Works with both WN9-IMG and WikiArt datasets.
    """
    def __init__(self, data, entity_to_id, relation_to_id):
        self.data = data.copy()
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

        # Map entities and relations to IDs
        self.data.loc[:, 'head_id'] = self.data['head'].map(self.entity_to_id)
        self.data.loc[:, 'tail_id'] = self.data['tail'].map(self.entity_to_id)
        self.data.loc[:, 'relation_id'] = self.data['relation'].map(self.relation_to_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        triple = self.data.iloc[idx]
        head_id, relation_id, tail_id = triple['head_id'], triple['relation_id'], triple['tail_id']
        return {
            "head_id": torch.tensor(head_id, dtype=torch.long),
            "relation_id": torch.tensor(relation_id, dtype=torch.long),
            "tail_id": torch.tensor(tail_id, dtype=torch.long)
        }


class KnowledgeGraphDataLoader:
    """
    Unified DataLoader for WN9-IMG and WikiArt-MKG datasets.
    
    Args:
        data_path: Path to CSV file with columns ['head', 'relation', 'tail', 'mode']
        dataset_name: Name of the dataset ('wn9img', 'wikiart-v1', 'wikiart-v2')
        
        # Filtering options:
        exclude_relations: List of relation names to remove from all splits
        exclude_relations_eval: List of relation names to remove only from val/test splits
        filter_entity_prefix: Keep only entities whose 'head' contains this prefix (e.g., 'images/')
        
        # Data augmentation:
        add_inverse_relations: Dict mapping relation_name -> inverse_relation_name
                              e.g., {'isCreatedByArtist': 'hasCreatedArtwork'}
        
        # Evaluation configuration:
        use_per_relation_candidates: If True, use per-relation candidate pools for evaluation
        artist2artist_relations: List of relation names that should use split-local candidates
                                (only relevant if use_per_relation_candidates=True)
        bidirectional_eval: If True, evaluate both (h,r,?) and (?,r,t). If False, only (h,r,?)
    """
    
    def __init__(self, 
                 data_path,
                 dataset_name,
                 exclude_relations=None,
                 exclude_relations_eval=None,
                 filter_entity_prefix=None,
                 add_inverse_relations=None,
                 use_per_relation_candidates=False,
                 artist2artist_relations=None,
                 bidirectional_eval=False):
        
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.bidirectional_eval = bidirectional_eval
        self.use_per_relation_candidates = use_per_relation_candidates
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.data):,} triples from {data_path}")
        
        
            
        # Apply filtering
        if exclude_relations:
            self._exclude_relations(exclude_relations)
        
        if exclude_relations_eval:
            self._exclude_relations_from_eval(exclude_relations_eval)
        
        if filter_entity_prefix:
            self._filter_by_entity_prefix(filter_entity_prefix)
            
        # Add inverse relations
        if add_inverse_relations:
            self._add_inverse_relations(add_inverse_relations)
        
        # Build entity and relation mappings
        self.entities = pd.unique(self.data[['head', 'tail']].values.ravel('K'))
        self.relations = pd.unique(self.data['relation'])
        self.entity_to_id = {e: i for i, e in enumerate(self.entities)}
        self.relation_to_id = {r: i for i, r in enumerate(self.relations)}
        
        print(f"Dataset: {self.dataset_name}")
        print(f"  Entities: {len(self.entities):,}")
        print(f"  Relations: {len(self.relations):,}")
        
        # Pre-map IDs for faster access
        self.data = self.data.copy()
        self.data['rid'] = self.data['relation'].map(self.relation_to_id)
        self.data['hid'] = self.data['head'].map(self.entity_to_id)
        self.data['tid'] = self.data['tail'].map(self.entity_to_id)
        
        # Build candidate pools for evaluation
        if self.use_per_relation_candidates:
            # Per-relation candidate pools (WikiArt style)
            self.artist2artist_relations = artist2artist_relations or []
            self.artist2artist_ids = {self.relation_to_id[r] 
                                     for r in self.artist2artist_relations 
                                     if r in self.relation_to_id}
            self._build_per_relation_pools()
            print(f"  Using per-relation candidate pools")
            if self.artist2artist_ids:
                print(f"  Artist-to-artist relations: {len(self.artist2artist_ids)}")
        else:
            # All entities as candidates (WN9-IMG style)
            self.relation_to_valid_tails = None
            self.relation_to_valid_tails_eval_train = None
            self.relation_to_valid_tails_eval_val = None
            self.relation_to_valid_tails_eval_test = None
            
    # ==================== Filtering Methods ====================
    
    def _exclude_relations(self, relations):
        """Remove specified relations from all splits."""
        initial_count = len(self.data)
        self.data = self.data[~self.data['relation'].isin(relations)]
        removed = initial_count - len(self.data)
        if removed > 0:
            print(f"Excluded relations {relations}: removed {removed:,} triples")
    
    def _exclude_relations_from_eval(self, relations):
        """Remove specified relations only from val and test splits."""
        initial_count = len(self.data)
        non_train_mask = self.data['mode'] != 'train'
        exclude_mask = self.data['relation'].isin(relations)
        self.data = self.data[~(non_train_mask & exclude_mask)]
        removed = initial_count - len(self.data)
        if removed > 0:
            print(f"Excluded relations {relations} from val/test: removed {removed:,} triples")
    
    def _filter_by_entity_prefix(self, prefix):
        """Keep only triples where head entity contains the specified prefix."""
        initial_count = len(self.data)
        self.data = self.data[self.data['head'].str.contains(prefix, na=False)]
        removed = initial_count - len(self.data)
        if removed > 0:
            print(f"Filtered entities by prefix '{prefix}': removed {removed:,} triples")
    
    def _add_inverse_relations(self, inverse_mapping):
        """
        Add inverse relations for specified relation types.
        
        Args:
            inverse_mapping: Dict mapping original_relation -> inverse_relation
                           e.g., {'isCreatedByArtist': 'hasCreatedArtwork'}
        """
        inverse_triples_list = []
        
        for original_rel, inverse_rel in inverse_mapping.items():
            # Find all triples with the original relation
            original_triples = self.data[(self.data['relation'] == original_rel) & 
                                         (self.data['mode'] == 'train')
                                         ].copy()
            
            if not original_triples.empty:
                # Create inverse triples with swapped head/tail
                inverse_triples = original_triples.copy()
                inverse_triples[['head', 'tail']] = inverse_triples[['tail', 'head']]
                inverse_triples['relation'] = inverse_rel
                inverse_triples_list.append(inverse_triples)
                print(f"Added inverse relation '{inverse_rel}' for '{original_rel}': {len(inverse_triples):,} triples")
        
        if inverse_triples_list:
            self.data = pd.concat([self.data] + inverse_triples_list, ignore_index=True)
    
    # ==================== Candidate Pool Methods ====================
    
    def _build_per_relation_pools(self):
        """Build per-relation candidate pools for evaluation."""
        # Base pool: all splits combined
        self.relation_to_valid_tails = self._build_rel2tails_from_df(self.data)
        
        # Per-split pools
        grp = self.data.groupby(['mode', 'rid'])['tid'].unique()
        pools = {'train': {}, 'val': {}, 'test': {}}
        for (mode, rid), arr in grp.items():
            pools[mode][int(rid)] = [int(x) for x in arr.tolist()]
        
        self.relation_to_valid_tails_train = pools['train']
        self.relation_to_valid_tails_val = pools['val']
        self.relation_to_valid_tails_test = pools['test']
        self.relation_to_valid_tails_all = self.relation_to_valid_tails
        
        # Build evaluation pools with artist2artist policy
        self._build_eval_pools()
    
    def _build_rel2tails_from_df(self, df):
        """Build relation -> valid tails mapping from dataframe."""
        s = df.groupby('rid')['tid'].unique()
        return {int(r): sorted(int(x) for x in arr.tolist()) for r, arr in s.items()}
    
    def _merge_pool_with_policy(self, split_pool, all_pool, artist2artist_ids):
        """
        Merge pools with policy: artist2artist uses split-local pool, 
        others use all-splits pool.
        """
        out = {}
        all_keys = set(split_pool.keys()) | set(all_pool.keys())
        for rid in all_keys:
            if rid in artist2artist_ids:
                out[rid] = split_pool.get(rid, [])
            else:
                out[rid] = all_pool.get(rid, [])
        return out
    
    def _build_eval_pools(self):
        """Build evaluation candidate pools with artist2artist policy."""
        # For TRAINING: artist2artist uses train-only, others use all-splits
        self.relation_to_valid_tails_eval_train = self._merge_pool_with_policy(
            self.relation_to_valid_tails_train,
            self.relation_to_valid_tails_all,
            self.artist2artist_ids
        )
        self.relation_to_valid_tails_eval_val = self._merge_pool_with_policy(
            self.relation_to_valid_tails_val,
            self.relation_to_valid_tails_all,
            self.artist2artist_ids
        )
        self.relation_to_valid_tails_eval_test = self._merge_pool_with_policy(
            self.relation_to_valid_tails_test,
            self.relation_to_valid_tails_all,
            self.artist2artist_ids
        )
    
    # ==================== Public API ====================
    
    def get_entities_and_relations(self):
        """Get entity and relation ID mappings."""
        return self.entity_to_id, self.relation_to_id
    
    def get_valid_tails_per_relation(self):
        """
        Get per-relation valid tail candidates.
        Returns None if using all entities as candidates.
        """
        return self.relation_to_valid_tails if self.use_per_relation_candidates else None
    
    def split_data(self):
        """Split data into train, validation, and test sets."""
        train = self.data[self.data['mode'] == 'train']
        val = self.data[self.data['mode'] == 'val']
        test = self.data[self.data['mode'] == 'test']
        
        # Return original columns for downstream compatibility
        cols = ['head', 'relation', 'tail', 'mode']
        return train[cols], val[cols], test[cols]
    
    def compute_filter_map(self):
        """ 
        Filtered protocol map on ALL splits (fast vectorized). 
        Produces (h,r)->tails and (t,r)->heads. 
        """
        filt = defaultdict(set)
        # (h,r) -> tails
        g1 = self.data.groupby(['hid', 'rid'])['tid'].unique()
        for (hid, rid), arr in g1.items():
            filt[(int(hid), int(rid))].update(int(x) for x in arr.tolist())
        # (t,r) -> heads
        g2 = self.data.groupby(['tid', 'rid'])['hid'].unique()
        for (tid, rid), arr in g2.items():
            filt[(int(tid), int(rid))].update(int(x) for x in arr.tolist())
        return filt
        
    def compute_relation_probs(self):
        """
        Compute p(h|r) for negative sampling.
        
        Returns dict mapping relation_id -> probability of corrupting head vs tail.
        p(h|r) = |unique_heads| / (|unique_heads| + |unique_tails|)
        """
        heads_per_r = self.data.groupby('rid')['hid'].nunique()
        tails_per_r = self.data.groupby('rid')['tid'].nunique()
        
        probs = {}
        for rid in heads_per_r.index.union(tails_per_r.index):
            nh = int(heads_per_r.get(rid, 0))
            nt = int(tails_per_r.get(rid, 0))
            probs[int(rid)] = nh / (nh + nt) if (nh + nt) > 0 else 0.5
        
        return probs
    
    def get_num_entities(self):
        """Get total number of entities."""
        return len(self.entities)
    
    def get_num_relations(self):
        """Get total number of relations."""
        return len(self.relations)
    
    def get_split_sizes(self):
        """Get number of triples in each split."""
        return {
            'train': len(self.data[self.data['mode'] == 'train']),
            'val': len(self.data[self.data['mode'] == 'val']),
            'test': len(self.data[self.data['mode'] == 'test'])
        }