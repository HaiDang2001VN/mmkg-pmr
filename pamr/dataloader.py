import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from pamr.relation_registry import RelationRegistry


class DirectionalFilterMap:
    """Separate filtered tails and heads while preserving tail-map access."""

    def __init__(self, tails, heads):
        self.tails = tails
        self.heads = heads

    def get(self, key, default=None):
        """Backward-compatible tail lookup used by tail corruption."""
        return self.tails.get(key, default)

    def get_tails(self, key, default=None):
        return self.tails.get(key, default)

    def get_heads(self, key, default=None):
        return self.heads.get(key, default)


class KGDataset(Dataset):
    """
    PyTorch Dataset for Knowledge Graph triples.
    Works with both WN9-IMG and WikiArt datasets.
    """
    def __init__(self, data, entity_to_id, relation_to_id):
        self.data = data.copy()
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

        # Map entities and relations to IDs. DataLoader supplies numeric IDs
        # for materialized internal inverse relations, which have no public
        # relation name.
        self.data.loc[:, 'head_id'] = self.data['head'].map(self.entity_to_id)
        self.data.loc[:, 'tail_id'] = self.data['tail'].map(self.entity_to_id)
        if 'relation_id' not in self.data.columns:
            self.data.loc[:, 'relation_id'] = self.data['relation'].map(
                self.relation_to_id)

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
         inverse_relation_pairs: Dict mapping existing relation names to their
                               existing inverse relation names
         symmetric_relations: Explicitly undirected relation names
        
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
                  inverse_relation_pairs=None,
                  symmetric_relations=None,
                 use_per_relation_candidates=False,
                 artist2artist_relations=None,
                 bidirectional_eval=False):
        
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.bidirectional_eval = bidirectional_eval
        self.use_per_relation_candidates = use_per_relation_candidates
        
        # Load data and build the relation registry before split filtering.
        # Relation IDs must not depend on which split is currently selected.
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.raw_data):,} triples from {data_path}")
        if add_inverse_relations:
            raise ValueError(
                "add_inverse_relations is deprecated. Use inverse_relation_pairs "
                "and symmetric_relations with the canonical registry."
            )
        self.relation_registry = RelationRegistry(
            self.raw_data['relation'].dropna().unique(),
            inverse_pairs=inverse_relation_pairs,
            symmetric_relations=symmetric_relations,
        )
        self.relation_to_id = dict(self.relation_registry.relation_to_id)
        self.relations = list(self.relation_registry.raw_relation_names)

        # Use a deterministic entity mapping over the complete raw vocabulary.
        # Split filtering must not renumber entities.
        raw_entities = pd.unique(
            self.raw_data[['head', 'tail']].values.ravel('K'))
        self.entities = sorted(str(entity) for entity in raw_entities)
        self.entity_to_id = {entity: i for i, entity in enumerate(self.entities)}

        # Consume the raw frame instead of copying it: exclusion filters and
        # inverse materialization below own this data from here on, and
        # keeping a pristine 8M-row copy alongside the materialized frame
        # costs gigabytes of resident memory.
        self.data = self.raw_data
        self.raw_data = None
        self.data['relation_id'] = self.data['relation'].map(
            self.relation_to_id)
        self.data['is_inverse'] = False
        
        
            
        # Apply filtering
        if exclude_relations:
            self._exclude_relations(exclude_relations)
        
        if exclude_relations_eval:
            self._exclude_relations_from_eval(exclude_relations_eval)
        
        if filter_entity_prefix:
            self._filter_by_entity_prefix(filter_entity_prefix)
            
        # Add inverse relations
        # Materialize every inverse in the same split as its source fact.
        self._materialize_inverse_relations()
        
        print(f"Dataset: {self.dataset_name}")
        print(f"  Entities: {len(self.entities):,}")
        print(f"  Relations: {len(self.relations):,}")
        
        # Pre-map IDs for faster access. The frame is solely owned here (the
        # raw copy was released above), so no defensive copy is needed.
        self.data['rid'] = self.data['relation_id'].astype(int)
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
    
    def _materialize_inverse_relations(self):
        """Add reciprocal facts without inventing public relation names."""
        if len(self.data) == 0:
            return
        inverse_id_col = self.data['relation_id'].map(
            self.relation_registry.inverse_id)
        inverse_df = pd.DataFrame({
            'head': self.data['tail'],
            # Internal inverse IDs have no public name (None), matching the
            # previous row-wise construction.
            'relation': inverse_id_col.map(
                self.relation_registry.id_to_relation),
            'tail': self.data['head'],
            'mode': self.data['mode'],
            'relation_id': inverse_id_col,
            'is_inverse': True,
        })
        self.data = pd.concat([self.data, inverse_df], ignore_index=True)
        self.data = self.data.drop_duplicates(
            subset=['head', 'relation_id', 'tail', 'mode'],
            ignore_index=True,
        )
    
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

    def get_relation_registry(self):
        return self.relation_registry
    
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
        cols = ['head', 'relation', 'tail', 'mode', 'relation_id', 'is_inverse']
        return train[cols], val[cols], test[cols]
    
    def compute_filter_map(self):
        """ 
        Filtered protocol map on ALL splits (fast vectorized). 
        Produces (h,r)->tails and (t,r)->heads. 
        """
        tail_filter = defaultdict(set)
        head_filter = defaultdict(set)
        # (h,r) -> tails
        g1 = self.data.groupby(['hid', 'rid'])['tid'].unique()
        for (hid, rid), arr in g1.items():
            tail_filter[(int(hid), int(rid))].update(
                int(x) for x in arr.tolist())
        # (t,r) -> heads
        g2 = self.data.groupby(['tid', 'rid'])['hid'].unique()
        for (tid, rid), arr in g2.items():
            head_filter[(int(tid), int(rid))].update(
                int(x) for x in arr.tolist())
        return DirectionalFilterMap(tail_filter, head_filter)

    @staticmethod
    def _filter_maps_from_frame(frame):
        """Directional truth maps from a pre-mapped hid/rid/tid frame."""
        tail_filter = defaultdict(set)
        head_filter = defaultdict(set)
        g1 = frame.groupby(['hid', 'rid'])['tid'].unique()
        for (hid, rid), arr in g1.items():
            tail_filter[(int(hid), int(rid))].update(
                int(x) for x in arr.tolist())
        g2 = frame.groupby(['tid', 'rid'])['hid'].unique()
        for (tid, rid), arr in g2.items():
            head_filter[(int(tid), int(rid))].update(
                int(x) for x in arr.tolist())
        return DirectionalFilterMap(tail_filter, head_filter)

    def compute_filter_maps_by_scope(self, relation_ids=None):
        """
        Split-aware filtered protocol maps.

        Returns a dict of split_name -> DirectionalFilterMap where:
          train filters truths known in train,
          val   filters truths known in train + val,
          test  filters truths known in train + val + test.
        Inverse facts are materialized inside their source split, so every
        truth universe stays split-consistent including reciprocal rows.

        relation_ids: optional iterable of relation IDs. When given, only
        those relations enter the truth maps. Consumers look up (h, r) only
        for the relations their queries actually use (e.g. the train_eval
        target relations), so restricting the subset shrinks the maps by
        orders of magnitude without changing any lookup for prepared
        queries. Unlisted relations return the default empty set, exactly
        like a relation with no known tails.
        """
        scopes = {
            'train': ['train'],
            'val': ['train', 'val'],
            'test': ['train', 'val', 'test'],
        }
        base = self.data[['hid', 'rid', 'tid', 'mode']]
        if relation_ids is not None:
            allowed = {int(rid) for rid in relation_ids}
            base = base[base['rid'].isin(allowed)]
        out = {}
        for split_name, allowed_modes in scopes.items():
            scope_frame = base[base['mode'].isin(allowed_modes)]
            out[split_name] = self._filter_maps_from_frame(scope_frame)
            # Each scope frame is a transient copy of the base frame; free
            # it before the next scope so only the (small) maps accumulate.
            del scope_frame
        return out
        
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
        return self.relation_registry.num_relations
    
    def get_split_sizes(self):
        """Get number of triples in each split."""
        return {
            'train': len(self.data[self.data['mode'] == 'train']),
            'val': len(self.data[self.data['mode'] == 'val']),
            'test': len(self.data[self.data['mode'] == 'test'])
        }
