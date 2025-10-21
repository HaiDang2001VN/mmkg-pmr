"""
Utility functions for VL-KGE experiments
"""

import numpy as np
import os
import pickle
import random
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path


# ---------- Existing ----------
def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}

def save_yaml_config(config, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

# ---------- New: path utilities ----------
def expand_path(p):
    """Expand ~ and resolve to absolute; passthrough None."""
    return None if p is None else str(Path(p).expanduser().resolve())

def validate_path(p, description):
    """Validate that a path exists, raising a friendly error if not."""
    if p is None:
        return None
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return str(path)

def resolve_config_paths(config: dict, base_dir: Path, repo_root: Path = None):
    """
    Resolve any '*path' or known path-like keys inside the YAML.

    Resolution order for *relative* paths (first that exists wins):
      1) relative to the YAML file directory (base_dir)
      2) relative to repo_root (if provided)
      3) relative to current working directory

    Supports ${REPO_ROOT} and ~ expansion. If no candidate exists, leaves the
    original string unchanged so later validation can produce a helpful error.
    """
    def _is_path_key(k: str) -> bool:
        k_low = k.lower()
        return ('path' in k_low) or (k in ('data_path', 'save_path', 'resume_from'))

    def _resolve_one(p):
        if p is None:
            return None

        # expand env vars and ~
        s = os.path.expandvars(os.path.expanduser(str(p)))

        # ${REPO_ROOT} placeholder
        if repo_root is not None:
            s = s.replace('${REPO_ROOT}', str(repo_root))

        cand = Path(s)
        if cand.is_absolute():
            return str(cand)

        # Try candidates in order
        candidates = []
        if base_dir is not None:
            candidates.append((base_dir / cand))
        if repo_root is not None:
            candidates.append((Path(repo_root) / cand))
        candidates.append((Path.cwd() / cand))

        for c in candidates:
            if c.exists():
                return str(c.resolve())

        # Nothing matched; leave as-is so validate_path can error meaningfully
        return s

    # Top-level keys
    for k, v in list(config.items()):
        if isinstance(v, str) and _is_path_key(k):
            config[k] = _resolve_one(v)

    # Nested dicts
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in list(values.items()):
                if isinstance(v, str) and _is_path_key(k):
                    values[k] = _resolve_one(v)
    return config

def apply_config_overrides(args, config: dict):
    known = set(vars(args).keys())
    # only apply keys NOT owned by argparse
    for k, v in config.items():
        if isinstance(v, dict) or k in known:
            continue
        setattr(args, k, v)
    return args


# ==================== Reproducibility ====================

def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


# ==================== Device Management ====================

def set_gpu(device_index=0):
    """
    Set the GPU device to use for training.
    
    Args:
        device_index (int): Index of the GPU to use
        
    Returns:
        torch.device: Device object pointing to the specified GPU or CPU
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
        print(f"Using GPU: {torch.cuda.get_device_name(device_index)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    return device


# ==================== Feature Loading ====================

def _infer_dim(d):
    for v in d.values():
        e = process_embeddings(v)
        if e is not None:
            return e.shape[0]
    return None

def process_embeddings(embeddings):
    if isinstance(embeddings, dict):
        if embeddings:
            return torch.mean(
                torch.stack([torch.tensor(emb, dtype=torch.float32) 
                           for emb in embeddings.values()]), 
                dim=0
            )
    elif isinstance(embeddings, (list, tuple)):
        return torch.mean(
            torch.stack([torch.tensor(emb, dtype=torch.float32) 
                       for emb in embeddings]), 
            dim=0
        )
    else:
        return torch.tensor(embeddings, dtype=torch.float32)
    return None
        
def load_features(args, entity_to_id, relation_to_id):
    """
    Load and align visual/textual/relation features with entity/relation IDs.
    Works uniformly for both dense (WN9-IMG) and sparse (WikiArt-MKG) formats.
    """
    visual_features = None
    textual_features = None
    relation_features = None
    visual_entity_to_index = None
    textual_entity_to_index = None
    
    # ==================== Load Visual Features ====================
    if args.use_visual:
        print(f"Loading visual features from {args.visual_features_path}...")
        with open(args.visual_features_path, 'rb') as f:
            visual_dict = pickle.load(f)
        
        embedding_dim = _infer_dim(visual_dict)
        if embedding_dim is None:
            raise ValueError("No valid embeddings found to infer dimension.")

        # Align features with entity IDs
        visual_dict_aligned = {}
        for entity, embeddings in visual_dict.items():
            if entity not in entity_to_id:
                continue
            entity_id = entity_to_id[entity]
            emb = process_embeddings(embeddings)
            if emb is not None:
                visual_dict_aligned[entity_id] = emb
            else:
                visual_dict_aligned[entity_id] = torch.zeros(embedding_dim, dtype=torch.float32)
        
        if not visual_dict_aligned:
            raise ValueError("No valid visual features found for any entity")
        
        # Stack into tensor and create index mapping
        visual_features = torch.stack(list(visual_dict_aligned.values()))
        visual_entity_to_index = {
            entity_id: idx for idx, entity_id in enumerate(visual_dict_aligned.keys())
        }
        
        # Normalize if requested
        if args.normalize_visual:
            visual_features = F.normalize(visual_features, p=2, dim=-1)
            print("  Normalized visual features")
        
        print(f"  Loaded visual features: {len(visual_features)} entities, dim={visual_features.shape[1]}")
    
    # ==================== Load Textual Features ====================
    if args.use_textual:
        print(f"Loading textual features from {args.textual_features_path}...")
        with open(args.textual_features_path, 'rb') as f:
            textual_dict = pickle.load(f)
        
        embedding_dim = _infer_dim(textual_dict)
        if embedding_dim is None:
            raise ValueError("No valid embeddings found to infer dimension.")
        
        # Align features with entity IDs
        textual_dict_aligned = {}
        for entity, embeddings in textual_dict.items():
            if entity not in entity_to_id:
                continue
            entity_id = entity_to_id[entity]
            emb = process_embeddings(embeddings)
            if emb is not None:
                textual_dict_aligned[entity_id] = emb
            else:
                textual_dict_aligned[entity_id] = torch.zeros(embedding_dim, dtype=torch.float32)
        
        if not textual_dict_aligned:
            raise ValueError("No valid textual features found for any entity")
        
        # Stack into tensor and create index mapping
        textual_features = torch.stack(list(textual_dict_aligned.values()))
        textual_entity_to_index = {
            entity_id: idx for idx, entity_id in enumerate(textual_dict_aligned.keys())
        }
        
        # Normalize if requested
        if args.normalize_textual:
            textual_features = F.normalize(textual_features, p=2, dim=-1)
            print("  Normalized textual features")
        
        print(f"  Loaded textual features: {len(textual_features)} entities, dim={textual_features.shape[1]}")
    
    # ==================== Load Relation Features ====================
    if args.use_relation_features:
        print(f"Loading relation features from {args.relation_features_path}...")
        with open(args.relation_features_path, 'rb') as f:
            relation_dict = pickle.load(f)
        
        # Align features with relation IDs
        relation_dict_aligned = {}
        for relation, embeddings in relation_dict.items():
            if relation not in relation_to_id:
                continue
            relation_id = relation_to_id[relation]
            emb = process_embeddings(embeddings)
            if emb is not None:
                relation_dict_aligned[relation_id] = emb
        
        if not relation_dict_aligned:
            raise ValueError("No valid relation features found")
        
        # Relations always use all IDs
        num_relations = len(relation_to_id)
        embedding_dim = list(relation_dict_aligned.values())[0].shape[0]
        relation_features = torch.zeros((num_relations, embedding_dim), dtype=torch.float32)
        
        for relation_id, feature in relation_dict_aligned.items():
            relation_features[relation_id] = feature
        
        # Normalize if requested
        if args.normalize_relation:
            relation_features = F.normalize(relation_features, p=2, dim=-1)
            print("  Normalized relation features")
        
        print(f"  Loaded relation features: {num_relations} relations, dim={embedding_dim}")
    
    return visual_features, textual_features, relation_features, visual_entity_to_index, textual_entity_to_index

# ==================== Model Checkpoint Management ====================

def save_checkpoint(model, optimizer, scheduler, epoch, score, save_path, 
                   additional_info=None):
    """
    Save complete model checkpoint for resumable training.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (can be None)
        epoch: Current epoch
        score: Current validation score (MRR)
        save_path: Path to save checkpoint
        additional_info: Dict with additional info to save (e.g., config, best_score)
    """
    checkpoint = {
        'epoch': epoch,
        'score': score,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add any additional information (config, hyperparameters, etc.)
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, load_path, optimizer=None, scheduler=None, device='cpu'):
    """
    Load model checkpoint with optional optimizer and scheduler.
    
    Args:
        model: PyTorch model
        load_path: Path to checkpoint
        optimizer: Optimizer (optional, for resuming training)
        scheduler: Learning rate scheduler (optional, for resuming training)
        device: Device to load on
        
    Returns:
        tuple: (model, optimizer, scheduler, epoch, score, additional_info)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")
    
    print(f"Loading checkpoint from {load_path}...")
    checkpoint = torch.load(load_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get epoch and score
    epoch = checkpoint.get('epoch', 0)
    score = checkpoint.get('score', 0.0)
    
    # Get any additional info (excluding standard keys)
    standard_keys = {'epoch', 'score', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'}
    additional_info = {k: v for k, v in checkpoint.items() if k not in standard_keys}
    
    print(f"Loaded checkpoint from epoch {epoch} with score {score:.4f}")
    
    return model, optimizer, scheduler, epoch, score, additional_info


# ==================== Cosine Warmup Scheduler (Optional) ====================

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine learning rate scheduler with warmup.
    
    Optional scheduler that can be used instead of ReduceLROnPlateau.
    """
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor