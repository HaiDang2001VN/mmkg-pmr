"""
Linear Probe Evaluation for WikiArt-MKG datasets.

Supports four image-based tasks:
1. Style classification
2. Artist attribution
3. Tag prediction (multi-label)
4. Year prediction (regression)

Usage:
    # All tasks
    python3 -m vlkge.scripts.evaluate_linear_probe \
        --dataset wikiart_mkg_v1 \
        --data_path vlkge/data/wikiart_mkg_v1/wikiart_mkg_v1_triples.csv \
        --visual_features_path vlkge/data/wikiart_mkg_v1/features/wikiart_mkg_v1_vf_clip.pkl \
        --tasks all

    # Specific tasks
    python3 -m vlkge.scripts.evaluate_linear_probe \
        --dataset wikiart_mkg_v2 \
        --data_path vlkge/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
        --visual_features_path vlkge/data/wikiart_mkg_v2/features/wikiart_mkg_v2_vf_clip.pkl \
        --tasks style artist year
"""

import os
import argparse
from pathlib import Path
from copy import deepcopy
import re

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

import vlkge.utils as utils


# ============================================================================
# Dataset-specific configurations
# ============================================================================

DATASET_CONFIGS = {
    'wikiart_mkg_v1': {
        'artist_relation': 'isCreatedByArtist',
        'period_relation': 'belongsToTimeframe',
        'style_relation': 'hasStyle',
        'tag_relation': 'isAssociatedWithTag',
        'year_relation': None,  # Not available in v1
        'exclude_from_data': [],
    },
    
    'wikiart_mkg_v2': {
        'artist_relation': 'isCreatedByArtist',
        'style_relation': 'hasStyle',
        'period_relation': None,  # Not available in v2
        'tag_relation': 'isAssociatedWithTag',
        'year_relation': 'isCreatedInYear',
        'exclude_from_data': ['hasCreatedArtwork', 'isRelatedToArtwork', 'isRelatedToArtist',
                             'isInfluencedBy', 'isInfluencedOn', 'isPupilOf', 'isTeacherOf'],
    }
}

TASK_NAMES = {
    'artist': 'Artist Attribution',
    'period': 'Period Classification',
    'style': 'Style Classification',
    'tag': 'Tag Prediction',
    'year': 'Year Prediction'
}


SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent  # repo root containing vlkge/

parser = argparse.ArgumentParser(description='Linear probe evaluation for WikiArt-MKG datasets')
parser.add_argument('--dataset', type=str, required=True,
                    choices=['wikiart_mkg_v1', 'wikiart_mkg_v2'],
                    help='Dataset name')

# Require explicit paths (no auto-detect) for reproducibility
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to triples CSV file')
parser.add_argument('--visual_features_path', type=str, required=True,
                    help='Path to visual features pickle file (same backbone as features)')

# Task selection
parser.add_argument('--tasks', nargs='+', default=['all'],
                    choices=['all', 'artist', 'period', 'style', 'tag', 'year'],
                    help='Tasks to evaluate (default: all available tasks)')

# Training parameters
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--log_interval', type=int, default=10, help='Logging interval (epochs)')

# Regression-specific
parser.add_argument('--year_target_scaling', type=str, default='minmax',
                    choices=['none', 'minmax', 'zscore'],
                    help='Target scaling for year regression')

parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')

args = parser.parse_args()

# ============================================================================
# Verify paths and load data
# ============================================================================

# Expand + validate (friendly errors; supports ~)
args.data_path = utils.validate_path(utils.expand_path(args.data_path), "Triples CSV")
args.visual_features_path = utils.validate_path(utils.expand_path(args.visual_features_path), "Visual features")

print(f"Data path: {args.data_path}")
print(f"Visual features path: {args.visual_features_path}")

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
utils.set_seed(args.seed)

print("\n" + "="*80)
print(f"Linear Probe Evaluation: {args.dataset.upper()}")
print("="*80)
print(f"Device: {device}")
print(f"Seed: {args.seed}")
print(f"Tasks: {', '.join(args.tasks)}")
print("="*80 + "\n")

# Load dataset configuration
dataset_config = DATASET_CONFIGS[args.dataset]

# Determine which tasks to run
if 'all' in args.tasks:
    tasks_to_run = []
    if dataset_config['artist_relation']: tasks_to_run.append('artist')
    if dataset_config['period_relation']: tasks_to_run.append('period')
    if dataset_config['style_relation']: tasks_to_run.append('style')
    if dataset_config['tag_relation']: tasks_to_run.append('tag')
    if dataset_config['year_relation']: tasks_to_run.append('year')
else:
    tasks_to_run = args.tasks

print(f"Will evaluate: {', '.join([TASK_NAMES[t] for t in tasks_to_run])}\n")

# Load data
print("Loading data...")
data = pd.read_csv(args.data_path)

# Exclude relations if specified
if dataset_config.get('exclude_from_data'):
    exclude_rels = dataset_config['exclude_from_data']
    print(f"Excluding relations: {exclude_rels}")
    data = data[~data['relation'].isin(exclude_rels)]

print(f"Total triples: {len(data)}")
print(f"Unique relations: {data['relation'].nunique()}")

# Load visual embeddings
print(f"\nLoading visual embeddings...")
with open(args.visual_features_path, 'rb') as f:
    visual_embeddings = pickle.load(f)
print(f"Loaded {len(visual_embeddings)} visual embeddings")


# ============================================================================
# Helper functions for single-label classification
# ============================================================================

def prepare_classification_data(task_data, label_encoder=None, relation_name="", split_name=""):
    """Prepare X, y for a single-label classification task."""
    td = task_data.copy()
    
    # Map embeddings
    td['head_embedding'] = td['head'].map(visual_embeddings)
    td = td[td['head_embedding'].notnull()]
    
    if td.empty:
        raise RuntimeError(f"[{relation_name}][{split_name}] No rows with embeddings.")
    
    tails = td['tail'].values
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(tails)
    else:
        y = label_encoder.transform(tails)
    
    X = np.vstack(td['head_embedding'].values).astype(np.float32)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    
    return X, y, label_encoder


def compute_accuracy(outputs, labels):
    """Compute top-1 accuracy."""
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        return float((preds == labels).float().mean().item())


def train_classification_task(task_name, relation, train_data, val_data, test_data):
    """Train and evaluate a single-label classification task."""
    print("\n" + "="*80)
    print(f"{TASK_NAMES[task_name].upper()}")
    print("="*80)
    
    task_train = train_data[train_data['relation'] == relation]
    task_val = val_data[val_data['relation'] == relation]
    task_test = test_data[test_data['relation'] == relation]
    
    if task_train.empty:
        print(f"No training data for {relation}")
        return
    
    train_X, train_y, le = prepare_classification_data(task_train, relation_name=relation, split_name="train")
    val_X, val_y, _ = prepare_classification_data(task_val, label_encoder=le, relation_name=relation, split_name="val")
    test_X, test_y, _ = prepare_classification_data(task_test, label_encoder=le, relation_name=relation, split_name="test")
    
    print(f"Relation: {relation}")
    print(f"Classes: {len(le.classes_)}")
    print(f"Train: {train_X.shape[0]}, Val: {val_X.shape[0]}, Test: {test_X.shape[0]}")
    
    dataset_train = TensorDataset(train_X, train_y)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    
    # Linear probe model
    class LinearProbe(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    model = LinearProbe(train_X.shape[1], len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_acc = -float('inf')
    best_state = None
    best_epoch = -1
    
    print(f"\nTraining...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for bx, by in loader_train:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            tr_logits = model(train_X)
            va_logits = model(val_X)
            
            tr_acc = compute_accuracy(tr_logits, train_y)
            va_acc = compute_accuracy(va_logits, val_y)
        
        # Checkpoint selection
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        
        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(loader_train):.4f}")
            print(f"  Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")
    
    # Load best and test
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (Val Acc={best_val_acc:.4f})")
    
    print(f"\nFinal Test Evaluation:")
    model.eval()
    with torch.no_grad():
        te_logits = model(test_X)
        te_acc = compute_accuracy(te_logits, test_y)
    
    print(f"Test Accuracy: {te_acc:.4f}")


# ============================================================================
# Multi-label classification (Tags)
# ============================================================================

def prepare_tag_data(df_split, visual_embeddings, mlb=None, split_name=""):
    """Prepare data for multi-label tag prediction."""
    df_tags = df_split[df_split['relation'] == dataset_config['tag_relation']].copy()
    if df_tags.empty:
        raise RuntimeError(f"[{dataset_config['tag_relation']}][{split_name}] No rows in this split.")
    
    # Drop duplicate (head, tag) edges
    df_tags = df_tags.drop_duplicates(subset=['head', 'tail'])
    
    grouped = df_tags.groupby('head')['tail'].apply(list)
    
    # Keep only heads with embeddings
    heads = [h for h in grouped.index if h in visual_embeddings]
    if not heads:
        raise RuntimeError(f"[{dataset_config['tag_relation']}][{split_name}] No heads with embeddings.")
    
    X = np.vstack([visual_embeddings[h] for h in heads]).astype(np.float32)
    tag_lists = [grouped[h] for h in heads]
    
    if mlb is None:
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(tag_lists).astype(np.float32)
    else:
        Y = mlb.transform(tag_lists).astype(np.float32)
    
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)
    
    return X, Y, mlb, heads


def instance_map_from_scores(Y_true, Y_scores, batch_size=4096):
    """Compute instance-averaged mAP (batched for memory efficiency)."""
    N = Y_true.size(0)
    aps = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            y_t = Y_true[start:end]
            y_s = Y_scores[start:end]
            
            order = torch.argsort(y_s, dim=1, descending=True)
            y_sorted = torch.gather(y_t, 1, order)
            
            pos_mask = (y_sorted == 1.0)
            pos_counts = pos_mask.sum(dim=1)
            
            cumsum_tp = torch.cumsum(y_sorted, dim=1)
            ranks = torch.arange(1, y_sorted.size(1)+1, device=y_t.device).float()
            precisions = cumsum_tp / ranks
            
            prec_sum = (precisions * pos_mask).sum(dim=1)
            batch_ap = torch.where(pos_counts > 0, prec_sum / pos_counts.clamp(min=1),
                                  torch.full_like(prec_sum, float('nan')))
            aps.append(batch_ap.detach().cpu())
    
    aps = torch.cat(aps, dim=0).numpy()
    aps = aps[~np.isnan(aps)]
    return float(aps.mean()) if aps.size > 0 else 0.0


def predict_scores_in_batches(model, X, batch_size=4096):
    """Predict scores in batches."""
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            logits = model(X[i:i+batch_size])
            scores.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    return np.vstack(scores)


def label_macro_map(Y_true_np, Y_scores_np):
    """Compute macro-averaged mAP across labels."""
    valid = (Y_true_np.sum(axis=0) > 0)
    if valid.sum() == 0:
        return 0.0
    return float(average_precision_score(Y_true_np[:, valid], Y_scores_np[:, valid], average='macro'))


def label_micro_map(Y_true_np, Y_scores_np):
    """Compute micro-averaged mAP."""
    if Y_true_np.sum() == 0:
        return 0.0
    return float(average_precision_score(Y_true_np, Y_scores_np, average='micro'))


def train_tag_prediction():
    """Train and evaluate multi-label tag prediction."""
    print("\n" + "="*80)
    print("TAG PREDICTION (MULTI-LABEL)")
    print("="*80)
    
    if dataset_config['tag_relation'] not in data['relation'].unique():
        print(f"No data for {dataset_config['tag_relation']}")
        return
    
    train_X_t, train_Y_t, mlb, train_heads = prepare_tag_data(
        data[data['mode']=='train'], visual_embeddings, mlb=None, split_name="train")
    val_X_t, val_Y_t, _, val_heads = prepare_tag_data(
        data[data['mode']=='val'], visual_embeddings, mlb=mlb, split_name="val")
    test_X_t, test_Y_t, _, test_heads = prepare_tag_data(
        data[data['mode']=='test'], visual_embeddings, mlb=mlb, split_name="test")
    
    print(f"Relation: {dataset_config['tag_relation']}")
    print(f"Number of tags: {train_Y_t.shape[1]}")
    print(f"Train: {train_X_t.shape[0]}, Val: {val_X_t.shape[0]}, Test: {test_X_t.shape[0]}")
    
    # Linear probe model
    class LinearProbe(nn.Module):
        def __init__(self, input_dim, num_tags):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_tags)
        
        def forward(self, x):
            return self.fc(x)
    
    tag_model = LinearProbe(train_X_t.shape[1], train_Y_t.shape[1]).to(device)
    tag_criterion = nn.BCEWithLogitsLoss()
    tag_optimizer = optim.Adam(tag_model.parameters(), lr=args.lr)
    
    tag_train_ds = TensorDataset(train_X_t, train_Y_t)
    tag_loader = DataLoader(tag_train_ds, batch_size=args.batch_size, shuffle=True)
    
    best_val_inst_map = -float('inf')
    best_state = None
    best_epoch = -1
    
    print(f"\nTraining...")
    for epoch in range(args.epochs):
        # Train
        tag_model.train()
        total_loss = 0.0
        for bx, by in tag_loader:
            tag_optimizer.zero_grad()
            logits = tag_model(bx)
            loss = tag_criterion(logits, by)
            loss.backward()
            tag_optimizer.step()
            total_loss += loss.item()
        
        # Validation
        tag_model.eval()
        with torch.no_grad():
            val_logits = tag_model(val_X_t)
            val_scores = torch.sigmoid(val_logits)
            val_inst_map = instance_map_from_scores(val_Y_t, val_scores, batch_size=4096)
        
        if val_inst_map > best_val_inst_map:
            best_val_inst_map = val_inst_map
            best_state = deepcopy(tag_model.state_dict())
            best_epoch = epoch + 1
        
        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(tag_loader):.4f} | Val mAP: {val_inst_map:.4f}")
    
    # Load best and test
    if best_state is not None:
        tag_model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (Val mAP={best_val_inst_map:.4f})")
    
    print("\nFinal Test Evaluation:")
    test_scores_np = predict_scores_in_batches(tag_model, test_X_t, batch_size=4096)
    test_Y_np = test_Y_t.detach().cpu().numpy().astype(np.float32)
    
    te_macro_map = label_macro_map(test_Y_np, test_scores_np)
    te_micro_map = label_micro_map(test_Y_np, test_scores_np)
    
    print(f"Test mAP(macro): {te_macro_map:.4f}")
    print(f"Test mAP(micro): {te_micro_map:.4f}")


# ============================================================================
# Regression (Year prediction)
# ============================================================================

def parse_year_str(s: str):
    """Parse year string (handles BCE/BC, ranges, etc.)."""
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    lower = txt.lower()
    
    # Handle BCE/BC (negative years)
    is_bce = ('bce' in lower) or (re.search(r'\bbc\b', lower) is not None)
    
    # Extract 1-4 digit numbers
    nums = re.findall(r'\d{1,4}', lower)
    if not nums:
        return None
    
    vals = [int(n) for n in nums]
    year = float(sum(vals)) / len(vals)  # Average for ranges
    
    return -year if is_bce else year


def prepare_year_data(df_split, visual_embeddings, split_name=""):
    """Prepare data for year regression."""
    df_year = df_split[df_split['relation'] == dataset_config['year_relation']].copy()
    if df_year.empty:
        raise RuntimeError(f"[{dataset_config['year_relation']}][{split_name}] No rows in this split.")
    
    # Parse years
    df_year['year_num'] = df_year['tail'].apply(parse_year_str)
    df_year = df_year.dropna(subset=['year_num'])
    if df_year.empty:
        raise RuntimeError(f"[{dataset_config['year_relation']}][{split_name}] Could not parse any year strings.")
    
    # Keep only items with embeddings
    df_year = df_year[df_year['head'].isin(visual_embeddings)]
    if df_year.empty:
        raise RuntimeError(f"[{dataset_config['year_relation']}][{split_name}] No heads with embeddings.")
    
    X = np.vstack([visual_embeddings[h] for h in df_year['head'].values]).astype(np.float32)
    y = df_year['year_num'].astype(np.float32).values
    
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    
    return X, y


def regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    y_t = y_true.detach().cpu().numpy()
    y_p = y_pred.detach().cpu().numpy()
    
    mae = float(np.mean(np.abs(y_p - y_t)))
    rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))
    var = np.var(y_t)
    r2 = float(1.0 - np.mean((y_p - y_t) ** 2) / var) if var > 0 else 0.0
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def cumulative_accuracy(y_true, y_pred, tolerances=(5, 10, 25)):
    """Compute accuracy within tolerance thresholds."""
    y_t = y_true.detach().cpu().numpy()
    y_p = y_pred.detach().cpu().numpy()
    
    return {f"Acc@{k}": float(np.mean(np.abs(y_p - y_t) <= k)) for k in tolerances}


def train_year_prediction():
    """Train and evaluate year prediction (regression)."""
    print("\n" + "="*80)
    print("YEAR PREDICTION (REGRESSION)")
    print("="*80)
    
    if dataset_config['year_relation'] not in data['relation'].unique():
        print(f"No data for {dataset_config['year_relation']}")
        return
    
    train_X_y, train_y_raw = prepare_year_data(data[data['mode']=='train'], visual_embeddings, split_name="train")
    val_X_y, val_y_raw = prepare_year_data(data[data['mode']=='val'], visual_embeddings, split_name="val")
    test_X_y, test_y_raw = prepare_year_data(data[data['mode']=='test'], visual_embeddings, split_name="test")
    
    print(f"Relation: {dataset_config['year_relation']}")
    print(f"Train: {train_X_y.shape[0]}, Val: {val_X_y.shape[0]}, Test: {test_X_y.shape[0]}")
    print(f"Year range: [{train_y_raw.min().item():.0f}, {train_y_raw.max().item():.0f}]")
    
    # Target scaling
    mode = args.year_target_scaling
    
    if mode == 'none':
        def fwd_scale(y): return y
        def inv_scale(y): return y
        def clamp01(y): return y
        scale_info = "none"
    
    elif mode == 'zscore':
        mu = train_y_raw.mean().item()
        sigma = train_y_raw.std().item() + 1e-8
        def fwd_scale(y): return (y - mu) / sigma
        def inv_scale(z): return z * sigma + mu
        def clamp01(y): return y
        scale_info = f"zscore (mu={mu:.2f}, sigma={sigma:.2f})"
    
    elif mode == 'minmax':
        y_min = train_y_raw.min().item()
        y_max = train_y_raw.max().item()
        rng = (y_max - y_min) + 1e-8
        def fwd_scale(y): return (y - y_min) / rng
        def inv_scale(u): return u * rng + y_min
        def clamp01(u): return torch.clamp(u, 0.0, 1.0)
        scale_info = f"minmax (min={y_min:.1f}, max={y_max:.1f})"
    else:
        raise ValueError(f"Unknown year_target_scaling: {mode}")
    
    print(f"Target scaling: {scale_info}")
    
    train_y = fwd_scale(train_y_raw)
    val_y = fwd_scale(val_y_raw)
    test_y = fwd_scale(test_y_raw)
    
    # Baseline: predict train median
    train_median = float(torch.median(train_y_raw).item())
    val_baseline_mae = float(np.mean(np.abs(val_y_raw.detach().cpu().numpy() - train_median)))
    print(f"Baseline (predict train median={train_median:.1f}): Val MAE = {val_baseline_mae:.2f}")
    
    # Linear probe model
    class LinearProbe(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return self.fc(x).squeeze(1)
    
    model = LinearProbe(train_X_y.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_ds = TensorDataset(train_X_y, train_y)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    best_state = None
    best_val_mae = float('inf')
    best_epoch = -1
    
    print(f"\nTraining...")
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0.0
        total_n = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred_scaled = model(bx)
            loss = criterion(pred_scaled, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            total_n += bx.size(0)
        
        train_mse = total_loss / max(1, total_n)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(val_X_y)
            val_pred_scaled = clamp01(val_pred_scaled)
            va_pred = inv_scale(val_pred_scaled)
            va_mets = regression_metrics(val_y_raw, va_pred)
        
        # Checkpoint selection (minimize Val MAE)
        if va_mets["MAE"] < best_val_mae:
            best_val_mae = va_mets["MAE"]
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        
        if (epoch + 1) % args.log_interval == 0:
            va_accs = cumulative_accuracy(val_y_raw, va_pred)
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {np.sqrt(train_mse):.4f}")
            print(f"  Val MAE: {va_mets['MAE']:.2f} | RMSE: {va_mets['RMSE']:.2f} | R2: {va_mets['R2']:.4f} | Acc@5: {va_accs['Acc@5']:.3f} | Acc@10: {va_accs['Acc@10']:.3f} | Acc@25: {va_accs['Acc@25']:.3f}")
    
    # Load best and test
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (Val MAE={best_val_mae:.2f})")
    
    print("\nFinal Test Evaluation:")
    model.eval()
    with torch.no_grad():
        te_pred_scaled = model(test_X_y)
        te_pred_scaled = clamp01(te_pred_scaled)
        te_pred = inv_scale(te_pred_scaled)
        
        mets = regression_metrics(test_y_raw, te_pred)
        accs = cumulative_accuracy(test_y_raw, te_pred)
    
    print(f"Test MAE: {mets['MAE']:.2f} | RMSE: {mets['RMSE']:.2f} | R2: {mets['R2']:.4f}")
    print(f"Test Acc@5: {accs['Acc@5']:.3f} | Acc@10: {accs['Acc@10']:.3f} | Acc@25: {accs['Acc@25']:.3f}")


# ============================================================================
# Run selected tasks
# ============================================================================

# Prepare train/val/test splits
train_split = data[data['mode'] == 'train']
val_split = data[data['mode'] == 'val']
test_split = data[data['mode'] == 'test']

# Run classification tasks
if 'artist' in tasks_to_run and dataset_config['artist_relation']:
    train_classification_task('artist', dataset_config['artist_relation'], 
                              train_split, val_split, test_split)

if 'period' in tasks_to_run and dataset_config['period_relation']:
    train_classification_task('period', dataset_config['period_relation'], 
                              train_split, val_split, test_split)

if 'style' in tasks_to_run and dataset_config['style_relation']:
    train_classification_task('style', dataset_config['style_relation'], 
                              train_split, val_split, test_split)
                              
# Run multi-label tag prediction
if 'tag' in tasks_to_run and dataset_config['tag_relation']:
    train_tag_prediction()

# Run year regression
if 'year' in tasks_to_run and dataset_config['year_relation']:
    train_year_prediction()


print("\n" + "="*80)
print("Evaluation Complete!")
print("="*80 + "\n")