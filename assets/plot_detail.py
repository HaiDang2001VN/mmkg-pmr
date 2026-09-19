import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

path = os.path.dirname(os.path.abspath(__file__))


def load_result(path):
    """Load result.csv into a Method-indexed frame with flat rel_metric columns.

    Supports the grouped two-row header (row 1 = relations, row 2 = metrics)
    and falls back to the legacy flat single-header layout (resultv0.csv).
    """
    with open(path) as f:
        lines = f.readlines()
    if len(lines) > 1 and lines[1].lstrip().startswith('Method'):
        raw = pd.read_csv(path, header=[0, 1], index_col=0)
        df = pd.DataFrame(index=raw.index)
        df.index.name = 'Method'
        current = None
        for rel, metric in raw.columns:
            metric = str(metric).strip()
            if metric == 'MRR Overall':
                df['MRR Overall'] = raw[(rel, metric)].values
                continue
            rel = str(rel).strip()
            if rel and not rel.startswith('Unnamed'):
                current = rel
            df[f'{current}_{metric}'] = raw[(rel, metric)].values
        return df
    return pd.read_csv(path).set_index('Method')


df = load_result(os.path.join(path, 'result.csv'))

methods = [
    'VLKGE-TransE', 'VLKGE-RotatE', 'VLKGE-DistMult',
    'VLKGE-ComplEx', 'Path-Aware Reasoning',
    'Path-Aware Multi-modal Reasoning',
]
method_labels = ['TransE', 'RotatE', 'DistMult', 'ComplEx', 'PAR', 'PAMR']
relations = ['isInfluencedBy', 'isInfluencedOn', 'isPupilOf', 'isTeacherOf']
relation_labels = ['InfluencedBy', 'InfluencedOn', 'PupilOf', 'TeacherOf']

metrics = {
    'MRR':  ('MRR',    'MRR.svg'),
    'H1':   ('Hits@1', 'Hits1.svg'),
    'H3':   ('Hits@3', 'Hits3.svg'),
    'H5':   ('Hits@5', 'Hits5.svg'),
}

all_values = []
for metric_key in metrics:
    for rel in relations:
        all_values.extend(df[f'{rel}_{metric_key}'].tolist())
y_min = min(all_values) * 0.9
y_max = max(all_values) * 1.1

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

for metric_key, (metric_label, filename) in metrics.items():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ax_idx, (rel, rel_label) in enumerate(zip(relations, relation_labels)):
        ax = axes[ax_idx]
        values = df[f'{rel}_{metric_key}'].tolist()

        ax.bar(range(len(methods)), values, color=colors,
               edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(y_min, y_max)
        ax.set_title(rel_label, fontsize=10, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        if ax_idx == 0:
            ax.set_ylabel(metric_label, fontsize=10)

    fig.suptitle(metric_label, fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = os.path.join(path, filename)
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

# Combined multi-panel figure: rows = metrics, cols = relations
combined_name = 'per_relation.svg'
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharey=False)

for row_idx, (metric_key, metric_label) in enumerate(metrics.items()):
    for ax_idx, (rel, rel_label) in enumerate(zip(relations, relation_labels)):
        ax = axes[row_idx][ax_idx]
        values = df[f'{rel}_{metric_key}'].tolist()

        ax.bar(range(len(methods)), values, color=colors,
               edgecolor='white', linewidth=0.8)
        ax.set_xticks(range(len(methods)))
        if row_idx == 3:
            ax.set_xticklabels(method_labels, rotation=45, ha='right',
                               fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(y_min, y_max)
        if row_idx == 0:
            ax.set_title(rel_label, fontsize=12, fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel(metric_label, fontsize=11, fontweight='bold')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))

fig.tight_layout(w_pad=1.0, h_pad=1.0)
os.makedirs(os.path.join(path, 'figures'), exist_ok=True)
fig.savefig(os.path.join(path, 'figures', combined_name), format='svg',
            bbox_inches='tight')
plt.close(fig)
print(f'Saved figures/{combined_name}')
