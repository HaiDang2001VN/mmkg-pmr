import os
import numpy as np
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))
result_path = os.path.join(path, 'result.csv')
count_path = os.path.join(path, 'count.csv')

relations = ['isInfluencedBy', 'isInfluencedOn', 'isPupilOf', 'isTeacherOf']
metrics = ['MRR', 'H1', 'H3', 'H5']


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


df = load_result(result_path)

counts = pd.read_csv(count_path, header=None, names=['Relation', 'Count'],
                     skipinitialspace=True)
counts['Relation'] = counts['Relation'].astype(str).str.strip()
counts = counts.set_index('Relation')['Count']
missing = [rel for rel in relations if rel not in counts.index]
if missing:
    raise ValueError(f'count.csv is missing relations: {missing}')
W = np.array([float(counts.loc[rel]) for rel in relations])

result = df.reset_index()
for metric in metrics:
    mat = df[[f'{rel}_{metric}' for rel in relations]].to_numpy()
    result[f'{metric}_avg'] = mat.mean(axis=1).round(4)
    result[f'{metric}_std'] = mat.std(axis=1, ddof=1).round(4)
    wavg = (mat * W).sum(axis=1) / W.sum()
    wstd = np.sqrt(((mat - wavg[:, None]) ** 2 * W).sum(axis=1) / W.sum())
    result[f'{metric}_wavg'] = wavg.round(4)
    result[f'{metric}_wstd'] = wstd.round(4)

result['MRR_Overall'] = df['MRR Overall'].values
result.to_csv(os.path.join(path, 'average.csv'), index=False, float_format='%.4f')
print(result.to_string(index=False))
