import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

path = os.path.dirname(os.path.abspath(__file__))
bfs_path = os.path.join(path, 'bfs.csv')

RELATIONS = None
NUM_PROBES = 2222  # 202 val queries x (1 gold + 10 sampled negatives)

df = pd.read_csv(bfs_path)
df = df[(df['split'] == 'val') & (df['mask_query_edges'] == True)]
# keep the v3 batch: it contains the full grid; v2 configs reproduce it exactly
raw = df[df['worker_version'] == 'v3'].copy()
raw['H'] = raw['max_hops'].astype(int)
raw['F'] = raw['max_fanout'].astype(int)
raw['D'] = raw['decay_factor'].astype(float)
raw['reach_pct'] = raw['found'] / NUM_PROBES * 100.0

H_VALS = [2, 3, 5]
F_VALS = [10, 50, 100]
D_VALS = [0.2, 0.5, 1.0]

# ---------------- exported aggregates (single source for the paper text) ----
grid = raw[['H', 'F', 'D', 'found', 'reach_pct', 'wall_time_s',
            'peak_concurrent_rss_mb', 'avg_hops']].copy()
grid = grid.sort_values(['H', 'F', 'D'])
grid.columns = ['H', 'F', 'D', 'found', 'reach_pct', 'wall_s', 'rss_mb',
                'avg_hops']
grid_path = os.path.join(path, 'bfs_sweep_grid.csv')
grid.to_csv(grid_path, index=False, float_format='%.4f')

agg = (raw.groupby(['H', 'F'])['found']
          .agg(found_mean='mean', found_std=lambda s: s.std(ddof=1))
          .reset_index())
agg['reach_mean'] = agg['found_mean'] / NUM_PROBES * 100.0
agg['reach_std'] = agg['found_std'] / NUM_PROBES * 100.0
agg = agg.round({'found_mean': 1, 'found_std': 2, 'reach_mean': 2,
                 'reach_std': 2})
agg_path = os.path.join(path, 'bfs_sweep_ci.csv')
agg.to_csv(agg_path, index=False)
print(agg.to_string(index=False))

# ---------------- figure ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

# Panel (a): coverage vs fanout, lines per H, error bars = std over decay
ax = axes[0]
a_color = '#1f77b4'
dash_by_H = {5: '-', 3: (0, (4, 2)), 2: (0, (1, 1.5))}
for h, ls in dash_by_H.items():
    sub = agg[agg['H'] == h].set_index('F').reindex(F_VALS)
    ax.errorbar(range(len(F_VALS)), sub['reach_mean'],
                yerr=sub['reach_std'], ls=ls, marker='o', ms=5,
                lw=1.8, capsize=3, color=a_color,
                label=f'$K={h}$')
ax.set_xticks(range(len(F_VALS)))
ax.set_xticklabels([str(f) for f in F_VALS])
ax.set_xlabel('Fanout $F$', fontsize=11)
ax.set_ylabel('Candidate probes with paths (%)', fontsize=11)
ax.set_ylim(-3, 106)
ax.tick_params(labelsize=10)
ax.axhline(100, color='grey', lw=0.6, ls=':')
ax.annotate('no probe has a\npath at $K=2$',
            xy=(0, 0.2), xytext=(0.05, 22), fontsize=9, color='#1f77b4',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=0.9))
ax.legend(fontsize=9, loc='center right', title='±1 std',
          title_fontsize=8)
ax.set_title('(a) Reachability vs. fanout and hop bound', fontsize=11)

# Panel (b): wall-clock vs memory at H = 5; 3 polylines per decay factor
# (solid/dashed/dotted = D), F marked by distinct point markers on each line.
ax = axes[1]
b_color = '#d62728'
dash_by_D = {0.2: '-', 0.5: (0, (4, 2)), 1.0: (0, (1, 1))}
marker_by_F = {10: 'o', 50: 's', 100: '^'}
h5 = raw[raw['H'] == 5]
for d, ls in dash_by_D.items():
    sub = h5[h5['D'] == d].sort_values('F')
    ax.plot(sub['peak_concurrent_rss_mb'], sub['wall_time_s'], ls=ls,
            lw=1.6, color=b_color, zorder=2)
for f, mk in marker_by_F.items():
    sub = h5[h5['F'] == f]
    ax.plot(sub['peak_concurrent_rss_mb'], sub['wall_time_s'], ls='none',
            marker=mk, ms=7, mfc='white', mec=b_color, mew=1.6, zorder=3)
ax.set_xlabel('Peak memory (MB)', fontsize=11)
ax.set_ylabel('Wall-clock per config (s)', fontsize=11)
ax.tick_params(labelsize=10)
ax.set_xlim(200, 229)
ax.set_ylim(0, 16.5)
proxy_d = [Line2D([], [], color=b_color, lw=1.6, ls=ls,
                  label=f'$\\delta={d}$')
           for d, ls in dash_by_D.items()]
proxy_f = [Line2D([], [], color=b_color, ls='none', marker=mk, ms=7,
                  mfc='white', mew=1.6, label=f'$F={f}$')
           for f, mk in marker_by_F.items()]
leg1 = ax.legend(handles=proxy_d, fontsize=9, loc='lower right')
ax.add_artist(leg1)
ax.legend(handles=proxy_f, fontsize=9, loc='upper left',
          title='fanout', title_fontsize=8)
ax.set_title('Extraction cost vs. memory at $K=5$', fontsize=11)

fig.tight_layout(w_pad=1.5)
out = os.path.join(path, 'bfs_sweep.svg')
fig.savefig(out, format='svg', bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
print(f'Saved {grid_path}')
print(f'Saved {agg_path}')
