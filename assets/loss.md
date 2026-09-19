$$\boxed{\mathcal{L}_{\text{ranking}} = \frac{\sum\limits_{q \in \mathcal{Q}}\;\sum\limits_{p \in \mathbf{P}_q}\;\sum\limits_{n \in \mathbf{N}_q} \max\!\big(0,\; 1 - s^{(p)} + s^{(n)}\big)}{\sum\limits_{q \in \mathcal{Q}} |\mathbf{P}_q| \cdot |\mathbf{N}_q|}}$$
Where:
- $\mathcal{Q}$ = set of queries (filtered: only queries with both ≥1 positive and ≥1 negative)
- $\mathbf{P}_q, \mathbf{N}_q$ = positive / negative sample indices for query $q$
- $s^{(p)}$ = model score for a positive sample (wanted high)
- $s^{(n)}$ = model score for a negative sample (wanted low)
- Margin = $1.0$ — the positive score must exceed the negative score by at least 1 to incur zero loss