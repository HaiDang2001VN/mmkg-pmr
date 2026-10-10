#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-equiv
#SBATCH -o /media02/hlhdang/projects/pamr/reports/equiv_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/equiv_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:45:00

# CPU-only gate: host-bound engineering equivalence (KGDataset
# tensorization + grouped-vectorized negative sampler). Runs BEFORE any
# backbone relaunch. Login node serves login only.
echo "ACCEPTANCE equiv node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src

# ── Step 1: import battery (post-restore env sanity) ──
"$PY" - <<'PYEOF'
import torch, pandas, numpy, sklearn, yaml, tqdm
import pamr
print("ACCEPTANCE battery", "torch", torch.__version__, "pandas", pandas.__version__,
      "numpy", numpy.__version__, "sklearn", sklearn.__version__,
      "pamr", pamr.__name__, "OK")
PYEOF
rc=$?
echo "ACCEPTANCE battery rc=$rc"
if [ "$rc" -ne 0 ]; then echo "ACCEPTANCE equiv FAILED battery"; exit "$rc"; fi

# ── Step 2: workspace lock ↔ members consistency ──
cd /media02/hlhdang/projects
/media02/hlhdang/tools/uv/uv lock --check
rc=$?
echo "ACCEPTANCE uv-lock-check rc=$rc"
if [ "$rc" -ne 0 ]; then echo "ACCEPTANCE equiv FAILED lock-check"; exit "$rc"; fi
cd /media02/hlhdang/projects/pamr/src

# ── Step 3: equivalence harness (loader + sampler, real data) ──
"$PY" scripts/test_loader_sampler_equiv.py \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml
rc=$?
echo "ACCEPTANCE equiv rc=$rc end=$(date -Iseconds)"
exit "$rc"
