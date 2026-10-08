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
"$PY" scripts/test_loader_sampler_equiv.py \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml
rc=$?
echo "ACCEPTANCE equiv rc=$rc end=$(date -Iseconds)"
exit "$rc"
