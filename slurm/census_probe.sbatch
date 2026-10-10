#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-census
#SBATCH -o /media02/hlhdang/projects/pamr/reports/census_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/census_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:45:00

# Census probe: val/test facts whose true tail sits OUTSIDE their eval
# candidate pool (grounding for the pooled-evaluator union fix).
echo "ACCEPTANCE census node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src
"$PY" scripts/census_eval_pools.py \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml
rc=$?
echo "ACCEPTANCE census rc=$rc end=$(date -Iseconds)"
exit "$rc"
