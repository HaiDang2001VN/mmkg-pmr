#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-parity
#SBATCH -o /media02/hlhdang/projects/pamr/reports/parity_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/parity_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:00:00

# CPU-only verification job: evaluate_kge parity harness (legacy per-triple
# reference vs batched pooled path) on real v2 slices + synthetic anchors.
# ALL compute runs inside slurm — never on the login node.
echo "ACCEPTANCE parity node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src
"$PY" scripts/test_eval_kge_parity.py \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml --device cpu
rc=$?
echo "ACCEPTANCE parity rc=$rc end=$(date -Iseconds)"
exit "$rc"
