#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-probe-bs
#SBATCH -o /media02/hlhdang/projects/pamr/reports/probe_bs_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/probe_bs_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Batch-size / sampler-sync / eval-throughput probe (GPU, protocol-neutral
# measurements; the batch-size DECISION stays with the user because it is a
# training hyperparameter). Data loads once; three bench sections.
echo "ACCEPTANCE probe-bs node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src
"$PY" scripts/probe_batch_size.py --batches 60
rc=$?
echo "ACCEPTANCE probe-bs rc=$rc end=$(date -Iseconds)"
exit "$rc"
