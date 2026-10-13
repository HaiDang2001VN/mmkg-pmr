#!/bin/bash
#SBATCH -p batch
#SBATCH -N1
#SBATCH -J pamr-gpucache
#SBATCH -o /media02/hlhdang/projects/pamr/reports/gpucache_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/gpucache_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -x gpu03

# GPU cached-eval gate: the cached-scored evaluation path has only ever
# run parity on CPU. This gate runs the SAME harness on CUDA for all
# four scorer families (real slices, artist rows forced) with
# pooled-cached vs the per-triple reference - plus GPU memory-headroom
# observation. Gate for every GPU use of the cached path.
echo "ACCEPTANCE gpucache node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /media02/hlhdang/projects/pamr/src
"$PY" scripts/test_eval_kge_parity.py \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml --device cuda
rc=$?
echo "ACCEPTANCE gpucache rc=$rc end=$(date -Iseconds)"
exit "$rc"
