#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-agg
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2agg_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2agg_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "ACCEPTANCE v2agg node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src
V2=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2_v2
V1=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2
"$PY" -m scripts.aggregate_v2 --v2-root "$V2" --v1-root "$V1" \
  --out-dir assets/report_v2
echo "ACCEPTANCE v2agg end=$(date -Iseconds) ALL-STEPS-DONE"
