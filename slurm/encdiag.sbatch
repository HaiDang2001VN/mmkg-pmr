#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-encdiag
#SBATCH -o /media02/hlhdang/projects/pamr/reports/encdiag_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/encdiag_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00

echo "ACCEPTANCE encdiag node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src
"$PY" -m scripts.diagnose_pamr_encoder
echo "ACCEPTANCE encdiag end=$(date -Iseconds) ALL-OK"
