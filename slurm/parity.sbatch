#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-parity
#SBATCH -o /media02/hlhdang/projects/pamr/reports/parity_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/parity_%A.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

echo "ACCEPTANCE parity node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src
"$PY" -m scripts.test_pamr_encoder_parity
echo "ACCEPTANCE parity end=$(date -Iseconds)"
