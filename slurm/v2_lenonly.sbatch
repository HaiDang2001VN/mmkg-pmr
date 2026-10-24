#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-lenonly
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2lenonly_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2lenonly_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=01:30:00

# Length-only path-feature ablation: 5 PAMR seeds, GPU only.
echo "ACCEPTANCE v2lenonly node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src

run() {
  local name="$1"; shift
  echo ""
  echo "════════ STEP $name ════════ start=$(date -Iseconds)"
  "$@"
  local rc=$?
  echo "ACCEPTANCE step=$name rc=$rc end=$(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then
    echo "ACCEPTANCE v2lenonly FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
V2=$CP/preproc/wikiart_mkg_v2_v2
K100=$V2/k100
ARM=random

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 6 ]; then
  echo "ACCEPTANCE v2lenonly BLOCKED — /media02 free ${FREE_GB:-unknown}G < 6G"
  exit 4
fi
echo "ACCEPTANCE v2lenonly disk-ok free=${FREE_GB}G"

clone_paths() {
  local src="$1" dst="$2"
  mkdir -p "$dst" || return 1
  cp "$src"/{train,val,test}_paths.csv "$dst/" || return 1
  cp "$src"/*_paths_meta.json "$dst/" || return 1
  cp "$src"/{entity_mapping.csv,relation_mapping.csv,relation_manifest.json} "$dst/" || return 1
}

for seed in 42 1 2 3 4; do
  RUN=$V2/campaign/lenonly/seed$seed
  rm -rf "$RUN"
  mkdir -p "$RUN"
  clone_paths "$V2/$ARM/paths" "$RUN/paths"
  run "train-lenonly-s$seed" uvpython -m pamr.scripts.train_pamr \
    --config pamr/configs/pr/pamr_v2_campaign_lenonly.yaml \
    --paths-dir "$RUN/paths" --queries-dir "$V2/$ARM/queries" --seed "$seed"
  run "control-k10-s$seed" uvpython -m scripts.path_subset_control \
    --queries "$V2/$ARM/queries/test_queries.csv" \
    --paths "$V2/$ARM/paths/test_paths.csv" \
    --scores "$RUN/paths/test_scores.csv" \
    --backbone-scores "$PRE1/full/vlkge/TransE/test_scores.csv" \
    --out "$RUN/path_control_k10.json"
  run "control-k100-s$seed" uvpython -m scripts.path_subset_control \
    --queries "$K100/queries/test_queries.csv" \
    --paths "$K100/paths_prod/test_paths.csv" \
    --scores "$RUN/paths/test_k100_scores.csv" \
    --backbone-scores "$K100/full/vlkge/TransE/test_scores.csv" \
    --out "$RUN/path_control_k100.json"
done

echo ""
echo "ACCEPTANCE v2lenonly end=$(date -Iseconds) ALL-STEPS-DONE"
