#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-pilot
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2pilot_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2pilot_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=04:00:00

# Journal v2 pilot: {random,hard} x {pairwise,listwise}, seed 42.
# Each run gets an output-own cloned paths dir; the gate then requires a
# path-content gain on the all-path subset over backbone and presence control.
echo "ACCEPTANCE v2pilot node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE v2pilot FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}

uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
V2=$CP/preproc/wikiart_mkg_v2_v2
RUNS=$V2/pilot

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 8 ]; then
  echo "ACCEPTANCE v2pilot BLOCKED — /media02 free ${FREE_GB:-unknown}G < 8G"
  exit 4
fi
echo "ACCEPTANCE v2pilot disk-ok free=${FREE_GB}G"

for arm in random hard; do
  for p in "$V2/$arm/queries/train_queries.csv" "$V2/$arm/paths/train_paths.csv" \
           "$V2/$arm/paths/val_paths.csv" "$V2/$arm/paths/test_paths.csv"; do
    [ -f "$p" ] || { echo "ACCEPTANCE v2pilot BLOCKED — missing $p"; exit 3; }
  done
done
echo "ACCEPTANCE v2pilot artifacts-ok"

run parity uvpython -m scripts.test_pamr_encoder_parity

clone_paths() {
  local src="$1" dst="$2"
  mkdir -p "$dst" || return 1
  cp "$src"/{train,val,test}_paths.csv "$dst/" || return 1
  cp "$src"/*_paths_meta.json "$dst/" || return 1
  cp "$src"/{entity_mapping.csv,relation_mapping.csv,relation_manifest.json} "$dst/" || return 1
}

for arm in random hard; do
  for loss in pw lf; do
    label="${arm}_${loss}"
    RUN=$RUNS/$label
    rm -rf "$RUN"
    mkdir -p "$RUN"
    clone_paths "$V2/$arm/paths" "$RUN/paths"
    CFG=pamr/configs/pr/pamr_v2_pairwise.yaml
    [ "$loss" = "lf" ] && CFG=pamr/configs/pr/pamr_v2_listwise.yaml
    run "train-$label" uvpython -m pamr.scripts.train_pamr \
      --config "$CFG" --paths-dir "$RUN/paths" \
      --queries-dir "$V2/$arm/queries" --seed 42
    run "control-$label" uvpython -m scripts.path_subset_control \
      --queries "$V2/$arm/queries/test_queries.csv" \
      --paths "$V2/$arm/paths/test_paths.csv" \
      --scores "$RUN/paths/test_scores.csv" \
      --backbone-scores "$PRE1/full/vlkge/TransE/test_scores.csv" \
      --out "$RUN/path_control.json"
  done
done

run gate uvpython -m scripts.check_pilot_gate \
  --run random_pw "$RUNS/random_pw/paths/pamr_training_metrics.json" "$RUNS/random_pw/path_control.json" \
  --run random_lf "$RUNS/random_lf/paths/pamr_training_metrics.json" "$RUNS/random_lf/path_control.json" \
  --run hard_pw "$RUNS/hard_pw/paths/pamr_training_metrics.json" "$RUNS/hard_pw/path_control.json" \
  --run hard_lf "$RUNS/hard_lf/paths/pamr_training_metrics.json" "$RUNS/hard_lf/path_control.json" \
  --out "$V2/pilot_gate.json"

echo ""
echo "ACCEPTANCE v2pilot end=$(date -Iseconds) ALL-STEPS-DONE"
