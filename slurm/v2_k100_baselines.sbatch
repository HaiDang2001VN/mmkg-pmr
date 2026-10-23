#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-k100
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2k100_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2k100_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:30:00

# Journal v2 — K=100 baseline score/evaluate chains + path-subset metrics.
echo "ACCEPTANCE v2k100 node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE v2k100 FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}

uvpython() { "$PY" "$@"; }

V2=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2_v2
K100=$V2/k100

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 6 ]; then
  echo "ACCEPTANCE v2k100 BLOCKED — /media02 free ${FREE_GB:-unknown}G < 6G"
  exit 4
fi
echo "ACCEPTANCE v2k100 disk-ok free=${FREE_GB}G"

for p in "$K100/queries/val_queries.csv" "$K100/queries/test_queries.csv" \
         "$K100/paths_prod/val_paths.csv" "$K100/paths_prod/test_paths.csv"; do
  [ -f "$p" ] || { echo "ACCEPTANCE v2k100 BLOCKED — missing $p"; exit 3; }
done
echo "ACCEPTANCE v2k100 artifacts-ok"

run parity uvpython -m scripts.test_pamr_encoder_parity

for m in transe complex distmult rotate; do
  sfx=""
  [ "$m" != "transe" ] && sfx="_$m"
  run "score-$m" uvpython -m pamr.models.vlkge.score \
    --config "pamr/configs/eval/wikiart_mkg_v2_v2_k100_score$sfx.yaml"
  run "eval-$m" uvpython -m pamr.scripts.evaluate \
    --config "pamr/configs/eval/wikiart_mkg_v2_v2_k100_neg_eval$sfx.yaml"
done

for pair in "transe:TransE" "complex:ComplEx" "distmult:DistMult" "rotate:RotatE"; do
  key=${pair%%:*}; label=${pair##*:}
  run "subset-$key" uvpython -m scripts.path_subset_control \
    --queries "$K100/queries/test_queries.csv" \
    --paths "$K100/paths_prod/test_paths.csv" \
    --scores "$K100/full/vlkge/$label/test_scores.csv" \
    --out "$K100/full/vlkge/$label/path_control.json"
done

echo ""
echo "ACCEPTANCE v2k100 end=$(date -Iseconds) ALL-STEPS-DONE"
