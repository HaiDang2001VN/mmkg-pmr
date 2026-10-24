#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-rescore
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2rescore_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2rescore_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# GPU-only rescoring of the 8 extra-seed backbone checkpoints (K10 + K<=100)
# plus their path-subset controls.
echo "ACCEPTANCE v2rescore node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE v2rescore FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
V2=$CP/preproc/wikiart_mkg_v2_v2
K100=$V2/k100

for seed in 1 2; do
  for pair in "transe:TransE" "complex:ComplEx" "distmult:DistMult" "rotate:RotatE"; do
    key=${pair%%:*}; label=${pair##*:}
    [ -f "$CP/vlkge/wikiart_mkg_v2/${key}_wikiart_mkg_v2_train_s${seed}.pt" ] || {
      echo "ACCEPTANCE v2rescore BLOCKED — missing ${key} seed$seed"; exit 3; }
  done
done
echo "ACCEPTANCE v2rescore checkpoints-ok"

run gen-configs uvpython -m scripts.make_backbone_seed_configs --seeds 1 2

for seed in 1 2; do
  for pair in "transe:TransE" "complex:ComplEx" "distmult:DistMult" "rotate:RotatE"; do
    key=${pair%%:*}; label=${pair##*:}
    run "score-k10-$key-s$seed" uvpython -m pamr.models.vlkge.score \
      --config "pamr/configs/eval/wikiart_mkg_v2_score_${key}_s${seed}.yaml"
    run "eval-k10-$key-s$seed" uvpython -m pamr.scripts.evaluate \
      --config "pamr/configs/eval/wikiart_mkg_v2_neg_eval_${key}_s${seed}.yaml"
    run "score-k100-$key-s$seed" uvpython -m pamr.models.vlkge.score \
      --config "pamr/configs/eval/wikiart_mkg_v2_v2_k100_score_${key}_s${seed}.yaml"
    run "eval-k100-$key-s$seed" uvpython -m pamr.scripts.evaluate \
      --config "pamr/configs/eval/wikiart_mkg_v2_v2_k100_neg_eval_${key}_s${seed}.yaml"
    run "subset-k10-$key-s$seed" uvpython -m scripts.path_subset_control \
      --queries "$V2/random/queries/test_queries.csv" \
      --paths "$V2/random/paths/test_paths.csv" \
      --scores "$PRE1/full/seed_scores/${label}_s${seed}/test_scores.csv" \
      --out "$PRE1/full/seed_scores/${label}_s${seed}/path_control_k10.json"
    run "subset-k100-$key-s$seed" uvpython -m scripts.path_subset_control \
      --queries "$K100/queries/test_queries.csv" \
      --paths "$K100/paths_prod/test_paths.csv" \
      --scores "$K100/full/seed_scores/${label}_s${seed}/test_scores.csv" \
      --out "$K100/full/seed_scores/${label}_s${seed}/path_control_k100.json"
  done
done

echo ""
echo "ACCEPTANCE v2rescore end=$(date -Iseconds) ALL-STEPS-DONE"
