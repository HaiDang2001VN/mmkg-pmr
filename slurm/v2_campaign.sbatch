#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-campaign
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2camp_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2camp_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=08:00:00

# Journal v2 full campaign. Usage: sbatch v2_campaign.sbatch A|B
#   A: campaign, par, finetune, depth12       (5 seeds each)
#   B: cls, lf_temp, hop3, no_mask, v1fixed, loser-arm analysis
# Every run gets an output-own cloned paths dir and reports K=10 + K=100.
GROUP=${1:-A}
echo "ACCEPTANCE v2camp group=$GROUP node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE v2camp FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
PRE1_H3=$CP/preproc/wikiart_mkg_v2_h3
V2=$CP/preproc/wikiart_mkg_v2_v2
K100=$V2/k100
RUNS=$V2/campaign
TRIPLES=/media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv
SEEDS="42 1 2 3 4"

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 10 ]; then
  echo "ACCEPTANCE v2camp BLOCKED — /media02 free ${FREE_GB:-unknown}G < 10G"
  exit 4
fi
echo "ACCEPTANCE v2camp disk-ok free=${FREE_GB}G"

LABEL="${V2_LABEL:-}"
if [ -z "$LABEL" ]; then
  LABEL=$("$PY" - "$V2/pilot_gate.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
sel = d.get("selected") or {}
print(sel.get("label", ""))
PY
)
fi
run gen-configs uvpython -m scripts.make_v2_campaign_configs \
  --gate "$V2/pilot_gate.json" --v2-root "$V2" --v1-root "$PRE1" \
  --label "$LABEL"
[ -n "$LABEL" ] || { echo "ACCEPTANCE v2camp BLOCKED — no pilot selection"; exit 3; }
ARM=${LABEL%_*}; LOSS=${LABEL#*_}
LOSER_ARM=$([ "$ARM" = "random" ] && echo hard || echo random)
echo "ACCEPTANCE v2camp selected=$LABEL arm=$ARM loss=$LOSS loser=$LOSER_ARM"

for p in "$V2/$ARM/paths/train_paths.csv" "$V2/$ARM/queries/train_queries.csv" \
         "$K100/queries/test_queries.csv" "$K100/paths_prod/test_paths.csv"; do
  [ -f "$p" ] || { echo "ACCEPTANCE v2camp BLOCKED — missing $p"; exit 3; }
done
echo "ACCEPTANCE v2camp artifacts-ok"

clone_paths() {
  local src="$1" dst="$2"
  mkdir -p "$dst" || return 1
  cp "$src"/{train,val,test}_paths.csv "$dst/" || return 1
  cp "$src"/*_paths_meta.json "$dst/" || return 1
  cp "$src"/{entity_mapping.csv,relation_mapping.csv,relation_manifest.json} "$dst/" || return 1
}

analyze() {
  local run_dir="$1" qdir="$2" paths="$3" scores="$4" backbone="$5" tag="$6"
  uvpython -m scripts.path_subset_control \
    --queries "$qdir/test_queries.csv" --paths "$paths/test_paths.csv" \
    --scores "$scores" --backbone-scores "$backbone" \
    --out "$run_dir/path_control_${tag}.json"
}

run_one() {
  local name="$1" cfg="$2" src_paths="$3" qdir="$4" seed="$5" save="$6"
  local k100_mask="${7:-$K100/paths_prod}"
  local run_dir="$RUNS/$name/seed$seed"
  rm -rf "$run_dir"
  mkdir -p "$run_dir"
  clone_paths "$src_paths" "$run_dir/paths"
  local args=(--config "$cfg" --paths-dir "$run_dir/paths" \
              --queries-dir "$qdir" --seed "$seed")
  [ "$save" = "1" ] && args+=(--save-checkpoint)
  run "train-${name}-s${seed}" uvpython -m pamr.scripts.train_pamr "${args[@]}"
  analyze "$run_dir" "$qdir" "$src_paths" "$run_dir/paths/test_scores.csv" \
    "$PRE1/full/vlkge/TransE/test_scores.csv" k10
  if [ -f "$run_dir/paths/test_k100_scores.csv" ]; then
    analyze "$run_dir" "$K100/queries" "$k100_mask" \
      "$run_dir/paths/test_k100_scores.csv" \
      "$K100/full/vlkge/TransE/test_scores.csv" k100
  fi
}

if [ "$GROUP" = "A" ]; then
  for pair in "transe:TransE" "complex:ComplEx" "distmult:DistMult" "rotate:RotatE"; do
    key=${pair%%:*}; label=${pair##*:}
    run "subset-k10-$key" uvpython -m scripts.path_subset_control \
      --queries "$V2/$ARM/queries/test_queries.csv" \
      --paths "$V2/$ARM/paths/test_paths.csv" \
      --scores "$PRE1/full/vlkge/$label/test_scores.csv" \
      --out "$PRE1/full/vlkge/$label/path_control.json"
    run "subset-k100-$key" uvpython -m scripts.path_subset_control \
      --queries "$K100/queries/test_queries.csv" \
      --paths "$K100/paths_prod/test_paths.csv" \
      --scores "$K100/full/vlkge/$label/test_scores.csv" \
      --out "$K100/full/vlkge/$label/path_control.json"
  done
fi

CONFIGS=""
[ "$GROUP" = "A" ] && CONFIGS="campaign par finetune depth12"
[ "$GROUP" = "B" ] && CONFIGS="cls lf_temp hop3 no_mask no_feats"
[ -n "$CONFIGS" ] || { echo "ACCEPTANCE v2camp BLOCKED — bad group"; exit 3; }

for cfg in $CONFIGS; do
  case "$cfg" in
    hop3)
      src_paths="$V2/${ARM}_h3/paths" ;;
    *)
      src_paths="$V2/$ARM/paths" ;;
  esac
  cfgfile="pamr/configs/pr/pamr_v2_campaign_${cfg}.yaml"
  [ "$cfg" = "campaign" ] && cfgfile="pamr/configs/pr/pamr_v2_campaign.yaml"
  k100_mask="$K100/paths_prod"
  [ "$cfg" = "hop3" ] && k100_mask="$V2/k100_h3/paths"
  for seed in $SEEDS; do
    save=0
    [ "$cfg" = "campaign" ] && [ "$seed" = "42" ] && save=1
    [ "$cfg" = "par" ] && [ "$seed" = "42" ] && save=1
    run_one "$cfg" "$cfgfile" "$src_paths" "$V2/$ARM/queries" "$seed"       "$save" "$k100_mask"
  done
done

if [ "$GROUP" = "B" ]; then
  run_one "v1fixed" "pamr/configs/pr/pamr_v1fixed.yaml" \
    "$PRE1/full/pamr" "$PRE1" 42 0
  run_one "loser_${LOSER_ARM}" "pamr/configs/pr/pamr_v2_campaign.yaml" \
    "$V2/$LOSER_ARM/paths" "$V2/$LOSER_ARM/queries" 42 0
fi

echo ""
echo "ACCEPTANCE v2camp group=$GROUP end=$(date -Iseconds) ALL-STEPS-DONE"
