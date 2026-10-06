#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-phaseD
#SBATCH -o /media02/hlhdang/projects/pamr/reports/phaseD_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/phaseD_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Phase D — one sequential GPU job: baseline scores x4 -> PAMR prod -> PAR
# -> evaluate.py -> verify_paths check C (candidate identity).
# Gate: every Phase C checkpoint must exist (checked in-script).
echo "ACCEPTANCE phaseD node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
# torch host-side thread cap: 12-CPU allocation minus loader/main reserve
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
cd /media02/hlhdang/projects/pamr/src

run() {
  local name="$1"; shift
  echo ""
  echo "════════ STEP $name ════════ start=$(date -Iseconds)"
  "$@"
  local rc=$?
  echo "ACCEPTANCE step=$name rc=$rc end=$(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then
    echo "ACCEPTANCE phaseD FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
VDIR=$CP/vlkge/wikiart_mkg_v2
PRE=$CP/preproc/wikiart_mkg_v2
SRCP=$PRE/full/pamr

# ── 0. Gate: Phase C checkpoints ──
MISMATCH=0
for c in transe_wikiart_mkg_v2_train.pt complex_wikiart_mkg_v2_train.pt \
         distmult_wikiart_mkg_v2_train.pt rotate_wikiart_mkg_v2_train.pt \
         transe_wikiart_mkg_v2_struct_train.pt; do
  [ -f "$VDIR/$c" ] || { echo "ACCEPTANCE phaseD BLOCKED — missing ckpt $c"; MISMATCH=1; }
done
[ "$MISMATCH" = 0 ] || exit 3
echo "ACCEPTANCE phaseD checkpoints-ok"

# helpers: clone the regenerated path artifacts into an output-own dir
clone_paths() {
  local target="$1"
  mkdir -p "$target"
  cp "$SRCP"/{train,val,test}_paths.csv "$target/"
  cp "$SRCP"/*_paths_meta.json "$target/"
  cp "$SRCP"/{entity_mapping.csv,relation_mapping.csv,relation_manifest.json} "$target/"
  echo "ACCEPTANCE cloned paths -> $target"
}

# ── 1. Baseline scoring (4 chains; target-relations split, seconds each) ──
run score-transe uvpython -m pamr.models.vlkge.score \
  --config pamr/configs/eval/wikiart_mkg_v2_score.yaml
run score-complex uvpython -m pamr.models.vlkge.score \
  --config pamr/configs/eval/wikiart_mkg_v2_score_complex.yaml
run score-distmult uvpython -m pamr.models.vlkge.score \
  --config pamr/configs/eval/wikiart_mkg_v2_score_distmult.yaml
run score-rotate uvpython -m pamr.models.vlkge.score \
  --config pamr/configs/eval/wikiart_mkg_v2_score_rotate.yaml

# ── 2. PAMR prod (writes into full/pamr, its own paths_dir) ──
run pamr-prod uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr.yaml

# ── 3. PAR (output-own dir; clone paths, run struct-backbone config) ──
run clone-par clone_paths "$PRE/full/par"
# the struct backbone saves the checkpoint whose state must satisfy
# embedding_scope=train + the same registry (transe_struct config)
run pamr-par uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr_par.yaml

# ── 4. Baseline metric computation (CPU inside the same job) ──
run evaluate-baselines uvpython -m pamr.scripts.evaluate \
  --config pamr/configs/eval/wikiart_mkg_v2_neg_eval.yaml

# ── 5. verify C: PAMR test scores vs TransE train-scope baseline scores ──
run verify-C uvpython -m pamr.scripts.verify_paths \
  --run-dir "$SRCP" \
  --triples /media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
  --splits test \
  --pamr-scores "$SRCP/test_scores.csv" \
  --vlkge-scores "$PRE/test_scores.csv" \
  --scores-split test \
  --prepared-queries "$PRE/test_queries.csv"

echo ""
echo "ACCEPTANCE phaseD node=$(hostname) end=$(date -Iseconds) ALL-STEPS-DONE"
