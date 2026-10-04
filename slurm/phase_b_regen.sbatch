#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-regenB
#SBATCH -N1
#SBATCH -o /media02/hlhdang/projects/pamr/reports/regenB_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/regenB_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# Phase B — CPU artifact regeneration (prepare.py -> preprocess.py BFS prod + h3
# -> verify_paths checks A/B). Torch imports run CPU-only in the universal env;
# no GPU code path executes. Downstream GPU phases are handled separately.
echo "ACCEPTANCE regenB node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv.sh
export UV_PROJECT_ENVIRONMENT=/media02/hlhdang/uvenvs/main-py312
export OMP_NUM_THREADS=4
cd /media02/hlhdang/projects/pamr/src

run() {
  local name="$1"; shift
  echo ""
  echo "════════ STEP $name ════════ start=$(date -Iseconds)"
  "$@"
  local rc=$?
  echo "ACCEPTANCE step=$name rc=$rc end=$(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then
    echo "ACCEPTANCE regenB FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}

uvpython() { "$UV_BIN" run --project . --no-sync python "$@"; }

# ── 1. Registry acceptance (uses the full triples CSV; CPU only) ──
run registry uvpython -m pamr.scripts.check_registry \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml \
  --expect-raw 22 --expect-total 40

# ── 2. Query preparation (target relations, K: train 1 / val 10 / test 10) ──
run prepare uvpython -m pamr.scripts.prepare \
  --config pamr/configs/eval/wikiart_mkg_v2_neg_sampling.yaml

# ── 3. Path extraction prod (K=5/F=100/D=0.5, own-edge masked) ──
run paths-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_paths.yaml --no-compile

# ── 4. Path extraction hop-3 (ablation evidence, identical pipeline) ──
run paths-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_paths_h3.yaml --no-compile

# ── 5. Flatten + leakage-safety verification on regenerated artifacts ──
BASE=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2
H3BASE=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2_h3

flatten() {
  local from="$1" to="$2"
  mkdir -p "$(dirname "$to")"
  rm -rf "$to"
  mv "$from" "$to"
  echo "ACCEPTANCE flatten $from -> $to"
}
PROD_TS=$(ls -1d "$BASE"/202* 2>/dev/null | sort | tail -1)
H3_TS=$(ls -1d "$H3BASE"/202* 2>/dev/null | sort | tail -1)
if [ -n "$PROD_TS" ]; then
  run flatten-prod flatten "$PROD_TS" "$BASE/full/pamr"
else
  echo "ACCEPTANCE flatten-prod SKIPPED (no prod run dir)"
fi
if [ -n "$H3_TS" ]; then
  run flatten-h3 flatten "$H3_TS" "$H3BASE/pamr"
else
  echo "ACCEPTANCE flatten-h3 SKIPPED (no h3 run dir)"
fi

echo ""
echo "════════ VERIFY prod ════════"
if [ -d "$BASE/full/pamr" ]; then
  run verify-prod uvpython -m pamr.scripts.verify_paths \
    --run-dir "$BASE/full/pamr" \
    --triples /media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
    --splits train,val,test
else
  echo "ACCEPTANCE verify-prod SKIPPED (no $BASE/full/pamr)"
fi

echo ""
echo "════════ VERIFY h3 ════════"
if [ -d "$H3BASE/pamr" ]; then
  run verify-h3 uvpython -m pamr.scripts.verify_paths \
    --run-dir "$H3BASE/pamr" \
    --triples /media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv \
    --splits train,val,test
else
  echo "ACCEPTANCE verify-h3 SKIPPED (no $H3BASE/pamr)"
fi

echo ""
echo "ACCEPTANCE regenB node=$(hostname) end=$(date -Iseconds) ALL-STEPS-DONE"
