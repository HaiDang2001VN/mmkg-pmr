#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-verifyB
#SBATCH -o /media02/hlhdang/projects/pamr/reports/verifyB_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/verifyB_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:45:00

# Recovery check: leakage-verification (A/B) on the RESTORED Phase B
# artifacts — proof the snapshot survived intact, NOT a regeneration.
# Fallback on any FAIL: full phase_b_regen.sbatch rerun.
echo "ACCEPTANCE verifyB node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src

BASE=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2
H3BASE=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2_h3
TRIPLES=/media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv

rc_all=0
run() {
  local name="$1"; shift
  echo ""
  echo "════════ STEP $name ════════ start=$(date -Iseconds)"
  "$@"
  local rc=$?
  echo "ACCEPTANCE step=$name rc=$rc end=$(date -Iseconds)"
  [ "$rc" -ne 0 ] && rc_all=1
}

run verify-prod "$PY" -m pamr.scripts.verify_paths \
  --run-dir "$BASE/full/pamr" --triples "$TRIPLES" --splits train,val,test

run verify-h3 "$PY" -m pamr.scripts.verify_paths \
  --run-dir "$H3BASE/pamr" --triples "$TRIPLES" --splits train,val,test

echo ""
echo "ACCEPTANCE verifyB rc=$rc_all end=$(date -Iseconds)"
exit "$rc_all"
