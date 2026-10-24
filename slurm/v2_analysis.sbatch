#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-analysis
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2analysis_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2analysis_%A.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# CPU-only J1 analysis: aggregate, stats/CIs, logistic, figures, tables,
# runtime, verification. Depends on the length-only GPU job.
echo "ACCEPTANCE v2analysis node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
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
    echo "ACCEPTANCE v2analysis FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
V2=$CP/preproc/wikiart_mkg_v2_v2

run logistic uvpython -m scripts.path_feature_baseline \
  --v1-root "$PRE1" --v2-root "$V2" --out-dir assets/report_v2
run aggregate uvpython -m scripts.aggregate_v2 \
  --v2-root "$V2" --v1-root "$PRE1" --out-dir assets/report_v2
run stats uvpython -m scripts.stats_v2 \
  --v1-root "$PRE1" --v2-root "$V2" --out-dir assets/report_v2/stats
run figures uvpython -m scripts.figures_v2 \
  --report-dir assets/report_v2 --v2-root "$V2"
run tables uvpython -m scripts.tables_v2 --report-dir assets/report_v2
run runtime uvpython -m scripts.runtime_v2 \
  --v2-root "$V2" --out-dir assets/report_v2
run verify uvpython -m scripts.verify_campaign \
  --v2-root "$V2" --v1-root "$PRE1" --phase J1 --out reports

echo ""
echo "ACCEPTANCE v2analysis end=$(date -Iseconds) ALL-STEPS-DONE"
