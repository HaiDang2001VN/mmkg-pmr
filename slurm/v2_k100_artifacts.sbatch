#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-k100-art
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2k100art_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2k100art_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Journal v2 — K=100 ("up to 100" variable negatives) eval artifacts.
# The artist split-local pools hold fewer than 100 eligible candidates for
# every test query, so the protocol uses min(100, available) negatives.
echo "ACCEPTANCE v2k100art node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE v2k100art FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

V2=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2_v2
K100=$V2/k100
K100_H3=$V2/k100_h3
TRIPLES=/media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 6 ]; then
  echo "ACCEPTANCE v2k100art BLOCKED — /media02 free ${FREE_GB:-unknown}G < 6G"
  exit 4
fi
echo "ACCEPTANCE v2k100art disk-ok free=${FREE_GB}G"

flatten() {
  local from="$1" to="$2"
  mkdir -p "$(dirname "$to")"
  rm -rf "$to"
  mv "$from" "$to"
  echo "ACCEPTANCE flatten $from -> $to"
}

run prepare-k100 uvpython -m pamr.scripts.prepare \
  --config pamr/configs/eval/wikiart_mkg_v2_v2_k100.yaml --workers 10

run paths-k100-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_k100_paths.yaml \
  --splits val,test --no-compile
TS=$(ls -1d "$K100"/202* | sort | tail -1)
run flatten-k100-prod flatten "$TS" "$K100/paths_prod"
run verify-k100-prod uvpython -m pamr.scripts.verify_paths \
  --run-dir "$K100/paths_prod" --triples "$TRIPLES" --splits val,test

run paths-k100-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_k100_paths_h3.yaml \
  --splits val,test --no-compile
TS=$(ls -1d "$K100_H3"/202* | sort | tail -1)
run flatten-k100-h3 flatten "$TS" "$K100_H3/paths"
run verify-k100-h3 uvpython -m pamr.scripts.verify_paths \
  --run-dir "$K100_H3/paths" --triples "$TRIPLES" --splits val,test

run counts uvpython - "$K100" "$K100_H3" <<'PY'
import sys
import pandas as pd

k100, k100_h3 = sys.argv[1], sys.argv[2]
for split, expected_q in (("val", 202), ("test", 239)):
    q = pd.read_csv(f"{k100}/queries/{split}_queries.csv")
    p = pd.read_csv(f"{k100}/paths_prod/{split}_paths.csv")
    p3 = pd.read_csv(f"{k100_h3}/paths/{split}_paths.csv")
    k = q[q.is_positive == 0].groupby("query_id").size()
    print(f"{split}: queries={q.query_id.nunique()} rows={len(q)} "
          f"negs min/median/max={int(k.min())}/{int(k.median())}/{int(k.max())} "
          f"paths={len(p)} path={(p.hops >= 0).sum()} "
          f"h3paths={len(p3)} h3path={(p3.hops >= 0).sum()}")
    assert q.query_id.nunique() == expected_q, f"{split} query count"
    assert len(p) == len(q) and len(p3) == len(q), "path/probe mismatch"
print("ACCEPTANCE counts OK")
PY

echo ""
echo "ACCEPTANCE v2k100art end=$(date -Iseconds) ALL-STEPS-DONE"
