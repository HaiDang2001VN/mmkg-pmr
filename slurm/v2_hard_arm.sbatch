#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-hard
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2hard_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2hard_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=01:00:00

# Journal v2 hard-negative arm: mine top-10 of 100 sampled pool candidates
# with the frozen TransE backbone, then extract train paths (prod + h3).
echo "ACCEPTANCE v2hard node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE v2hard FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}

uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
PRE1_H3=$CP/preproc/wikiart_mkg_v2_h3
V2=$CP/preproc/wikiart_mkg_v2_v2
BASE=$V2/hard
BASE_H3=$V2/hard_h3
TRIPLES=/media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 6 ]; then
  echo "ACCEPTANCE v2hard BLOCKED — /media02 free ${FREE_GB:-unknown}G < 6G"
  exit 4
fi
echo "ACCEPTANCE v2hard disk-ok free=${FREE_GB}G"

flatten() {
  local from="$1" to="$2"
  mkdir -p "$(dirname "$to")"
  rm -rf "$to"
  mv "$from" "$to"
  echo "ACCEPTANCE flatten $from -> $to"
}

run mine uvpython -m scripts.mine_hard_negatives \
  --config pamr/configs/eval/wikiart_mkg_v2_v2_random.yaml \
  --out-dir "$BASE/queries" --m 100 --k 10 --seed 42

run copy-eval-queries bash -c \
  "cp '$PRE1/val_queries.csv' '$BASE/queries/' && cp '$PRE1/test_queries.csv' '$BASE/queries/'"

run paths-hard-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_hard_paths.yaml \
  --splits train --no-compile
TS=$(ls -1d "$BASE"/202* | sort | tail -1)
run flatten-hard-prod flatten "$TS" "$BASE/paths"
run copy-eval-paths-prod bash -c \
  "cp '$PRE1/full/pamr/val_paths.csv' '$PRE1/full/pamr/val_paths_meta.json' \
      '$PRE1/full/pamr/test_paths.csv' '$PRE1/full/pamr/test_paths_meta.json' '$BASE/paths/'"
run verify-hard-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$BASE/paths" --triples "$TRIPLES" --splits train

run paths-hard-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_hard_paths_h3.yaml \
  --splits train --no-compile
TS=$(ls -1d "$BASE_H3"/202* | sort | tail -1)
run flatten-hard-h3 flatten "$TS" "$BASE_H3/paths"
run copy-eval-paths-h3 bash -c \
  "cp '$PRE1_H3/pamr/val_paths.csv' '$PRE1_H3/pamr/val_paths_meta.json' \
      '$PRE1_H3/pamr/test_paths.csv' '$PRE1_H3/pamr/test_paths_meta.json' '$BASE_H3/paths/'"
run verify-hard-h3-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$BASE_H3/paths" --triples "$TRIPLES" --splits train

run counts uvpython - "$BASE" <<'PY'
import sys
import pandas as pd

base = sys.argv[1]
train = pd.read_csv(f"{base}/queries/train_queries.csv")
paths = pd.read_csv(f"{base}/paths/train_paths.csv")
print(f"hard train queries={train.query_id.nunique()} rows={len(train)}")
print(f"hard train paths={len(paths)} path={(paths.hops >= 0).sum()} "
      f"no_path={(paths.hops < 0).sum()}")
assert train.query_id.nunique() == 709
assert len(train) == 7799
assert len(paths) == 7799
print("ACCEPTANCE counts OK")
PY

run checksums bash -c \
  "cmp '$BASE/queries/val_queries.csv' '$PRE1/val_queries.csv' && \
   cmp '$BASE/queries/test_queries.csv' '$PRE1/test_queries.csv' && \
   cmp '$BASE/paths/val_paths.csv' '$PRE1/full/pamr/val_paths.csv' && \
   cmp '$BASE/paths/test_paths.csv' '$PRE1/full/pamr/test_paths.csv' && \
   echo 'ACCEPTANCE checksums val/test identical'"

echo ""
echo "ACCEPTANCE v2hard end=$(date -Iseconds) ALL-STEPS-DONE"
