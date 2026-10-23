#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-artifacts
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2artifacts_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2artifacts_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Journal v2 artifacts (CPU): random-K10 train queries + paths (prod/h3),
# byte-identical val/test copies, and the K=100 val/test evaluation sets.
echo "ACCEPTANCE v2artifacts node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE v2artifacts FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}

uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
PRE1_H3=$CP/preproc/wikiart_mkg_v2_h3
V2=$CP/preproc/wikiart_mkg_v2_v2
BASE=$V2/random
BASE_H3=$V2/random_h3
K100=$V2/k100
K100_H3=$V2/k100_h3
TRIPLES=/media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 6 ]; then
  echo "ACCEPTANCE v2artifacts BLOCKED — /media02 free ${FREE_GB:-unknown}G < 6G"
  exit 4
fi
echo "ACCEPTANCE v2artifacts disk-ok free=${FREE_GB}G"

flatten() {
  local from="$1" to="$2"
  mkdir -p "$(dirname "$to")"
  rm -rf "$to"
  mv "$from" "$to"
  echo "ACCEPTANCE flatten $from -> $to"
}

# ── 1. Random-K10 training queries (train only) ──────────────────────────
run prepare-random uvpython -m pamr.scripts.prepare \
  --config pamr/configs/eval/wikiart_mkg_v2_v2_random.yaml --workers 10

# ── 2. Byte-identical val/test query copies ──────────────────────────────
run copy-eval-queries bash -c \
  "cp '$PRE1/val_queries.csv' '$BASE/queries/' && cp '$PRE1/test_queries.csv' '$BASE/queries/' && \
   sha256sum '$BASE/queries/val_queries.csv' '$BASE/queries/test_queries.csv' '$PRE1/val_queries.csv' '$PRE1/test_queries.csv'"

# ── 3. Path extraction: random arm prod (train) ──────────────────────────
run paths-random-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_random_paths.yaml \
  --splits train --no-compile
TS=$(ls -1d "$BASE"/202* | sort | tail -1)
run flatten-random-prod flatten "$TS" "$BASE/paths"
run copy-eval-paths-prod bash -c \
  "cp '$PRE1/full/pamr/val_paths.csv' '$PRE1/full/pamr/val_paths_meta.json' \
      '$PRE1/full/pamr/test_paths.csv' '$PRE1/full/pamr/test_paths_meta.json' '$BASE/paths/'"
run verify-random-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$BASE/paths" --triples "$TRIPLES" --splits train

# ── 4. Path extraction: random arm hop3 (train) ──────────────────────────
run paths-random-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_random_paths_h3.yaml \
  --splits train --no-compile
TS=$(ls -1d "$BASE_H3"/202* | sort | tail -1)
run flatten-random-h3 flatten "$TS" "$BASE_H3/paths"
run copy-eval-paths-h3 bash -c \
  "cp '$PRE1_H3/pamr/val_paths.csv' '$PRE1_H3/pamr/val_paths_meta.json' \
      '$PRE1_H3/pamr/test_paths.csv' '$PRE1_H3/pamr/test_paths_meta.json' '$BASE_H3/paths/'"
run verify-random-h3-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$BASE_H3/paths" --triples "$TRIPLES" --splits train

# ── 5. K=100 evaluation queries (val/test) ───────────────────────────────
run prepare-k100 uvpython -m pamr.scripts.prepare \
  --config pamr/configs/eval/wikiart_mkg_v2_v2_k100.yaml --workers 10

# ── 6. K=100 paths: prod (val/test) ──────────────────────────────────────
run paths-k100-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_k100_paths.yaml \
  --splits val,test --no-compile
TS=$(ls -1d "$K100"/202* | sort | tail -1)
run flatten-k100-prod flatten "$TS" "$K100/paths_prod"
run verify-k100-prod uvpython -m pamr.scripts.verify_paths \
  --run-dir "$K100/paths_prod" --triples "$TRIPLES" --splits val,test

# ── 7. K=100 paths: hop3 (val/test) ──────────────────────────────────────
run paths-k100-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_k100_paths_h3.yaml \
  --splits val,test --no-compile
TS=$(ls -1d "$K100_H3"/202* | sort | tail -1)
run flatten-k100-h3 flatten "$TS" "$K100_H3/paths"
run verify-k100-h3 uvpython -m pamr.scripts.verify_paths \
  --run-dir "$K100_H3/paths" --triples "$TRIPLES" --splits val,test

# ── 8. Counts + checksum gate ────────────────────────────────────────────
run counts uvpython - "$BASE" "$K100" <<'PY'
import sys
import pandas as pd

base, k100 = sys.argv[1], sys.argv[2]
train = pd.read_csv(f"{base}/queries/train_queries.csv")
tr_paths = pd.read_csv(f"{base}/paths/train_paths.csv")
val = pd.read_csv(f"{k100}/queries/val_queries.csv")
test = pd.read_csv(f"{k100}/queries/test_queries.csv")
print(f"train queries={train.query_id.nunique()} rows={len(train)}")
print(f"train paths={len(tr_paths)} path={(tr_paths.hops >= 0).sum()} no_path={(tr_paths.hops < 0).sum()}")
print(f"k100 val queries={val.query_id.nunique()} rows={len(val)}")
print(f"k100 test queries={test.query_id.nunique()} rows={len(test)}")
assert train.query_id.nunique() == 709, "train query count changed"
assert len(train) == 7799, "train row count changed"
assert len(tr_paths) == 7799, "train path probe count mismatch"
assert val.query_id.nunique() == 202 and len(val) == 20402, "k100 val shape"
assert test.query_id.nunique() == 239 and len(test) == 24139, "k100 test shape"
print("ACCEPTANCE counts OK")
PY

# ── 9. Verify val/test copies are byte-identical to v1 ───────────────────
run checksums bash -c \
  "cmp '$BASE/queries/val_queries.csv' '$PRE1/val_queries.csv' && \
   cmp '$BASE/queries/test_queries.csv' '$PRE1/test_queries.csv' && \
   cmp '$BASE/paths/val_paths.csv' '$PRE1/full/pamr/val_paths.csv' && \
   cmp '$BASE/paths/test_paths.csv' '$PRE1/full/pamr/test_paths.csv' && \
   echo 'ACCEPTANCE checksums val/test identical'"

echo ""
echo "ACCEPTANCE v2artifacts end=$(date -Iseconds) ALL-STEPS-DONE"
