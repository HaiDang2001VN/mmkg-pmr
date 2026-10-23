#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-remask
#SBATCH -o /media02/hlhdang/projects/pamr/reports/remask_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/remask_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Protocol fix (journal v2 round 2): mask EVERY direct edge between the query
# head and tail (any relation, both orientations), then re-extract all paths.
# Backs up the old-mask path CSVs for the appendix evidence.
echo "ACCEPTANCE remask node=$(hostname) start=$(date -Iseconds)"
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
    echo "ACCEPTANCE remask FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
PRE1=$CP/preproc/wikiart_mkg_v2
PRE1_H3=$CP/preproc/wikiart_mkg_v2_h3
V2=$CP/preproc/wikiart_mkg_v2_v2
K100=$V2/k100
K100_H3=$V2/k100_h3
BK=$V2/oldmask_backup
TRIPLES=/media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 10 ]; then
  echo "ACCEPTANCE remask BLOCKED — /media02 free ${FREE_GB:-unknown}G < 10G"
  exit 4
fi
echo "ACCEPTANCE remask disk-ok free=${FREE_GB}G"

flatten() {
  local from="$1" to="$2"
  mkdir -p "$(dirname "$to")"
  rm -rf "$to"
  mv "$from" "$to"
  echo "ACCEPTANCE flatten $from -> $to"
}

# ── backups of the old-mask path CSVs (small; evidence for the appendix) ──
run backup bash -c "
  set -e
  mkdir -p '$BK/v1' '$BK/v1_h3' '$BK/random' '$BK/random_h3' '$BK/hard' '$BK/hard_h3' '$BK/k100' '$BK/k100_h3'
  cp '$PRE1/full/pamr'/*_paths.csv '$PRE1/full/pamr'/*_paths_meta.json '$BK/v1/' 2>/dev/null || true
  cp '$PRE1_H3/pamr'/*_paths.csv '$PRE1_H3/pamr'/*_paths_meta.json '$BK/v1_h3/' 2>/dev/null || true
  cp '$V2/random/paths'/*_paths.csv '$V2/random/paths'/*_paths_meta.json '$BK/random/' 2>/dev/null || true
  cp '$V2/random_h3/paths'/*_paths.csv '$V2/random_h3/paths'/*_paths_meta.json '$BK/random_h3/' 2>/dev/null || true
  cp '$V2/hard/paths'/*_paths.csv '$V2/hard/paths'/*_paths_meta.json '$BK/hard/' 2>/dev/null || true
  cp '$V2/hard_h3/paths'/*_paths.csv '$V2/hard_h3/paths'/*_paths_meta.json '$BK/hard_h3/' 2>/dev/null || true
  cp '$K100/paths_prod'/*_paths.csv '$K100/paths_prod'/*_paths_meta.json '$BK/k100/' 2>/dev/null || true
  cp '$K100_H3/paths'/*_paths.csv '$K100_H3/paths'/*_paths_meta.json '$BK/k100_h3/' 2>/dev/null || true
  [ -f '$V2/pilot_gate.json' ] && cp '$V2/pilot_gate.json' '$BK/pilot_gate_round1.json' || true
  echo 'ACCEPTANCE old-mask paths backed up'
"

# ── v1 production paths (all splits) — first run compiles the patched C++ ─
run paths-v1-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_paths.yaml
TS=$(ls -1d "$PRE1"/202* | sort | tail -1)
run flatten-v1-prod flatten "$TS" "$PRE1/full/pamr"
run verify-v1-prod uvpython -m pamr.scripts.verify_paths \
  --run-dir "$PRE1/full/pamr" --triples "$TRIPLES" --splits train,val,test

run paths-v1-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_paths_h3.yaml --no-compile
TS=$(ls -1d "$PRE1_H3"/202* | sort | tail -1)
run flatten-v1-h3 flatten "$TS" "$PRE1_H3/pamr"
run verify-v1-h3 uvpython -m pamr.scripts.verify_paths \
  --run-dir "$PRE1_H3/pamr" --triples "$TRIPLES" --splits train,val,test

# ── v2 random arm (train re-extracted; new v1 eval paths copied) ─────────
run paths-random-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_random_paths.yaml \
  --splits train --no-compile
TS=$(ls -1d "$V2/random"/202* | sort | tail -1)
run flatten-random-prod flatten "$TS" "$V2/random/paths"
run copy-random-eval bash -c \
  "cp '$PRE1/full/pamr/val_paths.csv' '$PRE1/full/pamr/val_paths_meta.json' \
      '$PRE1/full/pamr/test_paths.csv' '$PRE1/full/pamr/test_paths_meta.json' '$V2/random/paths/'"
run verify-random-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$V2/random/paths" --triples "$TRIPLES" --splits train

run paths-random-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_random_paths_h3.yaml \
  --splits train --no-compile
TS=$(ls -1d "$V2/random_h3"/202* | sort | tail -1)
run flatten-random-h3 flatten "$TS" "$V2/random_h3/paths"
run copy-random-h3-eval bash -c \
  "cp '$PRE1_H3/pamr/val_paths.csv' '$PRE1_H3/pamr/val_paths_meta.json' \
      '$PRE1_H3/pamr/test_paths.csv' '$PRE1_H3/pamr/test_paths_meta.json' '$V2/random_h3/paths/'"
run verify-random-h3-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$V2/random_h3/paths" --triples "$TRIPLES" --splits train

# ── v2 hard arm ──────────────────────────────────────────────────────────
run paths-hard-prod uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_hard_paths.yaml \
  --splits train --no-compile
TS=$(ls -1d "$V2/hard"/202* | sort | tail -1)
run flatten-hard-prod flatten "$TS" "$V2/hard/paths"
run copy-hard-eval bash -c \
  "cp '$PRE1/full/pamr/val_paths.csv' '$PRE1/full/pamr/val_paths_meta.json' \
      '$PRE1/full/pamr/test_paths.csv' '$PRE1/full/pamr/test_paths_meta.json' '$V2/hard/paths/'"
run verify-hard-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$V2/hard/paths" --triples "$TRIPLES" --splits train

run paths-hard-h3 uvpython -m pamr.models.pr.preprocess \
  --config pamr/configs/pr/wikiart_mkg_v2_v2_hard_paths_h3.yaml \
  --splits train --no-compile
TS=$(ls -1d "$V2/hard_h3"/202* | sort | tail -1)
run flatten-hard-h3 flatten "$TS" "$V2/hard_h3/paths"
run copy-hard-h3-eval bash -c \
  "cp '$PRE1_H3/pamr/val_paths.csv' '$PRE1_H3/pamr/val_paths_meta.json' \
      '$PRE1_H3/pamr/test_paths.csv' '$PRE1_H3/pamr/test_paths_meta.json' '$V2/hard_h3/paths/'"
run verify-hard-h3-train uvpython -m pamr.scripts.verify_paths \
  --run-dir "$V2/hard_h3/paths" --triples "$TRIPLES" --splits train

# ── K100 eval paths ──────────────────────────────────────────────────────
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

# ── path-length distributions (train/test alignment evidence) ────────────
run distributions uvpython - "$V2" <<'PY'
import sys
import pandas as pd

v2 = sys.argv[1]
for tag, path in (
        ("train-pos", f"{v2}/random/paths/train_paths.csv"),
        ("train-neg", f"{v2}/random/paths/train_paths.csv"),
        ("test-pos", f"{v2}/random/paths/test_paths.csv"),
        ("test-neg", f"{v2}/random/paths/test_paths.csv")):
    p = pd.read_csv(path)
    p = p[p.is_positive == (1 if tag.endswith("pos") else 0)]
    vc = p.hops.value_counts().sort_index().to_dict()
    print(f"{tag}: n={len(p)} mean={p.hops.replace(-1, 0).mean():.2f} "
          f"median={p.hops.median()} no_path={(p.hops < 0).sum()} "
          f"hops={vc}")
PY

# ── baseline path-conditional / presence-control reports (new paths) ─────
for pair in "transe:TransE" "complex:ComplEx" "distmult:DistMult" "rotate:RotatE"; do
  key=${pair%%:*}; label=${pair##*:}
  run "subset-k10-$key" uvpython -m scripts.path_subset_control \
    --queries "$V2/random/queries/test_queries.csv" \
    --paths "$V2/random/paths/test_paths.csv" \
    --scores "$PRE1/full/vlkge/$label/test_scores.csv" \
    --out "$PRE1/full/vlkge/$label/path_control.json"
done
for pair in "transe:TransE" "complex:ComplEx" "distmult:DistMult" "rotate:RotatE"; do
  key=${pair%%:*}; label=${pair##*:}
  run "subset-k100-$key" uvpython -m scripts.path_subset_control \
    --queries "$K100/queries/test_queries.csv" \
    --paths "$K100/paths_prod/test_paths.csv" \
    --scores "$K100/full/vlkge/$label/test_scores.csv" \
    --out "$K100/full/vlkge/$label/path_control.json"
done

echo ""
echo "ACCEPTANCE remask end=$(date -Iseconds) ALL-STEPS-DONE"
