#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-phaseE
#SBATCH -o /media02/hlhdang/projects/pamr/reports/phaseE_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/phaseE_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Phase E/G — seed-1 rerun + core-5 ablations, sequential on ONE GPU
# (each run is minutes-scale). Every run's paths_dir is output-own: the
# chain clones the regenerated path artifacts first so artifacts never
# clobber prod or each other. Configs already carry the distinct dirs.
echo "ACCEPTANCE phaseE node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
# torch host-side thread cap: 12-CPU allocation minus loader/main reserve
export OMP_NUM_THREADS=10
export PYTHONUNBUFFERED=1
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
    echo "ACCEPTANCE phaseE FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$PY" "$@"; }

CP=/media02/hlhdang/projects/pamr/checkpoints
VDIR=$CP/vlkge/wikiart_mkg_v2
PRE=$CP/preproc/wikiart_mkg_v2
SRCP=$PRE/full/pamr
H3P=$CP/preproc/wikiart_mkg_v2_h3/pamr

MISMATCH=0
for c in transe_wikiart_mkg_v2_train.pt complex_wikiart_mkg_v2_train.pt \
         distmult_wikiart_mkg_v2_train.pt rotate_wikiart_mkg_v2_train.pt; do
  [ -f "$VDIR/$c" ] || { echo "ACCEPTANCE phaseE BLOCKED — missing ckpt $c"; MISMATCH=1; }
done
[ -d "$H3P" ] || { echo "ACCEPTANCE phaseE BLOCKED — missing h3 paths"; MISMATCH=1; }
[ "$MISMATCH" = 0 ] || exit 3
echo "ACCEPTANCE phaseE checks-ok"

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 12 ]; then
  echo "ACCEPTANCE phaseE BLOCKED — /media02 free ${FREE_GB:-unknown}G < 12G"
  exit 4
fi
echo "ACCEPTANCE phaseE disk-ok free=${FREE_GB}G"

clone_paths() {
  local target="$1" src="$2"
  mkdir -p "$target" || return 1
  cp "$src"/{train,val,test}_paths.csv "$target/" || return 1
  cp "$src"/*_paths_meta.json "$target/" || return 1
  cp "$src"/{entity_mapping.csv,relation_mapping.csv,relation_manifest.json} "$target/" || return 1
  echo "ACCEPTANCE cloned paths from $src -> $target"
}

ABL=$PRE/full/ablations

# ── seed-1 (n=2 std partner for prod) ──
run clone-seed1 clone_paths "$ABL/seed1" "$SRCP"
run pamr-seed1 uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr_seed1.yaml

# ── ablation: fine-tuned backbone ──
run clone-finetune clone_paths "$ABL/finetune" "$SRCP"
run pamr-finetune uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr_finetune.yaml

# ── ablation: depth N=12 ──
run clone-depth12 clone_paths "$ABL/depth12" "$SRCP"
run pamr-depth12 uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr_depth12.yaml

# ── ablation: readout cls ──
run clone-cls clone_paths "$ABL/cls" "$SRCP"
run pamr-cls uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr_cls.yaml

# ── ablation: contrastive (listwise_softmax + lf_temperature 0.1) ──
run clone-lf_temp clone_paths "$ABL/lf_temp" "$SRCP"
run pamr-lf_temp uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr_lf_temp.yaml

# ── ablation: hop bound K=3 (fresh h3 paths extraction from Phase B) ──
run pamr-hop3 uvpython -m pamr.scripts.train_pamr \
  --config pamr/configs/pr/pamr_hop3.yaml

echo ""
echo "ACCEPTANCE phaseE node=$(hostname) end=$(date -Iseconds) ALL-STEPS-DONE"
