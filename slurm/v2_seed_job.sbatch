#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-v2-seed
#SBATCH -o /media02/hlhdang/projects/pamr/reports/v2seed_%A_%j.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/v2seed_%A_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# Backbone retrain for one extra seed: four models sequentially in ONE GPU
# job (continuous GPU use, no CPU-only phases). Seed 42 canonicals are
# untouched; extra seeds write <name>_s<SEED>.pt slims.
# Usage: sbatch v2_seed_job.sbatch 1
SEED=${1:-1}
echo "ACCEPTANCE v2seed seed=$SEED node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export OMP_NUM_THREADS=10
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /media02/hlhdang/projects/pamr/src

STAGE_DIR=/media02/hlhdang/projects/pamr/checkpoints/vlkge/wikiart_mkg_v2
SCRATCH_ROOT=/dev/shm/pamr_seed_${SLURM_JOB_ID}
mkdir -p "$SCRATCH_ROOT"
CUR_CKPT=""
CUR_STAGE=""

sync_cur() {
  if [ -n "$CUR_CKPT" ] && [ -f "$CUR_CKPT" ]; then
    echo "ACCEPTANCE stage-sync $CUR_CKPT -> $CUR_STAGE ($(date -Iseconds))"
    cp -f "$CUR_CKPT" "$CUR_STAGE" || echo "ACCEPTANCE stage-sync FAILED"
  fi
}
handle_usr1() {
  echo "ACCEPTANCE USR1 — syncing current model ($(date -Iseconds))"
  sync_cur
  exit 0
}
cleanup() { rm -rf "$SCRATCH_ROOT"; }
trap handle_usr1 USR1
trap cleanup EXIT

FREE_GB=$(df -BG /media02 | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ -z "$FREE_GB" ] || [ "$FREE_GB" -lt 12 ]; then
  echo "ACCEPTANCE v2seed BLOCKED — /media02 free ${FREE_GB:-unknown}G < 12G"
  exit 4
fi
echo "ACCEPTANCE v2seed disk-ok free=${FREE_GB}G"

"$PY" - <<PYEOF
import torch
print("ACCEPTANCE torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
PYEOF

train_one() {
  local bb="$1"
  local cfg can base
  case "$bb" in
    transe)   cfg=pamr/configs/wikiart_mkg_v2/transe_clip.yaml;   base=transe_wikiart_mkg_v2_train.pt ;;
    complex)  cfg=pamr/configs/wikiart_mkg_v2/complex_clip.yaml;  base=complex_wikiart_mkg_v2_train.pt ;;
    distmult) cfg=pamr/configs/wikiart_mkg_v2/distmult_clip.yaml; base=distmult_wikiart_mkg_v2_train.pt ;;
    rotate)   cfg=pamr/configs/wikiart_mkg_v2/rotate_clip.yaml;   base=rotate_wikiart_mkg_v2_train.pt ;;
    *) echo "ACCEPTANCE seed FAILED unknown $bb"; return 2 ;;
  esac
  if [ "$SEED" = "42" ]; then
    can="$base"
  else
    can="${base%.pt}_s${SEED}.pt"
  fi
  local scratch="$SCRATCH_ROOT/$bb"
  mkdir -p "$scratch"
  local ckpt="$scratch/$can"
  local stage="$STAGE_DIR/stage_$can"
  CUR_CKPT="$ckpt"; CUR_STAGE="$stage"
  local res=()

  if [ -f "$STAGE_DIR/$can" ]; then
    echo "ACCEPTANCE $bb seed$SEED canonical exists — skip"
    return 0
  fi
  if [ -f "$stage" ]; then
    cp -f "$stage" "$ckpt"
    res=(--resume_from "$ckpt")
    echo "ACCEPTANCE $bb seed$SEED resuming from stage"
  fi

  "$PY" -m pamr.scripts.train_embedding \
    --config "$cfg" --embedding_scope train --seed "$SEED" \
    --save_path "$ckpt" "${res[@]}"
  local rc=$?
  echo "ACCEPTANCE train-$bb-seed$SEED rc=$rc end=$(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then
    sync_cur
    return "$rc"
  fi

  "$PY" - <<PYEOF
import torch, os
src = "$ckpt"; dst = "$STAGE_DIR/$can"
ck = torch.load(src, map_location="cpu")
slim = {
    "model_state_dict": ck["model_state_dict"],
    "epoch": ck.get("epoch"), "score": ck.get("score"),
    "model_name": ck.get("model_name"),
    "dataset_name": ck.get("dataset_name"),
    "embedding_scope": ck.get("embedding_scope"),
    "relation_registry": ck.get("relation_registry"),
    "entity_mapping_hash": ck.get("entity_mapping_hash"),
    "relation_mapping_hash": ck.get("relation_mapping_hash"),
    "num_neg_samples": ck.get("num_neg_samples"),
    "batch_size": ck.get("batch_size"),
    "best_val_mrr": ck.get("best_val_mrr"),
}
torch.save(slim, dst)
print(f"ACCEPTANCE slim {dst} ({os.path.getsize(dst)/(2**20):.0f} MiB)")
if os.path.exists("$stage"):
    os.remove("$stage")
PYEOF
  local src_rc=$?
  if [ "$src_rc" -ne 0 ]; then
    sync_cur
    return "$src_rc"
  fi
  CUR_CKPT=""; CUR_STAGE=""
  rm -rf "$scratch"
  return 0
}

for bb in transe complex distmult rotate; do
  echo ""
  echo "════════ $bb seed $SEED ════════ start=$(date -Iseconds)"
  train_one "$bb"
  rc=$?
  echo "ACCEPTANCE step=$bb-seed$SEED rc=$rc end=$(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then
    echo "ACCEPTANCE v2seed FAILED at $bb rc=$rc"
    exit "$rc"
  fi
done

echo ""
echo "ACCEPTANCE v2seed seed=$SEED end=$(date -Iseconds) ALL-STEPS-DONE"
