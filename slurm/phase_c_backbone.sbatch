#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-bb
#SBATCH -o /media02/hlhdang/projects/pamr/reports/backbone_%A_%j.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/backbone_%A_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# Phase C — one VL-KGE backbone retrain (train scope, 20 epochs), under the
# uv WORKSPACE contract (t251cu121-py312, torch 2.5.1+cu121, driver 535).
# Usage: sbatch --export=BACKBONE=transe slurm/phase_c_backbone.sbatch
#        BACKBONE ∈ {transe, complex, distmult, rotate, struct}
# Scratch-checkpoint model: saves land on node /dev/shm; USR1 and EXIT sync
# the full checkpoint back to media02, so a requeue on ANY node resumes
# from the last best checkpoint; at normal completion the slim checkpoint
# (model_state + metadata, no optimizer) replaces the stage file.
echo "ACCEPTANCE backbone=$BACKBONE node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
# torch host-side thread cap: 12-CPU allocation minus loader/main reserve
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
cd /media02/hlhdang/projects/pamr/src

SCRATCH=/dev/shm/pamr_bb_${SLURM_JOB_ID}
STAGE_DIR=/media02/hlhdang/projects/pamr/checkpoints/vlkge/wikiart_mkg_v2
mkdir -p "$SCRATCH"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

case "$BACKBONE" in
  transe)   CFG=pamr/configs/wikiart_mkg_v2/transe_clip.yaml;   CANON=transe_wikiart_mkg_v2_train.pt ;;
  complex)  CFG=pamr/configs/wikiart_mkg_v2/complex_clip.yaml;  CANON=complex_wikiart_mkg_v2_train.pt ;;
  distmult) CFG=pamr/configs/wikiart_mkg_v2/distmult_clip.yaml; CANON=distmult_wikiart_mkg_v2_train.pt ;;
  rotate)   CFG=pamr/configs/wikiart_mkg_v2/rotate_clip.yaml;   CANON=rotate_wikiart_mkg_v2_train.pt ;;
  struct)   CFG=pamr/configs/wikiart_mkg_v2/transe_struct.yaml; CANON=transe_wikiart_mkg_v2_struct_train.pt ;;
  *) echo "ACCEPTANCE backbone FAILED: unknown BACKBONE=$BACKBONE"; exit 2 ;;
esac
SCRATCH_CKPT="$SCRATCH/$CANON"
STAGE="$STAGE_DIR/stage_$CANON"

sync_to_stage() {
  if [ -f "$SCRATCH_CKPT" ]; then
    echo "ACCEPTANCE stage-sync $SCRATCH_CKPT -> $STAGE ($(date -Iseconds))"
    cp -f "$SCRATCH_CKPT" "$STAGE" || echo "ACCEPTANCE stage-sync FAILED"
  else
    echo "ACCEPTANCE stage-sync skipped (no scratch checkpoint yet)"
  fi
}
handle_usr1() {
  echo "ACCEPTANCE USR1 received — syncing + clean exit for requeue ($(date -Iseconds))"
  sync_to_stage
  exit 0
}
trap handle_usr1 USR1

"$PY" - <<PYEOF
import torch
print("ACCEPTANCE torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "ACCEPTANCE torch sanity FAILED rc=$rc"
  exit "$rc"
fi

# ── Resume staging: media02 stage (or canonical slim) -> scratch ──
RES_ARGS=()
if [ -f "$STAGE" ]; then
  cp -f "$STAGE" "$SCRATCH_CKPT"
  RES_ARGS=(--resume_from "$SCRATCH_CKPT")
  echo "ACCEPTANCE resuming from stage file $STAGE"
elif [ -f "$STAGE_DIR/$CANON" ]; then
  echo "ACCEPTANCE canonical slim checkpoint already exists — training complete for $BACKBONE"
  exit 0
fi

"$PY" -m pamr.scripts.train_embedding \
  --config "$CFG" \
  --embedding_scope train \
  --save_path "$SCRATCH_CKPT" \
  "${RES_ARGS[@]}"
rc=$?
echo "ACCEPTANCE train rc=$rc end=$(date -Iseconds)"
if [ "$rc" -ne 0 ]; then
  sync_to_stage
  echo "ACCEPTANCE backbone=$BACKBONE FAILED rc=$rc"
  exit "$rc"
fi

# ── Completion: slim copyback (model_state + metadata only) ──
"$PY" - <<PYEOF
import torch, os
src = "$SCRATCH_CKPT"
dst = "$STAGE_DIR/$CANON"
ck = torch.load(src, map_location="cpu")
slim = {
    "model_state_dict": ck["model_state_dict"],
    "epoch": ck.get("epoch"),
    "score": ck.get("score"),
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
with open(dst, "rb") as f:
    sz = os.path.getsize(dst)
print(f"ACCEPTANCE slim-copyback {dst} ({sz/(2**20):.0f} MiB)")
os.remove("$STAGE") if os.path.exists("$STAGE") else None
PYEOF
src_rc=$?
echo "ACCEPTANCE slim rc=$src_rc"
[ "$src_rc" -ne 0 ] && sync_to_stage

echo "ACCEPTANCE backbone=$BACKBONE end=$(date -Iseconds) DONE"
exit "$src_rc"
