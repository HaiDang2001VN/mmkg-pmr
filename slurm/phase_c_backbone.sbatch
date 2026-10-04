#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-bb
#SBATCH --comment=phaseC-backbone
#SBATCH -o /media02/hlhdang/projects/pamr/reports/backbone_%A_%j.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/backbone_%A_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

# Phase C — one VL-KGE backbone retrain (train scope, 20 epochs).
# Usage: sbatch --export=BACKBONE=transe phase_c_backbone.sbatch
#        BACKBONE ∈ {transe, complex, distmult, rotate, struct}
# Resume: pass RESUME=1 to continue from the model's best checkpoint.
echo "ACCEPTANCE backbone=$BACKBONE node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv.sh
export UV_PROJECT_ENVIRONMENT=/media02/hlhdang/uvenvs/main-py312
cd /media02/hlhdang/projects/pamr/src

case "$BACKBONE" in
  transe)  CFG=pamr/configs/wikiart_mkg_v2/transe_clip.yaml;  CKPT=../checkpoints/vlkge/wikiart_mkg_v2/transe_wikiart_mkg_v2_train.pt ;;
  complex) CFG=pamr/configs/wikiart_mkg_v2/complex_clip.yaml; CKPT=../checkpoints/vlkge/wikiart_mkg_v2/complex_wikiart_mkg_v2_train.pt ;;
  distmult) CFG=pamr/configs/wikiart_mkg_v2/distmult_clip.yaml; CKPT=../checkpoints/vlkge/wikiart_mkg_v2/distmult_wikiart_mkg_v2_train.pt ;;
  rotate)  CFG=pamr/configs/wikiart_mkg_v2/rotate_clip.yaml;  CKPT=../checkpoints/vlkge/wikiart_mkg_v2/rotate_wikiart_mkg_v2_train.pt ;;
  struct)  CFG=pamr/configs/wikiart_mkg_v2/transe_struct.yaml; CKPT=../checkpoints/vlkge/wikiart_mkg_v2/transe_wikiart_mkg_v2_struct_train.pt ;;
  *) echo "ACCEPTANCE backbone FAILED: unknown BACKBONE=$BACKBONE"; exit 2 ;;
esac

RES_ARGS=()
if [ -n "$RESUME" ] && [ -f "$CKPT" ]; then
  RES_ARGS=(--resume_from "$CKPT")
  echo "ACCEPTANCE resuming from $CKPT"
fi

"$(command -v "$UV_BIN" || echo /media02/hlhdang/tools/uv/uv)" run --project . --no-sync python -m pamr.scripts.train_embedding \
  --config "$CFG" \
  --embedding_scope train \
  --save_path "$CKPT" \
  "${RES_ARGS[@]}"
rc=$?
echo "ACCEPTANCE backbone=$BACKBONE rc=$rc end=$(date -Iseconds)"
if [ "$rc" -ne 0 ]; then
  echo "ACCEPTANCE backbone=$BACKBONE FAILED rc=$rc"
fi
exit "$rc"
