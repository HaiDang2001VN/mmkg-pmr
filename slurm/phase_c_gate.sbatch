#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-gate
#SBATCH -o /media02/hlhdang/projects/pamr/reports/gate_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/gate_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Phase C GPU gate — acceptance under the FINAL env (torch 2.5.1+cu121,
# driver 535.216.03). Proves on-node: torch CUDA init, nn.Transformer /
# nn.MultiheadAttention forward+backward (the CUBLAS-sensitive path),
# AdamW step via a real 1-epoch PAMR run, and one GPU scoring forward via
# score.py machinery, using a synthetic contract-valid TransE checkpoint
# and the REAL Phase B paths in a scratch clone. Prod artifacts untouched;
# scratch self-cleans.
echo "ACCEPTANCE gate node=$(hostname) start=$(date -Iseconds)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
# torch host-side thread cap: 12-CPU allocation minus loader/main reserve
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
cd /media02/hlhdang/projects/pamr/src

SCRATCH=/dev/shm/pamr_gate_${SLURM_JOB_ID}
mkdir -p "$SCRATCH/vlkge" "$SCRATCH/pamr"
trap 'rm -rf "$SCRATCH"' EXIT

run() {
  local name="$1"; shift
  echo ""
  echo "════════ STEP $name ════════ start=$(date -Iseconds)"
  "$@"
  local rc=$?
  echo "ACCEPTANCE step=$name rc=$rc end=$(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then
    echo "ACCEPTANCE gate FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}

# ── 1. torch CUDA audit + the attention-sensitive microbench ──
run torch-audit "$PY" - <<'PYEOF'
import torch, torch.nn as nn
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print("device", p.name, f"cc={p.major}.{p.minor}", f"{p.total_memory/2**30:.1f} GiB")
x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum(); torch.cuda.synchronize()
print("ACCEPTANCE matmul", y.item())
mha = nn.MultiheadAttention(768, 8, batch_first=True).cuda()
in_ = torch.randn(32, 128, 768, device="cuda", requires_grad=True)
out, w = mha(in_, in_, in_, need_weights=False)
out.sum().backward()
torch.cuda.synchronize()
print("ACCEPTANCE mha fwd+bwd", tuple(out.shape))
enc = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(768, 8, dim_feedforward=2048, activation='gelu',
                               batch_first=True), num_layers=4).cuda()
out2 = enc(torch.randn(8, 129, 768, device="cuda"))
out2.sum().backward()
torch.cuda.synchronize()
print("ACCEPTANCE transformer fwd+bwd", tuple(out2.shape))
print("ACCEPTANCE torch-gate-ok")
PYEOF

# ── 2. synthetic contract-valid checkpoint (scratch-resident) ──
run synth "$PY" scripts/throwaway_synth_cpu_ckpt.py \
  --out "$SCRATCH/vlkge/synth_transe.pt" \
  --entity-mapping /media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/full/pamr/entity_mapping.csv \
  --manifest /media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/full/pamr/relation_manifest.json \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml

# ── 3. PAMR 1-epoch GPU run on the real paths (scratch clone) ──
PATHS=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/full/pamr
PRE=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2
cp "$PATHS"/train_paths.csv "$PATHS"/val_paths.csv "$PATHS"/test_paths.csv "$SCRATCH/pamr/"
cp "$PATHS"/train_paths_meta.json "$PATHS"/val_paths_meta.json "$PATHS"/test_paths_meta.json "$SCRATCH/pamr/"
cp "$PATHS"/entity_mapping.csv "$PATHS"/relation_mapping.csv "$PATHS"/relation_manifest.json "$SCRATCH/pamr/" 2>/dev/null
cp "$PRE"/train_queries.csv "$PRE"/val_queries.csv "$PRE"/test_queries.csv "$SCRATCH/pamr/"
sed -e "s|vlkge_config:.*|vlkge_config: /media02/hlhdang/projects/pamr/src/pamr/configs/wikiart_mkg_v2/transe_clip.yaml|" \
    -e "s|vlkge_checkpoint:.*|vlkge_checkpoint: $SCRATCH/vlkge/synth_transe.pt|" \
    -e "s|paths_dir:.*|paths_dir: $SCRATCH/pamr/|" \
    -e "s|queries_dir:.*|queries_dir: $SCRATCH/pamr/|" \
    -e "s|^epochs:.*|epochs: 1|" \
    pamr/configs/pr/smoke_pamr.yaml > "$SCRATCH/smoke_gpu.yaml"
run pamr-gpu "$PY" -m pamr.scripts.train_pamr --config "$SCRATCH/smoke_gpu.yaml" --workers 2

# ── 4. GPU scoring forward (score.py machinery against the scratch clone) ──
"$PY" - <<PYEOF
import yaml, pathlib
p = pathlib.Path("$SCRATCH/pamr/score_proto.yaml")
d = {
  "dataset": "wikiart_mkg_v2",
  "data_path": "/media02/hlhdang/projects/pamr/data/wikiart_mkg_v2/wikiart_mkg_v2_triples.csv",
  "output_dir": "$SCRATCH/pamr",
}
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
print("proto written", p)
PYEOF
sed -e "s|prepare_config:.*|prepare_config: $SCRATCH/pamr/score_proto.yaml|" \
    -e "s|checkpoint_path:.*|checkpoint_path: $SCRATCH/vlkge/synth_transe.pt|" \
    pamr/configs/eval/wikiart_mkg_v2_score.yaml > "$SCRATCH/score_gpu.yaml"
run score-gpu "$PY" -m pamr.models.vlkge.score --config "$SCRATCH/score_gpu.yaml"

echo ""
echo "ACCEPTANCE gate node=$(hostname) end=$(date -Iseconds) ALL-GREEN"
