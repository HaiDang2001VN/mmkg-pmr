#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-probe
#SBATCH -o /media02/hlhdang/projects/pamr/reports/probe_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/probe_%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

echo "ACCEPTANCE node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv.sh

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
echo "ACCEPTANCE nvidia-smi rc=$?"

export UV_PROJECT_ENVIRONMENT=/media02/hlhdang/uvenvs/main-py312
/media02/hlhdang/tools/uv/uv run --project /media02/hlhdang/projects/pamr/src --no-sync python - <<'PY'
import sys
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "py", sys.version.split()[0])
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"gpu{i} {p.name} {p.total_memory/2**30:.1f} GiB cc={p.major}.{p.minor}")
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum()
torch.cuda.synchronize()
print("ACCEPTANCE matmul", y.item())
print("ACCEPTANCE torch-ok")
PY
echo "ACCEPTANCE torch rc=$?"
echo "ACCEPTANCE node=$(hostname) end=$(date -Iseconds)"
