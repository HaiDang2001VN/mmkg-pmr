#!/bin/bash
#SBATCH -p batch
#SBATCH -N1
#SBATCH -J pamr-netgate
#SBATCH -o /media02/hlhdang/projects/pamr/reports/netgate_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/netgate_%A.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

# NetGate: does a batch (compute) node have outbound internet?
# CPU-only, no GPU quota. ACCEPTANCE net lines: rc=0 = reachable.
# Control target = internal watch UI (cluster network, not internet).
echo "ACCEPTANCE netgate node=$(hostname) start=$(date -Iseconds)"
echo "ACCEPTANCE proxy http_proxy='${http_proxy:-unset}' https_proxy='${https_proxy:-unset}' no_proxy='${no_proxy:-unset}'"

check() {
  local label="$1" url="$2"
  if command -v curl >/dev/null 2>&1; then
    code=$(curl -sI --max-time 8 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)
    rc=$?
    echo "ACCEPTANCE net $label curl_rc=$rc http_code=${code:-none}"
  else
    echo "ACCEPTANCE net $label curl-missing"
  fi
}

check huggingface https://huggingface.co
check wandb https://api.wandb.ai
check pypi https://pypi.org
check watchui http://172.29.74.82:8012/

# DNS layer (python from the venv = reliable getaddrinfo)
/media02/hlhdang/uvenvs/t251cu121-py312/bin/python - <<'PYEOF'
import socket, time
for host in ("huggingface.co", "api.wandb.ai", "pypi.org"):
    t0 = time.time()
    try:
        info = socket.getaddrinfo(host, 443)
        print(f"ACCEPTANCE dns {host} OK {info[0][4][0]} "
              f"{time.time()-t0:.2f}s")
    except Exception as exc:
        print(f"ACCEPTANCE dns {host} FAIL {type(exc).__name__}: {exc}")
PYEOF

echo "ACCEPTANCE netgate end=$(date -Iseconds)"
exit 0
