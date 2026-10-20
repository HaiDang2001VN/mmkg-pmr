#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-slimsweep
#SBATCH -o /media02/hlhdang/projects/pamr/reports/slimsweep_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/slimsweep_%A.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
echo "ACCEPTANCE slimsweep node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv_ws_env.sh
PY=/media02/hlhdang/uvenvs/t251cu121-py312/bin/python
export PYTHONUNBUFFERED=1
cd /media02/hlhdang/projects/pamr/src
"$PY" - <<'PYEOF'
import torch
D = "/media02/hlhdang/projects/pamr/checkpoints/vlkge/wikiart_mkg_v2"
for name in ("transe", "complex", "distmult", "struct"):
    path = (f"{D}/transe_wikiart_mkg_v2_struct_train.pt" if name == "struct"
            else f"{D}/{name}_wikiart_mkg_v2_train.pt")
    ck = torch.load(path, map_location="cpu")
    reg = (ck.get("relation_registry") or {}).get("registry_hash", "?")[:8]
    state = ck["model_state_dict"]
    ent = state["entity_embeddings.weight"].shape[0]
    d_ent = state["entity_embeddings.weight"].shape[1]
    d_rel = state["relation_embeddings.weight"].shape[1]
    print(f"ACCEPTANCE slim {name}: best_val={ck.get('best_val_mrr'):.6f} "
          f"epoch={ck.get('epoch')} scope={ck.get('embedding_scope')} "
          f"hash={reg} ents={ent} d_ent={d_ent} d_rel={d_rel} "
          f"model={ck.get('model_name')}")
PYEOF
rc=$?
echo "ACCEPTANCE slimsweep rc=$rc end=$(date -Iseconds)"
exit "$rc"
