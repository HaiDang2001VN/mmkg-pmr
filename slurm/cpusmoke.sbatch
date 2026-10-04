#!/bin/bash
#SBATCH -p batch
#SBATCH -J pamr-cpusmoke
#SBATCH -N1
#SBATCH -o /media02/hlhdang/projects/pamr/reports/cpusmoke_%A.out
#SBATCH -e /media02/hlhdang/projects/pamr/reports/cpusmoke_%A.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# CPU machinery smoke: validates the FULL downstream chain (score.py ->
# evaluate.py; train_pamr end-to-end 1 epoch) against the regenerated Phase
# B artifacts using a SYNTHETIC (random-init, contract-valid) TransE
# checkpoint. Numbers are meaningless; code paths are the target. The
# prod artifact dirs stay pristine: everything writes into a
# cpu_smoke/ scratch subtree.
echo "ACCEPTANCE cpusmoke node=$(hostname) start=$(date -Iseconds)"
source /media02/hlhdang/tools/uvenv.sh
export UV_PROJECT_ENVIRONMENT=/media02/hlhdang/uvenvs/main-py312
cd /media02/hlhdang/projects/pamr/src

run() {
  local name="$1"; shift
  echo ""
  echo "════════ STEP $name ════════ start=$(date -Iseconds)"
  "$@"
  local rc=$?
  echo "ACCEPTANCE step=$name rc=$rc end=$(date -Iseconds)"
  if [ "$rc" -ne 0 ]; then
    echo "ACCEPTANCE cpusmoke FAILED at step=$name rc=$rc"
    exit "$rc"
  fi
}
uvpython() { "$UV_BIN" run --project . --no-sync python "$@"; }

CKPT=/media02/hlhdang/projects/pamr/checkpoints/vlkge/wikiart_mkg_v2/cpu_smoke_synth_transe.pt
SMOKE_BASE=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/cpu_smoke
RUNDIR=/media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/full/pamr

mkdir -p "$(dirname "$CKPT")" "$SMOKE_BASE/pamr"
cp "$RUNDIR"/train_paths.csv "$RUNDIR"/val_paths.csv "$RUNDIR"/test_paths.csv "$SMOKE_BASE/pamr/" 2>/dev/null
cp "$RUNDIR"/*_paths_meta.json "$SMOKE_BASE/pamr/"
cp "$RUNDIR"/entity_mapping.csv "$RUNDIR"/relation_mapping.csv "$SMOKE_BASE/pamr/" 2>/dev/null
# prepared queries (for evaluate.py + the PAMR queries_dir contract):
cp /media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/{train,val,test}_queries.csv "$SMOKE_BASE/pamr/"
# evaluate proxy expects queries at the smoke ROOT (scores land there too)
cp /media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/{train,val,test}_queries.csv "$SMOKE_BASE/"

# ── 1. Synthesize the contract-valid checkpoint ──
run synth uvpython scripts/throwaway_synth_cpu_ckpt.py --out "$CKPT" \
  --entity-mapping "$RUNDIR/entity_mapping.csv" \
  --manifest "$RUNDIR/relation_manifest.json" \
  --config pamr/configs/wikiart_mkg_v2/transe_clip.yaml

# ── 2. score.py machinery on CPU (scores CSVs land beside the queries) ──
sed -e "s|checkpoint_path:.*|checkpoint_path: ../checkpoints/vlkge/wikiart_mkg_v2/cpu_smoke_synth_transe.pt|" \
    pamr/configs/eval/wikiart_mkg_v2_score.yaml > "$SMOKE_BASE/score.yaml"
printf 'output_dir: ../checkpoints/preproc/wikiart_mkg_v2/cpu_smoke\n' >> "$SMOKE_BASE/score.yaml"
run score uvpython -m pamr.models.vlkge.score --config "$SMOKE_BASE/score.yaml" --no-cuda

# ── 3. evaluate.py machinery over the score CSVs ──
# evaluate.py dereferences prepare_config's output_dir + exclude list, so
# give it a PROXY prepare config whose output_dir points at the smoke dir.
sed -e "s|^output_dir:.*|output_dir: ../checkpoints/preproc/wikiart_mkg_v2/cpu_smoke|" \
    pamr/configs/eval/wikiart_mkg_v2_neg_sampling.yaml > "$SMOKE_BASE/prepare_proxy.yaml"
cat > "$SMOKE_BASE/eval.yaml" <<EOF
prepare_config: /media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/cpu_smoke/prepare_proxy.yaml
scores_dir: /media02/hlhdang/projects/pamr/checkpoints/preproc/wikiart_mkg_v2/cpu_smoke
top_k: [1, 3, 5, 10]

EOF
run evaluate uvpython -m pamr.scripts.evaluate --config "$SMOKE_BASE/eval.yaml"

# ── 4. train_pamr end-to-end machinery: 1 epoch on CPU over real paths ──
sed -e "s|vlkge_config:.*|vlkge_config: /media02/hlhdang/projects/pamr/src/pamr/configs/wikiart_mkg_v2/transe_clip.yaml|" \
    -e "s|vlkge_checkpoint:.*|vlkge_checkpoint: ../checkpoints/vlkge/wikiart_mkg_v2/cpu_smoke_synth_transe.pt|" \
    -e "s|paths_dir:.*|paths_dir: ../checkpoints/preproc/wikiart_mkg_v2/cpu_smoke/pamr/|" \
    -e "s|queries_dir:.*|queries_dir: ../checkpoints/preproc/wikiart_mkg_v2/cpu_smoke/pamr/|" \
    -e "s|^epochs:.*|epochs: 1|" \
    pamr/configs/pr/smoke_pamr.yaml > "$SMOKE_BASE/smoke_cpu.yaml"
run train-pamr uvpython -m pamr.scripts.train_pamr --config "$SMOKE_BASE/smoke_cpu.yaml" --workers 0

echo ""
echo "ACCEPTANCE cpusmoke node=$(hostname) end=$(date -Iseconds) ALL-STEPS-DONE"
