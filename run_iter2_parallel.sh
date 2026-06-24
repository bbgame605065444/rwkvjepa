#!/bin/bash
# Iteration-2: CometVideoJEPA variants on ETTh1 h96 (10 ep), concurrency cap (default 5) on the GPU.
export PATH="/home/ll/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAXJOBS="${1:-5}"
R=/mnt/sdc/codes/Lorentz-rwkv/ts_decompose23d/run_tsvideo_cell.sh

CELLS=(
  "CometVideoJEPA cvjepa --cm_video lag --cm_motifs 10"
  "CometVideoJEPA cvjepa_nojepa --cm_video lag --cm_motifs 10 --vr_jepa_weight 0"
  "CometVideoJEPA cvjepa_k1 --cm_video lag --cm_motifs 1"
  "CometVideoJEPA cvjepa_raw --cm_video raw --cm_motifs 10"
  "CometVideoJEPA cvjepa_linres --cm_video lag --cm_motifs 10 --vr_linear_residual 1"
  "CometVideoJEPA cvjepa_k20 --cm_video lag --cm_motifs 20"
)

echo "[iter2] ${#CELLS[@]} cells, max $MAXJOBS concurrent  $(date +%T)"
for c in "${CELLS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[iter2] launch: $c"
  bash $R $c >/dev/null 2>&1 &
  sleep 2
done
wait
echo "[iter2] ALL DONE $(date +%T)"
echo "============ ITER-2 RESULTS (CometVideoJEPA) ============"
echo "controls: dlinear 0.3986 | jepa_lag 0.4295 | CometNet-paper 0.345"
for des in cvjepa cvjepa_nojepa cvjepa_k1 cvjepa_raw cvjepa_linres cvjepa_k20; do
  printf "%-16s " "$des"; grep "^mse:" /mnt/sdc/codes/Lorentz-rwkv/ts_decompose23d/.remember/tsv_${des}.log 2>/dev/null | tail -1 || echo "(no result)"
done
