#!/bin/bash
# Iteration-1 cells with a concurrency cap (default 5) on the single GPU (49GB; each cell ~1-2GB).
export PATH="/home/ll/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAXJOBS="${1:-5}"
R=/mnt/sdc/codes/Lorentz-rwkv/ts_decompose23d/run_tsvideo_cell.sh

CELLS=(
  "DLinear dlinear"
  "VideoRWKVJEPA jepa_raw --vr_encoder raw"
  "VideoRWKVJEPA jepa_splat --vr_encoder splat"
  "VideoRWKVJEPA jepa_gram --vr_encoder gram"
  "VideoRWKVJEPA jepa_gaf --vr_encoder gaf"
  "VideoRWKVJEPA jepa_recur --vr_encoder recur"
  "VideoRWKVJEPA jepa_lag --vr_encoder lag"
  "VideoRWKVJEPA jepa_fused --vr_encoder fused"
  "VideoRWKVJEPA jepa_fused_fc0 --vr_encoder fused --vr_jepa_weight 0"
  "VideoRWKVJEPA jepa_raw_linres --vr_encoder raw --vr_linear_residual 1"
  "VideoRWKVJEPA jepa_fused_linres --vr_encoder fused --vr_linear_residual 1"
)

echo "[parallel] $((${#CELLS[@]})) cells, max $MAXJOBS concurrent  $(date +%T)"
for c in "${CELLS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "[parallel] launch: $c"
  bash $R $c >/dev/null 2>&1 &
  sleep 2
done
wait
echo "[parallel] ALL DONE $(date +%T)"
echo "============ ITER-1 RESULTS ============"
for des in dlinear jepa_raw jepa_splat jepa_gram jepa_gaf jepa_recur jepa_lag jepa_fused jepa_fused_fc0 jepa_raw_linres jepa_fused_linres; do
  printf "%-20s " "$des"; grep "^mse:" /mnt/sdc/codes/Lorentz-rwkv/ts_decompose23d/.remember/tsv_${des}.log 2>/dev/null | tail -1 || echo "(no result)"
done
