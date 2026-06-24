#!/bin/bash
# Autoresearch round runner: train.py cells, $1 concurrent (default 5). Pass cells as "des|args".
export PATH="/home/ll/miniconda3/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/sdc/codes/Lorentz-rwkv/ts_decompose23d
MAX="${1:-5}"; shift
EPOCHS="${EPOCHS:-10}"
CELLS=("$@")
echo "[round] ${#CELLS[@]} cells, max $MAX concurrent, $EPOCHS ep  $(date +%T)"
for entry in "${CELLS[@]}"; do
  des="${entry%%|*}"; args="${entry#*|}"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do sleep 5; done
  echo "[round] launch $des : $args"
  python train.py $args --des "$des" --epochs "$EPOCHS" > ".remember/${des}.log" 2>&1 &
  sleep 2
done
wait
echo "[round] DONE $(date +%T)"
echo "=== results (sorted by mse) ==="
grep -E "^r[0-9]" autoresearch_results.tsv 2>/dev/null | sort -t$'\t' -k3 -g | awk -F'\t' '{printf "%-22s mse=%s mae=%s\n",$1,$3,$4}'
