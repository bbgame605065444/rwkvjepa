#!/bin/bash
# Gap-attribution ablations (run after exp2). 10 epochs each, ETTh1 h96.
set -u
cd /mnt/sdc/codes/Lorentz-rwkv/Time-Series-Library
LOG=/mnt/sdc/codes/Lorentz-rwkv/ts_decompose23d/.remember

# wait for exp2 (VideoRWKVJEPA, model_id contains vrwkvjepa) to release the GPU
while pgrep -f "run.py.*vrwkvjepa" >/dev/null 2>&1; do sleep 10; done
echo "[abl] exp2 done; starting gap ablations $(date +%T)"

COMMON="--task_name long_term_forecast --is_training 1 --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M \
  --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --d_ff 256 --e_layers 2 --batch_size 128 \
  --train_epochs 10 --patience 3 --learning_rate 0.0003 --itr 1"

echo "[abl] A0: VideoRWKV vr_mae_weight=0 (isolate aux loss)"
python run.py $COMMON --model VideoRWKV --model_id abl_a0 --vr_mae_weight 0 --des abl_a0 > $LOG/abl_a0.log 2>&1

echo "[abl] A1: VideoRWKV vr_linear_residual=1 (the diagnosed fix)"
python run.py $COMMON --model VideoRWKV --model_id abl_a1 --vr_linear_residual 1 --des abl_a1 > $LOG/abl_a1.log 2>&1

echo "[abl] A2: VideoRWKV linres + vr_mae_weight=0.1"
python run.py $COMMON --model VideoRWKV --model_id abl_a2 --vr_linear_residual 1 --vr_mae_weight 0.1 --des abl_a2 > $LOG/abl_a2.log 2>&1

echo "[abl] C: DLinear control (same config)"
python run.py $COMMON --model DLinear --model_id abl_c_dlinear --des abl_c > $LOG/abl_c_dlinear.log 2>&1

echo "============ GAP ABLATION RESULTS ($(date +%T)) ============"
echo "baseline VideoRWKV-MAE (15ep): mse 0.586 / mae 0.512   |   NLinear closed-form: 0.389"
for f in a0 a1 a2 c_dlinear; do printf "%-10s " "$f"; grep "^mse:" $LOG/abl_$f.log | tail -1; done
