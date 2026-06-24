#!/bin/bash
# One iteration-1 cell: ETTh1 h96, channel-mixed M, 10 epochs. $1=model $2=des, rest=extra flags.
export PATH="/home/ll/miniconda3/bin:$PATH"          # cron/relaunch PATH-no-conda trap
cd /mnt/sdc/codes/Lorentz-rwkv/Time-Series-Library || exit 1
MODEL="$1"; DES="$2"; shift 2
python run.py --task_name long_term_forecast --is_training 1 --model_id "etth1_96_96_${DES}" \
  --model "$MODEL" --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --d_ff 256 --e_layers 2 --batch_size 128 --train_epochs 10 --patience 3 \
  --learning_rate 0.0003 --des "$DES" --itr 1 "$@" \
  2>&1 | tee "/mnt/sdc/codes/Lorentz-rwkv/ts_decompose23d/.remember/tsv_${DES}.log"
