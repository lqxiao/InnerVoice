#!/bin/bash
# usage: sxs_eval_grid.sh <model> <ckpt_dir> <out_dir> <gpu> [limit] [extra args e.g. --no_expert_ft]
# Policies: gate, nothink, budget-32/64/128, fixed-32/64, stitch-64 ; think_cost measured (0) and simulated 0.03 s/token.
source /scratch/liqiangx/env.sh
MODEL=$1; CKPT=$2; OUT=$3; GPU=$4; LIMIT=${5:-0}; shift 5; EXTRA="$@"
export CUDA_VISIBLE_DEVICES=$GPU
for DATA in data_sxs/sxs_test_kk.jsonl data_sxs/sxs_test_mq.jsonl; do
  TAG=$(basename $DATA .jsonl)
  for POL in gate nothink budget-32 budget-64 budget-128 fixed-32 fixed-64 stitch-64; do
    python -u latent/sxs_eval.py --model $MODEL --ckpt $CKPT --data $DATA --out $OUT/$TAG --policy $POL --limit $LIMIT --think_cost 0.03 $EXTRA > $OUT/log_${TAG}_${POL}.log 2>&1
  done
  python -u latent/sxs_eval.py --model $MODEL --ckpt $CKPT --data $DATA --out $OUT/$TAG --policy gate --limit $LIMIT --think_cost 0 $EXTRA > $OUT/log_${TAG}_gate_measured.log 2>&1
done
echo GRID_DONE
