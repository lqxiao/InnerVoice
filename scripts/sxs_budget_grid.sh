#!/bin/bash
# Budget / policy grid (beyond the paper): nothink, budget-32/64/128, fixed-64, stitch-64 ; greedy ; 150 examples.
source /scratch/liqiangx/env.sh
MODEL=$1; CKPT=$2; OUT=$3; GPU=$4; shift 4; EXTRA="$@"
export CUDA_VISIBLE_DEVICES=$GPU; mkdir -p $OUT
for DATA in data_sxs/sxs_test_kk.jsonl data_sxs/sxs_test_mq.jsonl; do
  TAG=$(basename $DATA .jsonl)
  for POL in nothink budget-32 budget-64 budget-128 fixed-64 stitch-64; do
    python -u latent/sxs_eval.py --model $MODEL --ckpt $CKPT --data $DATA --out $OUT/$TAG --policy $POL --stride 4 --limit 150 --think_cost 0.03 $EXTRA > $OUT/log_${TAG}_${POL}.log 2>&1
  done
done
echo GRID_DONE > $OUT/GRID_DONE
