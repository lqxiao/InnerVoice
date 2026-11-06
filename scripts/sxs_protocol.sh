#!/bin/bash
# SxS-paper protocol for one checkpoint: router policy, greedy + Average@4 (temperature 1.0), both test sets, 150 examples (stride 4).
# usage: sxs_protocol.sh <model> <ckpt> <out> <gpu> [extra args]
source /scratch/liqiangx/env.sh
MODEL=$1; CKPT=$2; OUT=$3; GPU=$4; shift 4; EXTRA="$@"
export CUDA_VISIBLE_DEVICES=$GPU; mkdir -p $OUT
for DATA in data_sxs/sxs_test_kk.jsonl data_sxs/sxs_test_mq.jsonl; do
  TAG=$(basename $DATA .jsonl)
  python -u latent/sxs_eval.py --model $MODEL --ckpt $CKPT --data $DATA --out $OUT/$TAG --policy gate --stride 4 --limit 150 --think_cost 0.03 $EXTRA > $OUT/log_${TAG}_gate.log 2>&1
  python -u latent/sxs_eval.py --model $MODEL --ckpt $CKPT --data $DATA --out $OUT/$TAG --policy gate --stride 4 --limit 150 --think_cost 0.03 --samples 4 $EXTRA > $OUT/log_${TAG}_gate_avg4.log 2>&1
done
echo PROTOCOL_DONE > $OUT/PROTOCOL_DONE
