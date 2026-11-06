#!/bin/bash
# usage: phase1_sweep.sh <model> <tag> <steps> <seed> <gpu list as "0 2 3 4 5 6 7">
# Runs: lora_only, K0 (static soft tokens), K1, K2, K4, K8, K4+codi  -- one per GPU
source /scratch/liqiangx/env.sh
MODEL=$1; TAG=$2; STEPS=$3; SEED=$4; shift 4; GPUS=($@)
BS=${BS:-32}; ACCUM=${ACCUM:-1}; ONLY=${ONLY:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUNS=("lora_only --n_slots 0 --K 0" "K0 --K 0 --eval_K 1,2,4,8" "K1 --K 1 --eval_K 0,2,4,8" "K2 --K 2 --eval_K 0,1,4,8" "K4 --K 4 --eval_K 0,1,2,8" "K8 --K 8 --eval_K 0,1,2,4" "K4codi --K 4 --distill codi --eval_K 0,1,2,8")
mkdir -p results/$TAG logs
for i in "${!RUNS[@]}"; do
  set -- ${RUNS[$i]}; NAME=$1; shift
  GPU=${GPUS[$i]}
  [ -z "$GPU" ] && { echo "no gpu for $NAME"; continue; }
  [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $NAME "* ]] && continue
  CUDA_VISIBLE_DEVICES=$GPU nohup python -u latent/train_fixed_k.py --model $MODEL --out results/$TAG/${NAME}_s$SEED --steps $STEPS --seed $SEED --bs $BS --grad_accum $ACCUM --save "$@" > logs/${TAG}_${NAME}_s$SEED.log 2>&1 &
  echo "launched $NAME on gpu $GPU"
done
