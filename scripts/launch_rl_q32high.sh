#!/bin/bash
# RL round 2 on the Qwen3-32B high-effort InnerVoice 1.7B: correctness + small gated FCO term, overlong masking, long rollouts.
# usage: launch_rl_q32high.sh <max_new> <gpu> [steps]
source /scratch/liqiangx/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAXNEW=$1; GPU=$2; STEPS=${3:-60}; N=rl_q32high_1.7B_m$MAXNEW
mkdir -p results/rl/$N results/sxs_eval/$N/aime25
CUDA_VISIBLE_DEVICES=$GPU nohup bash -c "python -u latent/sxs_rl.py --ckpt results/sxs/q32high_sxs_1.7B/ckpt --data data_math/self/sxs_q32high_train.jsonl --out results/rl/$N --steps $STEPS --prompts 4 --G 8 --max_new $MAXNEW --micro_bs 1 --grad_ckpt --mask_truncated --w_fco 0.1 --fco_ref 600 --save_every 10 > logs/$N.log 2>&1; [ -f results/rl/$N/ckpt/sxs.pt ] && python -u latent/sxs_eval_math.py --ckpt results/rl/$N/ckpt --out results/sxs_eval/$N/aime25 --samples 4 --max_new_tokens 12000 --bs 15 > logs/aime_$N.log 2>&1; echo RLQ_DONE >> logs/$N.log" > /dev/null 2>&1 &
echo "launched $N on gpu $GPU"
