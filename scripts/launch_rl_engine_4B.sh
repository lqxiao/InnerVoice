#!/bin/bash
# InnerVoice RL, Scenario B, 4B on one 8-GPU host (run ON the GPU host): 4 vLLM rollout engines (GPUs 4-7) + 4 trainer ranks (GPUs 0-3).
# usage: scripts/launch_rl_engine_4B.sh <sft_ckpt_dir> <run_name> [steps] [max_new]
set -u
CKPT=$1; NAME=$2; STEPS=${3:-300}; MAXNEW=${4:-8000}
H=/fsx-alignment/dedicated-fsx-data-repo-alignment-us-east-1/home/liqiangx/innervoice
source $H/env.sh
LIVE=/tmp/instance_storage/liqiangx/live_$NAME; OUT=results/rl/$NAME; mkdir -p logs results/rl
# 1) merged export served by the engines (rewritten every RL step by trainer rank 0)
CUDA_VISIBLE_DEVICES=0 python latent/sxs_export.py --model Qwen/Qwen3-4B --ckpt $CKPT --out $LIVE || exit 1
# 2) engines
bash scripts/launch_engines.sh $LIVE "4 5 6 7" 13000
for g in 4 5 6 7; do
  until sudo docker logs iv_engine_$g 2>&1 | grep -q -E "SERVER_READY|Traceback"; do sleep 10; done
  sudo docker logs iv_engine_$g 2>&1 | grep -E "SERVER_READY|Traceback" | tail -1
done
# 3) trainer
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node=4 --master_port=29621 latent/sxs_rl_engine.py \
  --model Qwen/Qwen3-4B --ckpt $CKPT --data data_math/self/sxs_q32high_train.jsonl --out $OUT --live $LIVE \
  --engines http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004 \
  --steps $STEPS --prompts 32 --G 8 --max_new $MAXNEW --micro_bs 1 --label_bs 4 \
  --w_fco 0.1 --fco_ref 600 --eval_every 50 --aime_data $H/extdata/aime25/data/train-00000-of-00001.parquet --aime_samples 4 --aime_max_new 12000 \
  > logs/rl_$NAME.log 2>&1 &
echo "trainer launched: logs/rl_$NAME.log"
