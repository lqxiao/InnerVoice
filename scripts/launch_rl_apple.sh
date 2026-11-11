#!/bin/bash
# Apple-style (Xie et al. 2025) interleaved-reasoning RL replication on Qwen3: plain LoRA policy, <think>/<answer> tags, rule-based rewards.
# usage: MODEL=Qwen/Qwen3-1.7B TEMPLATE=interleave IR=1 ENGINE_GPUS="1" TRAIN_GPUS=0 PORT0=8041 MASTER_PORT=29681 \
#        scripts/launch_rl_apple.sh <run_name> [steps] [max_new]
set -u
NAME=$1; STEPS=${2:-300}; MAXNEW=${3:-2048}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}; TEMPLATE=${TEMPLATE:-interleave}; IR=${IR:-0}; ENGINE_GPUS=${ENGINE_GPUS:-"1"}; TRAIN_GPUS=${TRAIN_GPUS:-0}; MML=${MML:-4096}; PORT0=${PORT0:-8041}
ENGINE_ARGS=${ENGINE_ARGS:-}; MICRO_BS=${MICRO_BS:-4}; G=${G:-8}; PROMPTS=${PROMPTS:-32}; TEMP=${TEMP:-0.8}; KL=${KL:-0.001}
H=/fsx-alignment/dedicated-fsx-data-repo-alignment-us-east-1/home/liqiangx/innervoice
source ${ENV_SH:-$H/env.sh}; cd $H/think; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # ENV_SH: alternative venv (e.g. /scratch/liqiangx/env.sh on p41 whose /opt/conda differs)
LIVE=/tmp/instance_storage/liqiangx/live_$NAME; OUT=results/rl/$NAME; mkdir -p logs results/rl
FIRST_TRAIN=${TRAIN_GPUS%%,*}
CUDA_VISIBLE_DEVICES=$FIRST_TRAIN python latent/sxs_export.py --model $MODEL --plain --out $LIVE || exit 1   # untrained base as the initial policy
i=0; URLS=""
for g in $ENGINE_GPUS; do
  PORT=$((PORT0 + i)); CN=iv_engine_$g; sudo docker rm -f $CN >/dev/null 2>&1
  sudo docker run -d --name $CN --gpus "\"device=$g\"" --network host --ipc host --shm-size 16g -v /fsx-alignment:/fsx-alignment -v /tmp/instance_storage/liqiangx:/tmp/instance_storage/liqiangx \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e PYTHONUNBUFFERED=1 --entrypoint bash vllm/vllm-openai:v0.9.1 \
    -c "pip install -q --no-deps --no-build-isolation $H/think/latent/vllm_plugin >/tmp/pip.log 2>&1; python3 $H/think/latent/vllm_server.py --model $LIVE --port $PORT --max_model_len $MML --max_num_seqs 256 $ENGINE_ARGS" >/dev/null
  URLS="${URLS:+$URLS,}http://127.0.0.1:$PORT"; i=$((i + 1))
done
for g in $ENGINE_GPUS; do until sudo docker logs iv_engine_$g 2>&1 | grep -q -E "SERVER_READY|Traceback"; do sleep 10; done; sudo docker logs iv_engine_$g 2>&1 | grep -E "SERVER_READY|Traceback" | tail -1; done
NR=$(echo $TRAIN_GPUS | tr ',' '\n' | wc -l); IRFLAG=""; [ "$IR" = "1" ] && IRFLAG="--ir"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS nohup torchrun --nproc_per_node=$NR --master_port=${MASTER_PORT:-29681} latent/sxs_rl_engine.py \
  --plain --template $TEMPLATE $IRFLAG --temperature $TEMP --model $MODEL --data data_apple/train.jsonl --out $OUT --live $LIVE --engines $URLS \
  --steps $STEPS --prompts $PROMPTS --G $G --max_new $MAXNEW --micro_bs $MICRO_BS --label_bs 16 --w_route 0 --kl_beta $KL --kl_route_beta 0 --soft_overlong --w_trunc 0 \
  --eval_every 50 --eval_data data_apple/test_kk.jsonl,data_apple/test_mq.jsonl --eval_stride 1 --eval_limit 1000 --eval_samples 4 --eval_max_new $MAXNEW --no_eval0 \
  > logs/rl_$NAME.log 2>&1 &
echo "trainer launched: logs/rl_$NAME.log"
