#!/bin/bash
# InnerVoice RL on K&K + MuSiQue (Apple interleaved-RL replication) with engine rollouts. Run ON the GPU host.
# usage: MODEL=Qwen/Qwen3-1.7B ENGINE_GPUS="5 6" TRAIN_GPUS=3,4 scripts/launch_rl_engine_kk.sh <sft_ckpt_dir> <run_name> [steps] [max_new]
set -u
CKPT=$1; NAME=$2; STEPS=${3:-300}; MAXNEW=${4:-1024}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}; ENGINE_GPUS=${ENGINE_GPUS:-"4 5 6 7"}; TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}; MML=${MML:-4096}; PORT0=${PORT0:-8001}; ENGINE_ARGS=${ENGINE_ARGS:-}; MICRO_BS=${MICRO_BS:-4}
EXTRA=${IV_EXTRA:-"--w_inter 0.5 --w_prem 1.0 --w_fmt 0.2 --w_fco 0 --w_trunc 0.5 --soft_overlong --kl_beta 0.02 --kl_route_beta 0.5 --w_short_pen 0.3 --w_route 0.2 --route_sample"}
H=/fsx-alignment/dedicated-fsx-data-repo-alignment-us-east-1/home/liqiangx/innervoice
source ${ENV_SH:-$H/env.sh}; cd $H/think; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # ENV_SH: alternative venv (e.g. /scratch/liqiangx/env.sh on p41 whose /opt/conda differs)
LIVE=/tmp/instance_storage/liqiangx/live_$NAME; OUT=results/rl/$NAME; mkdir -p logs results/rl
FIRST_TRAIN=${TRAIN_GPUS%%,*}
CUDA_VISIBLE_DEVICES=$FIRST_TRAIN python latent/sxs_export.py --model $MODEL --ckpt $CKPT --out $LIVE --route_sample || exit 1
# engines (ports PORT0, PORT0+1, ...)
i=0; URLS=""
for g in $ENGINE_GPUS; do
  PORT=$((PORT0 + i)); CN=iv_engine_$g; sudo docker rm -f $CN >/dev/null 2>&1
  sudo docker run -d --name $CN --gpus "\"device=$g\"" --network host --ipc host --shm-size 16g -v /fsx-alignment:/fsx-alignment -v /tmp/instance_storage/liqiangx:/tmp/instance_storage/liqiangx \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e PYTHONUNBUFFERED=1 --entrypoint bash vllm/vllm-openai:v0.9.1 \
    -c "cp -r $H/think/latent/vllm_plugin /tmp/plugin && pip install -q --no-deps --no-build-isolation /tmp/plugin >/tmp/pip.log 2>&1; python3 $H/think/latent/vllm_server.py --model $LIVE --port $PORT --max_model_len $MML --max_num_seqs 256 $ENGINE_ARGS" >/dev/null
  URLS="${URLS:+$URLS,}http://127.0.0.1:$PORT"; i=$((i + 1))
done
for g in $ENGINE_GPUS; do until sudo docker logs iv_engine_$g 2>&1 | grep -q -E "SERVER_READY|Traceback"; do sleep 10; done; sudo docker logs iv_engine_$g 2>&1 | grep -E "SERVER_READY|Traceback" | tail -1; done
NR=$(echo $TRAIN_GPUS | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS nohup torchrun --nproc_per_node=$NR --master_port=${MASTER_PORT:-29631} latent/sxs_rl_engine.py \
  --model $MODEL --ckpt $CKPT --data data_sxs/sxs_train.jsonl --out $OUT --live $LIVE --engines $URLS \
  --steps $STEPS --prompts 32 --G 8 --max_new $MAXNEW --micro_bs $MICRO_BS --label_bs 16 \
  --eval_every 50 --eval_data data_sxs/sxs_test_kk.jsonl,data_sxs/sxs_test_mq.jsonl --eval_stride 4 --eval_limit 150 --eval_samples 4 --eval_max_new $MAXNEW $EXTRA \
  > logs/rl_$NAME.log 2>&1 &
echo "trainer launched: logs/rl_$NAME.log"
