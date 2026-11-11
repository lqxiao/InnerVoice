#!/bin/bash
# Launch one InnerVoice vLLM rollout server per GPU (docker, image vllm/vllm-openai:v0.9.1), ports 8001+i.
# usage: scripts/launch_engines.sh <model_dir> "<gpu list, e.g. 4 5 6 7>" [max_model_len] [extra server args]
set -u
MODEL=$1; GPUS=$2; MML=${3:-13000}; EXTRA=${4:-}
H=/fsx-alignment/dedicated-fsx-data-repo-alignment-us-east-1/home/liqiangx/innervoice
THINK=$H/think
i=0
for g in $GPUS; do
  PORT=$((8001 + i)); NAME=iv_engine_$g
  sudo docker rm -f $NAME >/dev/null 2>&1
  sudo docker run -d --name $NAME --gpus "\"device=$g\"" --network host --ipc host --shm-size 16g \
    -v /fsx-alignment:/fsx-alignment -v /tmp/instance_storage/liqiangx:/tmp/instance_storage/liqiangx \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e PYTHONUNBUFFERED=1 \
    --entrypoint bash vllm/vllm-openai:v0.9.1 -c "cp -r $THINK/latent/vllm_plugin /tmp/plugin && pip install -q --no-deps --no-build-isolation /tmp/plugin >/tmp/pip.log 2>&1; python3 $THINK/latent/vllm_server.py --model $MODEL --port $PORT --max_model_len $MML $EXTRA"
  echo "$NAME -> port $PORT"
  i=$((i + 1))
done
