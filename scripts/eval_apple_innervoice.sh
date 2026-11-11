#!/bin/bash
# Evaluate InnerVoice checkpoints on the Apple test subsets (InnerVoice prompt format) through a local engine. Run ON a GPU host.
# usage: GPU=5 PORT=8081 scripts/eval_apple_innervoice.sh  (edit SPECS below)
set -u
H=/fsx-alignment/dedicated-fsx-data-repo-alignment-us-east-1/home/liqiangx/innervoice
source ${ENV_SH:-$H/env.sh}; cd $H/think; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-5}; PORT=${PORT:-8081}; SAMPLES=${SAMPLES:-4}
SPECS=${SPECS:-"Qwen/Qwen3-1.7B|results/rl/kk_rl_1.7B_v5/ckpt_step100|iv_rl_1.7B Qwen/Qwen3-4B|results/sxs/kk_sxs_4B_noethink/ckpt|iv_sft_4B Qwen/Qwen3-4B|results/rl/kk_rl_4B_v5/ckpt|iv_rl_4B"}
mkdir -p results/rl/apple logs
for spec in $SPECS; do
  M=${spec%%|*}; rest=${spec#*|}; CK=${rest%%|*}; TAG=${rest#*|}; EXP=/tmp/instance_storage/liqiangx/live_eval_$TAG
  echo "=== $TAG ($M, $CK)"
  CUDA_VISIBLE_DEVICES=$GPU python latent/sxs_export.py --model $M --ckpt $CK --out $EXP --route_sample 2>&1 | grep -E "wrote|Error" | tail -2
  sudo -n docker rm -f iv_engine_$GPU >/dev/null 2>&1; sleep 5
  sudo -n docker run -d --name iv_engine_$GPU --gpus "\"device=$GPU\"" --network host --ipc host --shm-size 16g -v /fsx-alignment:/fsx-alignment -v /tmp/instance_storage/liqiangx:/tmp/instance_storage/liqiangx \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=WARNING -e PYTHONUNBUFFERED=1 --entrypoint bash vllm/vllm-openai:v0.9.1 \
    -c "cp -r $H/think/latent/vllm_plugin /tmp/plugin && pip install -q --no-deps --no-build-isolation /tmp/plugin >/tmp/pip.log 2>&1; python3 $H/think/latent/vllm_server.py --model $EXP --port $PORT --max_model_len 4096 --max_num_seqs 128 --gpu_mem ${GPU_MEM:-0.3}" >/dev/null
  for i in $(seq 1 60); do sudo -n docker logs iv_engine_$GPU 2>&1 | grep -q -E "SERVER_READY|Traceback" && break; sleep 10; done
  sudo -n docker logs iv_engine_$GPU 2>&1 | grep -v "retrieving safetensors" | grep -E "SERVER_READY|Error" | tail -2 | cut -c1-200
  CUDA_VISIBLE_DEVICES=$GPU python latent/eval_engine_kk.py --model $M --ckpt $CK --engines http://127.0.0.1:$PORT --data data_apple/test_kk_ivprompt.jsonl,data_apple/test_mq_ivprompt.jsonl --stride 1 --limit 1000 --samples $SAMPLES --max_new 2048 --out results/rl/apple/$TAG.json --route_sample > logs/eval_apple_$TAG.log 2>&1
  grep -E "^test_|Error" logs/eval_apple_$TAG.log | cut -c1-250
done
sudo -n docker rm -f iv_engine_$GPU >/dev/null 2>&1; echo iv_evals_done
