#!/bin/bash
# InnerVoice RL (Scenario B) on one Greenland 8xH200 node, inside the vllm/vllm-openai:v0.9.1 image.
#   GPUs 4-7: four vLLM rollout engines (latent/vllm_server.py, one per GPU, hot weight reload)
#   GPUs 0-3: torchrun trainer (latent/sxs_rl_engine.py), grads all-reduced
# Inputs come from public sources (no FSx dependency): HF liqiangx/InnerVoice-Qwen3-4B (merged weights for the engine,
# research_ckpt/sxs.pt for the trainer), HF Qwen/Qwen3-4B (base), HF MathArena/aime_2025 (eval), rl_data/rl_pool.jsonl.gz (overlay).
# Outputs: ${WORKSPACE_ROOT}/rl/{log.jsonl,eval.jsonl,ckpt_step*} on FSx (mirrored to S3), engine logs next to them.
# Knobs (env): IV_ENGINE_ARGS (e.g. --enforce_eager) IV_STEPS (480) IV_PROMPTS (32) IV_G (16) IV_MAXNEW (12000) IV_WFCO (0.05) IV_WUNSUP (0.5) IV_WSHORT (0.05) IV_EVAL_EVERY (50)
set -eux
IV=/opt/RufusGym/examples/innervoice
WS=${WORKSPACE_ROOT:-/data/innervoice_ws}; mkdir -p "$WS/rl"
export HF_HOME=/data/hf; mkdir -p /data/hf; unset HF_HUB_OFFLINE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [A] python deps (image: vllm 0.9.1 / torch 2.7 / transformers 4.52) ==="
python3 -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
pip install -q peft==0.15.2 tqdm "pyarrow<18" 2>&1 | tail -1 || pip install -q peft tqdm pyarrow 2>&1 | tail -1
pip install -q --no-deps --no-build-isolation "$IV/latent/vllm_plugin" 2>&1 | tail -1
python3 -c "import peft, innervoice_vllm; print('peft', peft.__version__, 'plugin ok')"

echo "=== [B] inputs from the Hub ==="
LIVE=/data/live
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("liqiangx/InnerVoice-Qwen3-4B", local_dir="/data/live")
snapshot_download("Qwen/Qwen3-4B", ignore_patterns=["*.gguf", "original/*"])          # base weights for the trainer (HF cache)
snapshot_download("MathArena/aime_2025", repo_type="dataset", local_dir="/data/aime25")
print("downloads ok")
PY
ls -la $LIVE $LIVE/research_ckpt /data/aime25/data
gunzip -kf "$IV/rl_data/rl_pool.jsonl.gz"; wc -l "$IV/rl_data/rl_pool.jsonl"
cp "$IV/rl_data/rl_pool.jsonl" "$WS/rl_pool.jsonl"

echo "=== [C] rollout engines on GPUs 4-7 (sequential start; isolated compile caches; eager fallback) ==="
cd "$IV"
URLS=""
start_engine() {  # i extra_args
  local i=$1; shift; local g=$((4 + i)); local port=$((8001 + i))
  mkdir -p /data/vllm_cache_$i /data/inductor_$i; : > "$WS/engine_$i.log"
  CUDA_VISIBLE_DEVICES=$g VLLM_CACHE_ROOT=/data/vllm_cache_$i TORCHINDUCTOR_CACHE_DIR=/data/inductor_$i VLLM_LOGGING_LEVEL=WARNING \
    nohup python3 latent/vllm_server.py --model $LIVE --port $port --max_model_len 13000 --max_num_seqs 160 ${IV_ENGINE_ARGS:-} "$@" >> "$WS/engine_$i.log" 2>&1 &
  for _ in $(seq 1 120); do grep -q -E "SERVER_READY|Traceback" "$WS/engine_$i.log" && break; sleep 10; done
  grep -q SERVER_READY "$WS/engine_$i.log"
}
for i in 0 1 2 3; do
  if ! start_engine $i; then
    echo "engine $i failed to start (compiled); retrying with --enforce_eager"; grep -n -i -m 5 rror "$WS/engine_$i.log" || true
    pkill -f "vllm_server.py --model $LIVE --port $((8001 + i))" || true; sleep 5
    start_engine $i --enforce_eager || { echo "engine $i failed twice"; tail -40 "$WS/engine_$i.log"; exit 1; }
  fi
  URLS="${URLS:+$URLS,}http://127.0.0.1:$((8001 + i))"; echo "engine $i ready"
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== [D] trainer on GPUs 0-3 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29621 latent/sxs_rl_engine.py \
  --model Qwen/Qwen3-4B --ckpt $LIVE/research_ckpt --data "$WS/rl_pool.jsonl" --out "$WS/rl" --live $LIVE --engines "$URLS" \
  --steps ${IV_STEPS:-480} --prompts ${IV_PROMPTS:-32} --G ${IV_G:-16} --max_new ${IV_MAXNEW:-12000} --micro_bs 2 --label_bs 8 \
  --w_fco ${IV_WFCO:-0.05} --fco_ref 600 --w_unsup ${IV_WUNSUP:-0.5} --w_short ${IV_WSHORT:-0.05} \
  --eval_every ${IV_EVAL_EVERY:-50} --aime_data /data/aime25/data/train-00000-of-00001.parquet --aime_samples 4 --aime_max_new 12000 \
  2>&1 | tee "$WS/trainer.log"
echo "=== DONE ==="; pkill -f "latent/vllm_serve[r].py" || true
