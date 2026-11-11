#!/bin/bash
# Apple-style (Xie et al. 2025) interleaved-RL replication on Qwen3-4B, three runs in ONE 8xH200 pod (vllm/vllm-openai:v0.9.1 image):
#   Interleave+IR : engine GPU 0, trainers GPUs 1,2       Interleave : engine GPU 3, trainer GPU 4      Think-answer : engine GPU 5, trainers GPUs 6,7
# Policy = plain Qwen3-4B + LoRA (no SFT), <think>/<answer> templates, rule-based rewards (format / final / conditional time-discounted IR).
# Outputs: ${WORKSPACE_ROOT}/<run>/{log.jsonl,eval.jsonl,ckpt*}, engine + trainer logs next to them. Knobs: IV_STEPS (300) IV_MAXNEW (2048)
set -eux -o pipefail
IV=/opt/RufusGym/examples/innervoice; WS=${WORKSPACE_ROOT:-/data/apple_ws}; mkdir -p "$WS"
export HF_HOME=/data/hf; mkdir -p /data/hf; unset HF_HUB_OFFLINE; export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [A] deps ==="; python3 -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)"
pip install -q peft==0.15.2 tqdm "pyarrow<18" 2>&1 | tail -1 || pip install -q peft tqdm pyarrow 2>&1 | tail -1
cp -r "$IV/latent/vllm_plugin" /tmp/plugin && pip install -q --no-deps --no-build-isolation /tmp/plugin 2>&1 | tail -1
echo "=== [B] base model + data ==="
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B", ignore_patterns=["*.gguf", "original/*"]); print("base ok")
PY
mkdir -p "$IV/data_apple"; for f in train test_kk test_mq test_kk_hard; do gunzip -c "$IV/rl_data/apple_$f.jsonl.gz" > "$IV/data_apple/$f.jsonl"; done; wc -l "$IV"/data_apple/*.jsonl
cd "$IV"
run_one() {  # name template ir engine_gpu train_gpus port master_port
  local NAME=$1 TPL=$2 IR=$3 EG=$4 TG=$5 PORT=$6 MP=$7; local LIVE=/data/live_$NAME; local OUT="$WS/$NAME"; mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=${TG%%,*} python3 latent/sxs_export.py --model Qwen/Qwen3-4B --plain --out $LIVE 2>&1 | grep wrote
  mkdir -p /data/vllm_cache_$NAME /data/inductor_$NAME
  CUDA_VISIBLE_DEVICES=$EG VLLM_CACHE_ROOT=/data/vllm_cache_$NAME TORCHINDUCTOR_CACHE_DIR=/data/inductor_$NAME VLLM_LOGGING_LEVEL=WARNING \
    nohup python3 latent/vllm_server.py --model $LIVE --port $PORT --max_model_len 4096 --max_num_seqs 256 > "$WS/engine_$NAME.log" 2>&1 &
  for _ in $(seq 1 120); do grep -q -E "SERVER_READY|Traceback" "$WS/engine_$NAME.log" && break; sleep 10; done
  grep -q SERVER_READY "$WS/engine_$NAME.log" || { echo "engine $NAME failed"; tail -30 "$WS/engine_$NAME.log"; return 1; }
  local NR=$(echo $TG | tr ',' '\n' | wc -l); local IRFLAG=""; [ "$IR" = "1" ] && IRFLAG="--ir"
  CUDA_VISIBLE_DEVICES=$TG nohup torchrun --nproc_per_node=$NR --master_port=$MP latent/sxs_rl_engine.py \
    --plain --template $TPL $IRFLAG --temperature 0.8 --model Qwen/Qwen3-4B --data data_apple/train.jsonl --out "$OUT" --live $LIVE --engines http://127.0.0.1:$PORT \
    --steps ${IV_STEPS:-300} --prompts 32 --G 8 --max_new ${IV_MAXNEW:-2048} --micro_bs 2 --label_bs 16 --w_route 0 --kl_beta 0.001 --kl_route_beta 0 --soft_overlong --w_trunc 0 \
    --eval_every 50 --eval_data data_apple/test_kk.jsonl,data_apple/test_mq.jsonl,data_apple/test_kk_hard.jsonl --eval_stride 1 --eval_limit 1000 --eval_samples 4 --eval_max_new ${IV_MAXNEW:-2048} --no_eval0 \
    > "$WS/trainer_$NAME.log" 2>&1 &
  echo "launched $NAME"
}
run_one apple_ir_4B interleave 1 0 1,2 8041 29681; sleep 30
run_one apple_il_4B interleave 0 3 4   8051 29691; sleep 30
run_one apple_ta_4B think_answer 0 5 6,7 8061 29701
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== [C] waiting for the three trainers ==="
while pgrep -f "latent/sxs_rl_engine.py" >/dev/null; do sleep 120; for n in apple_ir_4B apple_il_4B apple_ta_4B; do grep -E "^step " "$WS/trainer_$n.log" 2>/dev/null | tail -1 | cut -c1-120; done; done
for n in apple_ir_4B apple_il_4B apple_ta_4B; do echo "== $n"; grep -E "eval |DONE|Traceback" "$WS/trainer_$n.log" | tail -8 | cut -c1-200; done
pkill -f "latent/vllm_serve[r].py" || true; echo "=== DONE ==="
