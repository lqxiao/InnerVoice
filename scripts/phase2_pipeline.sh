#!/bin/bash
# usage: phase2_pipeline.sh <model> <tag> <gpu> [stageA_steps] [stageB_steps]
# stage A (random k) -> counterfactual labels -> stage B gate (buffer-aware) & baseline-6 gate (no buffer) -> streaming evals
source /scratch/liqiangx/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$1; TAG=$2; GPU=$3; SA=${4:-1200}; SB=${5:-800}
OUT=results/$TAG; mkdir -p $OUT logs
export CUDA_VISIBLE_DEVICES=$GPU
set -x
python -u latent/train_chunked.py --model $MODEL --mode random --out $OUT/stageA --steps $SA --bs 16 > logs/${TAG}_stageA.log 2>&1 || exit 1
python -u latent/label_counterfactual.py --model $MODEL --ckpt $OUT/stageA/ckpt --split chunked_train --out $OUT/cf_train.json --bs 48 > logs/${TAG}_cf_train.log 2>&1 || exit 1
python -u latent/label_counterfactual.py --model $MODEL --ckpt $OUT/stageA/ckpt --split chunked_test --out $OUT/cf_test.json --bs 48 > logs/${TAG}_cf_test.log 2>&1 || exit 1
python -u latent/train_chunked.py --model $MODEL --mode labels --init $OUT/stageA/ckpt --cf_labels $OUT/cf_train.json --out $OUT/gate_buf --steps $SB --bs 16 > logs/${TAG}_gate_buf.log 2>&1 || exit 1
python -u latent/train_chunked.py --model $MODEL --mode labels --init $OUT/stageA/ckpt --cf_labels $OUT/cf_train.json --out $OUT/gate_nobuf --no_buffer --steps $SB --bs 16 > logs/${TAG}_gate_nobuf.log 2>&1 || exit 1
for SC in 0.1 0.3; do
  python -u latent/eval_streaming.py --model $MODEL --ckpt $OUT/gate_buf/ckpt --gate --policy gate --step_cost $SC --out $OUT/eval --cf_labels $OUT/cf_test.json > logs/${TAG}_eval_gate_buf_$SC.log 2>&1
  python -u latent/eval_streaming.py --model $MODEL --ckpt $OUT/gate_nobuf/ckpt --gate --no_buffer --policy gate --step_cost $SC --out $OUT/eval_nobuf --cf_labels $OUT/cf_test.json > logs/${TAG}_eval_gate_nobuf_$SC.log 2>&1
  for K in 0 2 4 8; do
    python -u latent/eval_streaming.py --model $MODEL --ckpt $OUT/stageA/ckpt --policy fixed --fixed_k $K --step_cost $SC --out $OUT/eval_fixed > logs/${TAG}_eval_fixed${K}_$SC.log 2>&1
  done
  python -u latent/eval_streaming.py --model $MODEL --ckpt $OUT/stageA/ckpt --policy stitch --fixed_k 8 --step_cost $SC --out $OUT/eval_stitch > logs/${TAG}_eval_stitch_$SC.log 2>&1
  python -u latent/eval_streaming.py --model $MODEL --ckpt $OUT/stageA/ckpt --policy oracle --step_cost $SC --out $OUT/eval_oracle --cf_labels $OUT/cf_test.json > logs/${TAG}_eval_oracle_$SC.log 2>&1
done
echo PIPELINE_DONE
