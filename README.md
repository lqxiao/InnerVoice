# InnerVoice: token-level Think/Speak disclosure routing for streaming speech (Qwen3)

**InnerVoice** (Part 2, `latent/sxs_*.py`): Qwen3 + LoRA whose last decoder layer is a two-expert MoE — an *inner* (think)
expert and an *outer* (answer) expert — selected per token by a disclosure gate. Think tokens are readable but never sent to
TTS; answer tokens are spoken. Trained by SFT on interleaved trajectories (K&K, MuSiQue, ConvFill, DeepMath), evaluated under
the SxS protocol (ARI/ABO/AIRW, Average@k) plus playback-simulator metrics (TTFC, P95 pause, premature commitments).
Results: `REPORT_PART2.md`. The file prefix `sxs_` is historical (the trajectory format follows the SxS paper).

---

# Part 1 (historical): streaming latent reasoning on Qwen3 (Think/Speak gate)

**Finding (see REPORT.md):** fixed-budget latent recurrence gave no accuracy gain over a LoRA-only direct-answer baseline at any
K in {0,1,2,4,8} on Qwen3-1.7B and Qwen3-4B, for both an adapter recurrence and a COCONUT-style LM recurrence, with or without
teacher signals. Phase 2/3 training was therefore not run as an experiment; the code is implemented, unit-tested and smoke-validated.

Qwen3 + LoRA with a recurrent latent reasoning module and a binary Think/Speak gate. Between public answer
chunks the model can run extra hidden computation (`z_{k+1} = LayerNorm(z_k + F(z_k, context))`) without emitting
`<think>` tokens; the latent state is injected as M soft tokens that the answer attends to.

```
latent/
  model.py              LatentModule (recurrent F), GateHead, LatentReasoner (Qwen3+LoRA wrapper; fixed-K forward,
                        multi-chunk forward, KV-cache greedy decoding, batch-1 streaming decoder with gate decisions)
  data.py               synthetic verified tasks (arith_chain, word_prob, graph_reach; held-out even_sum, deep chains;
                        multi-question "chunked" prompts), answer normalisation
  common.py             chat-template prompts, left/right padding collator
  train_fixed_k.py      Phase 1: fixed-budget training/eval (K=0..8, LoRA-only, CODI self-distillation, aux intermediate
                        supervision, --recurrence adapter|lm, --z_init learned|last)
  diag_latent.py        is the latent used? (z shuffle / zero / noise probes, state statistics, aux decode accuracy)
  report_phase1.py, plot_phase1.py   tables and accuracy-vs-K figure
  baselines.py          base Qwen3 direct (no thinking) and full explicit CoT (thinking mode)
  chunked.py            Phase 2: chunk trajectories, weak + counterfactual k* labels, buffer-aware Think/Speak labels
  label_counterfactual.py  counterfactual budget experiments (chunk correctness vs k)
  train_chunked.py      Phase 2: stage A (random k) / stage B (Answer_CE + lambda*Gate_BCE)
  rl_gate.py            Phase 3: custom batched rollouts + GRPO-style policy gradient on gate (and answer) with
                        simulator-based reward (accuracy, TTFC, P95 stall, GPU time, premature statements)
  simulator.py          Phase 4a: text-only playback simulator (audio durations, buffer, stalls)
  eval_streaming.py     streaming evaluation under gate / fixed-k / STITCH-style / oracle policies
tests/                  gradient, causal-mask, left-padding, KV-cache, latent-influence, chunked, RL tests (CPU, tiny Qwen3)
scripts/                sweep launchers
results/, REPORT.md     experiment outputs and findings
```

## Setup
```
python -m venv .venv && .venv/bin/pip install torch "transformers>=4.52,<5" peft accelerate "numpy<2.3" tqdm safetensors pytest
.venv/bin/python -m pytest tests -q          # CPU unit tests (tiny random Qwen3; downloads only the tokenizer)
python latent/data.py --out data             # generate data
```

## Phase 1 (fixed K)
```
python latent/train_fixed_k.py --model Qwen/Qwen3-1.7B --out results/x/K4 --K 4 --n_slots 8 --eval_K 0,1,2,8 --save
python latent/train_fixed_k.py --model Qwen/Qwen3-1.7B --out results/x/lora_only --n_slots 0 --K 0
python latent/train_fixed_k.py ... --K 4 --distill codi --bs 16 --grad_accum 2
python latent/baselines.py --model Qwen/Qwen3-1.7B --out results/baselines_1.7B
python latent/report_phase1.py --dir results/p1_1.7B --baselines results/baselines_1.7B/results.json
python latent/diag_latent.py --ckpt results/x/K4/ckpt --K 4            # latent-usage diagnostics
python latent/train_fixed_k.py ... --recurrence lm --K 4               # COCONUT-style recurrence through the LM
```
Remote sweep launchers: `scripts/phase1_sweep.sh`, `scripts/phase2_pipeline.sh` (paths assume /scratch/liqiangx on the GPU hosts).
## Phase 2 (gate)
```
python latent/train_chunked.py --mode random --out results/p2/stageA                      # budget-robust chunked model
python latent/label_counterfactual.py --ckpt results/p2/stageA/ckpt --out results/p2/cf.json
python latent/train_chunked.py --mode labels --init results/p2/stageA/ckpt --cf_labels results/p2/cf.json --out results/p2/gate
python latent/train_chunked.py --mode labels ... --no_buffer --out results/p2/gate_nobuf  # baseline 6
python latent/eval_streaming.py --ckpt results/p2/gate/ckpt --gate --policy gate --step_cost 0.2 --out results/p2/eval
```
## Phase 3 (RL)
```
python latent/rl_gate.py --init results/p2/gate/ckpt --out results/p3/rl --step_cost 0.2
```
