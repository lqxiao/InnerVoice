---
license: apache-2.0
base_model: Qwen/Qwen3-4B
library_name: transformers
pipeline_tag: text-generation
language: [en]
tags: [innervoice, think-while-talking, interleaved-reasoning, streaming, speech-assistant, qwen3, moe, disclosure-policy]
datasets: [zwhe99/DeepMath-103K]
---

# InnerVoice-Qwen3-4B (SFT)

**InnerVoice** is an interactive-assistant model that *thinks while it talks*. Instead of a long silent `<think>` block followed by the
answer, it interleaves two readable token streams at token granularity:

- an **inner voice** (private reasoning, readable but not spoken / not sent to TTS), and
- an **outer voice** (the public answer, emitted in short chunks as soon as each statement is certain).

Architecture: Qwen3-4B with LoRA (r=16, all layers) and the **last decoder layer's FFN replaced by a two-expert MoE** (think expert /
answer expert, both initialised from the pretrained FFN) selected per token by a **disclosure router** (small MLP on the last-layer
hidden state). The router's decision at position t chooses the expert that produces token t+1, so the mode of every generated token is
a single bit that comes for free during decoding - no special tokens, no decode-time controller. LoRA is folded into the weights here.

This checkpoint is the **SFT stage** trained on 1,550 DeepMath-103K problems with interleaved think/answer trajectories distilled from
Qwen3-32B (high reasoning effort; private reasoning + reader-facing solution paragraphs aligned by anchor coverage). An RL stage
(GRPO with correctness, premature-commitment and first-content-onset rewards) is in progress. Code, data recipe and report:
https://github.com/lqxiao/InnerVoice

## Usage (transformers, remote code)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "liqiangx/InnerVoice-Qwen3-4B"
tok = AutoTokenizer.from_pretrained(repo)
m = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype="bfloat16").cuda()
out = m.generate_streams(tok, "Find the sum of all positive divisors of 360.", max_new_tokens=2048)
print("INNER :", out["think"])    # private reasoning (do not speak)
print("OUTER :", out["speak"])    # public answer, newline-separated chunks
```

`out["interleaved"]` gives the token-level (mode, text) sequence for a streaming UI. Plain `m.generate()` works too but returns both
streams mixed in one text; read `m.moe.last_route` to split them.

For math problems the model was trained with the instruction appended to the user turn:
`Solve the problem. Reason privately; publicly present the solution in short paragraphs as it becomes certain, and end with 'Final answer: \boxed{<answer>}'.`

## Usage (vLLM)

A vLLM plugin (custom `InnerVoiceQwen3ForCausalLM` class, vLLM 0.9.x) is in the GitHub repo under `latent/vllm_plugin`
(`pip install latent/vllm_plugin`). It reproduces the transformers decode token-for-token; the per-token mode is recovered from the
router (see `latent/sxs_rl_engine.py::label_types`).

## Evaluation (AIME 2025, Average@4, temperature 1.0, 12k-token budget, 30 problems x 4 samples, +-8 pt)

EVAL_TABLE

Metric definitions follow *When to Think, When to Speak: Learning Disclosure Policies for LLM Reasoning* (ICML 2026, arXiv
2605.03314): FCO = index of the first public token with task content, AIRW = mean private span before each public block.

## Limitations

- Trained on 1,550 math problems only; not instruction-tuned for general chat beyond what Qwen3-4B already does.
- Think tokens are readable; the model is small (4B) and its inner voice can be wrong - treat it as a transcript, not a guarantee.
- The `e_think` input embedding of the research code was dropped (ablation showed no effect), so this checkpoint is a plain
  per-token two-expert model and runs on any engine that can host a custom Qwen3 class.

## Citation

```
@misc{innervoice2026, title={InnerVoice: think while talking with a last-layer disclosure MoE},
      author={Liqiang Xiao}, year={2026}, url={https://github.com/lqxiao/InnerVoice}}
```
