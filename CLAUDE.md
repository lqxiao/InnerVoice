# STP/think — InnerVoice (Think/Speak disclosure routing for streaming speech)

## What this project is
- **InnerVoice**: Qwen3 + LoRA whose last decoder-layer FFN is a two-expert MoE (inner/think expert, outer/answer expert) chosen per
  token by a disclosure gate (router); think tokens are readable but never sent to TTS. Code: `latent/sxs_*.py` (prefix is historical:
  trajectories follow the SxS paper, arXiv 2605.03314). Part 1 (latent-vector recurrence, null result) is in `latent/model.py`,
  `latent/train_fixed_k.py`, `REPORT.md`. Part 2 results: `REPORT_PART2.md` (built by `python latent/build_report_part2.py`).
- Evaluation protocol = SxS paper: rows Base / SFT no-think / SFT Standard CoT / InnerVoice; metrics ARI, ABO, AIRW, FCO (token
  indices), Average@k at T=1.0; plus our playback-simulator metrics (TTFC, P50/P95 pause, premature-commitment rate).
- Data: K&K + MuSiQue (`data_sxs/`, from `latent/sxs_data.py`), ConvFill (`data_convfill/`), DeepMath R1 (`data_math/dm_sxs_*`),
  Bedrock teacher traces (`data_math/self/sxs_{q32,q32high,gptoss,astra}_*`). Build scripts: `sxs_data.py`, `convfill_data.py`,
  `sxs_math_data.py` (select/split/align/selfgen), `bedrock_gen.py`.
- Chat demo: `python latent/innervoice_chat.py --ckpt results/sxs/convfill_1.7B/ckpt` (run on a GPU host).

## Compute (read before launching anything)
- This dev box has NO GPU. GPU hosts: `p40`, `p41`, `p4`, `p43` (shared with other users — check `nvidia-smi` and who owns processes first).
- **Shared environment on FSx (visible from all GPU hosts, 379 GB free):**
  `/fsx-alignment/dedicated-fsx-data-repo-alignment-us-east-1/home/liqiangx/innervoice/` = `venv/`, `hf/` (Qwen3-1.7B/4B), `think/`
  (code+data+results, no ckpts), `env.sh` → `source .../innervoice/env.sh` on any host. This is the user's real FSx home
  (`/home/liqiangx` on p41/p43 points to the unmounted `/fsx-multimodal` and is dead; p40/p4 have local homes). Local copies on
  `/scratch/liqiangx` (p41, p4) still exist; the root disks there are nearly full — prefer FSx for new work.
- Docker images available: `vllm/vllm-openai:v0.9.1` on p40 (8 idle A100s, 112 GB disk), `slimerl/slime:latest` (SGLang 0.5.15) on p41.
  These are the only way to run an inference engine here (pip vLLM/SGLang cannot be installed on glibc 2.26).
- glibc 2.26: pin `numpy<2.3`, `scipy<1.15`, `pyarrow<18`; no `datasets`; pip vLLM does not install (max 0.21); docker needs sudo and
  the root disk is ~95% full — do not `docker pull` large images without checking `df -h /`.
- Long SSH sessions get dropped and Midway SSH certs expire (~20 h): wait with `scripts/poll_remote.sh <host> "<check>"` (short polls),
  not with one long-lived ssh. Launch remote jobs with `nohup ... &` and `python -u`.
- **Never put a `pkill -f <pattern>` in the same command as lines containing that pattern** (it kills the calling shell). Kill by PID or
  use the `patter[n]` bracket trick, and never in a command that also launches the target.
- Bedrock (boto3, us-west-2, creds on this dev box only): Qwen3-32B returns native `reasoningContent`; Claude models return
  `content_filtered` for training-data generation (usage policy — do not work around); GPT-6 Astra rejects `temperature`.

## Conventions
- Keep think and answer streams separate everywhere (training CE split, inference streams, eval metrics).
- Report numbers with their sample size; 150-example tests have ±4 pt noise; AIME25 is 30 problems x 4 samples (±8 pt).
- Do not claim gains that are within noise; the user wants paper numbers quoted exactly and provenance of every baseline stated.
- Results tables are generated from `results/**/summary_*.json` — never hand-edit numbers into REPORT_PART2.md.

## State (2026-11-05)
- Done: K&K/MuSiQue SFT suite (1.7B/4B, ablations, Average@4, budget grid), ConvFill (judge), DeepMath->AIME25 (R1 teacher, 1.7B/4B),
  teacher-trace comparison (R1 / Qwen3-32B default & high effort / GPT-OSS-120B / GPT-6 Astra), RL round 1 (K&K+MuSiQue stage1/2,
  math RL with a 3k cap that collapsed AIME), budget-matched AIME control.
- Open: RL with longer rollout budgets and voice-stage rewards; self-distillation needs a vLLM-class generator.
