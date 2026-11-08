# Greenland (Amazon EKS GPU cluster) submission

Flow mirrors `RufusSafetyRSI/scripts/submit-rsi.sh`: `scripts/submit_greenland_innervoice.sh` refreshes conduit creds into the
dedicated `greenland` AWS profile and calls RufusGym's `scripts.greenland.submit` (code tarball -> S3, Volcano manifest -> SubmitJob).
The job code lives in `rufus_gym/src/RufusGym/examples/innervoice/` (copy of `latent/` RL stack + vLLM plugin + gzipped RL pool);
`run-innervoice-rl-greenland.sh` (copy here) is the pod-side script: pulls model/ckpt/data from the Hub, 4 vLLM engines (GPUs 4-7)
+ 4 trainer ranks (GPUs 0-3) on one p5en 8xH200 node. Image `docker.io/vllm/vllm-openai:v0.9.1`; the `.gl_skip_pip` marker in the
RufusGym root makes the pod entrypoint skip RufusGym's pip install (whose transformers>=5.6 pin would break vLLM 0.9.1).
Outputs: FSx `<repo>/home/liqiangx/innervoice-rl/<exp>/rl/` (S3 mirror printed at submit). Status: `CentralGlobalSchedulerCLI
inspect-job <id> --initiative-id SFAI-shared-p5en-eks --region us-west-2` (Midway only).

Jobs: 2026-11-07 `liqiangx-innervoice-rl-1107-060539` JobId 23a46a7a-bbde-4807-b415-66bace7a8751 (480 steps, 32x16, 12k cap,
w_unsup 0.5 / w_short 0.05 / w_fco 0.05, eval every 50).
- 2026-11-07 job 1 (23a46a7a) failed at engine start: 4 engines launched concurrently shared ~/.cache/vllm -> torch.compile artifact
  corruption ("vLLM failed to compile the model"). Fix: sequential engine start, per-engine VLLM_CACHE_ROOT/TORCHINDUCTOR_CACHE_DIR,
  retry with --enforce_eager. Resubmitted as `liqiangx-innervoice-rl-1107-061456` JobId 9eac7fb3-e1f1-45a6-9291-272496d8b209.
- job 2 (9eac7fb3): all 4 engines compiled and started (~1 min each, no eager fallback needed); trainer died on
  `ModuleNotFoundError: latent.simulator` (incomplete copy of the `latent/` import closure) and the `| tee` hid the exit code
  (job marked Succeeded). Fix: full closure copied, `set -o pipefail`. Resubmitted as `liqiangx-innervoice-rl-1107-063208`
  JobId f259c58b-937d-4a71-bad9-0a26b1227bf2.
- job 3 (f259c58b, `innervoice-rl-1107-063208`): ran 112 steps (~4 min/step on 8xH200 at 32x16, 12k cap) and COLLAPSED:
  AIME25 0.192 (step 0) -> 0.025 (50) -> 0.000 (100); think tokens 2,866 -> 318 -> 45; flicker (short blocks) 0.45 -> 21 -> 0.2;
  blocks 3.7 -> 30 -> 1.3. Samples: step 50 = word-level think/speak oscillation; step 100 = 70-token think then one short public
  paragraph (speak-everything, brief). Dense penalties (unsup 0.5, short 0.05/block, FCO) dominated the sparse correctness signal and
  there was no KL anchor to the SFT policy. Cancelled at step 112. Checkpoints/logs mirrored to S3 (log.jsonl mirror was stale while open).
- job 4 `liqiangx-innervoice-rl-1107-142214` JobId 310433e2-faf3-4c10-8869-6836580e6bdd: reward v3 (gated bounded Q bonus, soft
  overlong -0.5, KL 0.02 tokens / 0.05 router to the frozen SFT reference), unseen DeepMath pool, 32x16, 12k, 480 steps. p4 runs the
  same reward at 32x8 / 8k as `results/rl/q32high_4B_engine_v3` (step 1 KL = 0 sanity check passed; ~4.2 min/step).
- p4 v3 (reward v3, 32x8, 8k): AIME 0.192 -> 0.308 (step 50, clean disclosure: FCO 1,214, 6.9 blocks) -> 0.283 (step 100) but router
  flicker returned between steps 70-80 (short blocks 1.4 -> 20/rollout; AIME 166 blocks, 112 short). Q's flicker term saturates at 5
  short blocks and the router KL at 0.05 was too weak. Reward v4 = v3 + unconditional flicker penalty 0.3*min(1, n_short/10) +
  router KL 0.5. Job 4 (310433e2, still queued, v3 config) cancelled; resubmitted with v4.
- job 5 (reward v4) `liqiangx-innervoice-rl-1107-215712` JobId 792bf62a-6c71-457b-8541-bfe578b61b98 (queued; p5en pool full). p4 runs v4 as `results/rl/q32high_4B_engine_v4`.
- job 5 (792bf62a, reward v4, 32x16/12k): step 50 AIME 0.317 (from 0.183) but 168 public blocks / 120 short blocks per sample = flicker;
  cancelled at step 63. p4 v4 (32x8/8k): 0.150 -> 0.333 (step 50, clean) -> 0.200 (step 100, 104 blocks). Diagnosis: argmax routing +
  REINFORCE on the taken decision; the SFT router is near-deterministic (0.4% of positions with 0.02<p<0.98) so there is no exploration
  and negative advantages push uncertain logits across 0 -> word-level oscillation. v5 = hash-reproducible sampled routing in the
  engine (latent/sxs_model.py::route_hash_u, engine == HF labels 882/882) + w_route 0.2 (+ v4 penalties/KL).
- job 6 (reward v5) `liqiangx-innervoice-rl-1108-054647` JobId e74e3ff3-23ad-472c-8d00-d0fb901cfc44; p4 runs v5 as `results/rl/q32high_4B_engine_v5`.
- job 6 (e74e3ff3) cancelled before start: the packed code had a bug exposed by sampled routing — `forward_train` cleared
  `moe.route` right after the forward, so under gradient checkpointing the last layer's recompute during backward used argmax/hash
  routing instead of the teacher-forced labels (wrong expert on label-disagreeing positions; crash with sampled routing). Fixed
  (route kept until the caller's backward). NOTE: this also affected every SFT run trained with --grad_ckpt (4B runs) at the
  ~1% of positions where the router disagreed with the labels. Resubmitted as job 7 `liqiangx-innervoice-rl-1108-055927`
  JobId a5bd2f22-50f3-4415-870b-d72d21d09566.
- p4 v5 (sampled routing, w_route 0.2, 32x8/8k) step 50: AIME 0.275 (step 0 with sampling 0.142), unfinished 0.067, FCO 2,040,
  7.3 blocks, 3.5 short blocks (baseline 2.4), think 4,922. Training: short blocks 3.8 -> 1.2, router KL ~0, no flicker. Clean.
- job 7 (a5bd2f22, reward v5, 32x16/12k): step 0 AIME 0.183 -> step 50 0.317 (unfinished 0.19, FCO 3,666, 9.3 blocks, 6.2 short
  blocks, think 6,525); training clean at step 89 (short blocks 1-2/rollout, ~5 min/step). p4 v5 (32x8/8k): 0.142 -> 0.275 -> 0.242
  (steps 0/50/100), 6.7% unfinished, no flicker through step 100.
