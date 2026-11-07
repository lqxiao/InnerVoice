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
