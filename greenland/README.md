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
