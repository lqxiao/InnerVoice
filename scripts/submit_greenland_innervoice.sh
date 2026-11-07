#!/bin/bash
# Submit the InnerVoice RL job to Greenland — same flow as RufusSafetyRSI/scripts/submit-rsi.sh:
# refresh conduit creds into the DEDICATED `greenland` aws profile (default untouched), then run
# RufusGym's scripts.greenland.submit with AWS_PROFILE=greenland.
#
# Usage: bash scripts/submit_greenland_innervoice.sh [--dry-run] [-n 1] [extra submit.py args...]
set -euo pipefail

PROFILE=greenland
REPO_ROOT=/local/home/liqiangx/STP/rufus_gym/src/RufusGym
SCRIPT_REL=examples/innervoice/run-innervoice-rl-greenland.sh
IMAGE=docker.io/vllm/vllm-openai:v0.9.1
INITIATIVE=${GREENLAND_INITIATIVE_ID:-SFAI-shared-p5en-eks}

echo "=== refreshing conduit creds into profile '${PROFILE}' (default profile untouched) ==="
ada credentials update \
  --account=684288478426 --provider=conduit --role=RufusScienceConduitRole \
  --profile="${PROFILE}" --once

echo "=== submitting innervoice-rl (AWS_PROFILE=${PROFILE}) ==="
cd "${REPO_ROOT}"
touch .gl_skip_pip   # marker: the run script manages its own env inside the vllm image (see runcmd.tpl guard)
AWS_PROFILE="${PROFILE}" "${REPO_ROOT}/.venv/bin/python" -m scripts.greenland.submit \
  --task innervoice-rl \
  --nodes 1 \
  --mode run \
  --script "${SCRIPT_REL}" \
  --image "${IMAGE}" \
  --initiative-id "${INITIATIVE}" \
  "$@"
