#!/bin/bash
# Robust waiter: short ssh polls every 120 s (long-lived ssh sessions get dropped). usage: poll_remote.sh <host> "<remote check command printing DONE when satisfied>" <max_polls>
H=$1; CMD=$2; N=${3:-400}
for i in $(seq 1 $N); do
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 $H "$CMD" 2>/dev/null | grep -v -E "bind:|channel_setup|Could not|^\]0;")
  echo "$OUT" | grep -q "^DONE" && { echo "$OUT"; exit 0; }
  sleep 120
done
echo "TIMEOUT: $OUT"
