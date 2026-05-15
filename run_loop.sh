#!/usr/bin/env bash
# run_loop.sh
set -u
INTERVAL=${INTERVAL:-300}   # 5 min

while true; do
  echo "──── $(date '+%F %T') starting round ────"
  python run_agent.py --rounds 1 --pairs-per-round 6 --modes-per-pair 4 --llm-workers 6 -v
  echo "──── done, sleeping ${INTERVAL}s ────"
  sleep "$INTERVAL"
done