#!/usr/bin/env bash
# run_loop.sh
set -u
INTERVAL=${INTERVAL:-120}   # 30 min

while true; do
  echo "──── $(date '+%F %T') starting round ────"
  python run_agent.py --rounds 1 --pairs-per-round 4 --modes-per-pair 2 -v
  echo "──── done, sleeping ${INTERVAL}s ────"
  sleep "$INTERVAL"
done