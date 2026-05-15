"""
Launcher: Alpha Mining Agent
=============================

Runs the evolutionary alpha mining loop:
    seed pool → diverse A-B pairing → LLM hybridization →
    WQ Brain simulation → feedback memory → repeat.

Pipeline interaction
--------------------
1. Reads candidate_seeds.json (populated by run_research.py).
2. On startup, if pending candidates >= --validation-min-batch OR
   --validate-now is passed, naked-simulates them and promotes those
   that clear the quality bar (sharpe >= 1.25, fitness >= 1.0,
   turnover <= 70%, no blocking IS-check failures) into the live pool.
3. Then runs evolutionary rounds, generating hybrids from the now-
   enriched pool.

Strict seed-pool purity: only the `landed` bucket enters the pool
(passes IS + has self_corr < 0.6 + perf_gain >= 150). Mediocre or
prod-book-correlated alphas stay in feedback memory only.

Examples
--------
    # Dry run — no WQ API calls, uses template fallback when no GEMINI_API_KEY
    python run_agent.py --dry-run --rounds 1

    # Two-round live run (validates candidates first if batch >= 10)
    python run_agent.py --rounds 2

    # Force validation of pending candidates this startup
    python run_agent.py --validate-now --rounds 0

    # Continuous mode
    python run_agent.py --continuous --interval 30

    # Larger search per round
    python run_agent.py --pairs-per-round 8 --modes-per-pair 3

    # Parallelize Gemini generation before WQ simulation
    python run_agent.py --pairs-per-round 8 --modes-per-pair 3 --llm-workers 6
"""

import os
import sys

# Make `alpha_agent.*` importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_agent.agent import main

if __name__ == "__main__":
    print()
    print("=" * 64)
    print("🧬  EVOLUTIONARY ALPHA MINER — Mining Agent")
    print("=" * 64)
    print("  Mode:  evolutionary loop on WorldQuant Brain")
    print("  Cost:  consumes WQ simulation budget per candidate")
    print("  Tip:   accumulate seeds first via `python run_research.py`")
    print("=" * 64)
    print()
    main()
