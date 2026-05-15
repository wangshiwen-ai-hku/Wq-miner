"""
Launcher: Research Agent
========================

Runs the parallel discovery agent that produces hypotheses and structured
seed candidates by querying Gemini about WorldQuant Brain topics.

Output goes to:
    data/research_report.html    — accumulating HTML research report
    data/seeds_and_hypotheses.md — markdown summary of each cycle
    data/candidate_seeds.json    — STRUCTURED candidate pool (status=pending)
    data/research_agent.log      — full log

Candidates are NOT validated here — they're queued. Validation happens
inside the main alpha agent when pending count crosses the batch
threshold (or you force it with `python run_agent.py --validate-now`).

Examples
--------
    # Single topic from the built-in rotation
    python run_research.py

    # Specific topic
    python run_research.py --topic "MDF 与价格反应的 Alpha"

    # Walk through 5 topics from the rotation list
    python run_research.py --topics-batch 5

    # Faster/cheaper model for iterative use
    python run_research.py --model gemini-2.5-flash
"""

import os
import sys

# Make `alpha_agent.*` importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_agent.research_agent import main

if __name__ == "__main__":
    print()
    print("=" * 64)
    print("🔬  EVOLUTIONARY ALPHA MINER — Research Agent")
    print("=" * 64)
    print("  Mode:  WQ-field-grounded LLM (no WQ simulation budget consumed)")
    print("  Goal:  produce hypotheses + queue seed candidates for")
    print("         later batched validation.")
    print("=" * 64)
    print()
    main()
