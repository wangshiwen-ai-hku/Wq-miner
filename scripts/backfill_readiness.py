"""
One-shot script: back-fill self_corr + performance_gain for IS-passing alphas
that were incorrectly classified as `landed_but_correlated` due to missing
readiness probe data (self_corr=-1, perf_gain=0).

Usage:
    python scripts/backfill_readiness.py [--rounds 20,21,22] [--dry-run]
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha_agent.color_log import apply_color_logging
from alpha_agent.wq_client import WQBrainClient
from alpha_agent.feedback_memory import FeedbackRecord

logger = logging.getLogger(__name__)


def extract_alpha_id(alpha_link: str) -> str:
    return alpha_link.rstrip("/").split("/")[-1]


def main():
    parser = argparse.ArgumentParser(description="Back-fill readiness probe for old alphas")
    parser.add_argument(
        "--rounds", type=str, default="20,21,22",
        help="Comma-separated round IDs to back-fill (default: 20,21,22)",
    )
    parser.add_argument(
        "--feedback-file", type=str, default="data/feedback_memory.json",
    )
    parser.add_argument(
        "--credentials", type=str, default="credentials.json",
    )
    parser.add_argument(
        "--max-self-corr", type=float, default=0.7,
        help="Pool admission threshold for self_corr",
    )
    parser.add_argument(
        "--min-perf-gain", type=float, default=150.0,
        help="Perf gain threshold used as proxy when self_corr unmeasured",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing",
    )
    args = parser.parse_args()

    apply_color_logging(level=logging.INFO)

    target_rounds = {int(r.strip()) for r in args.rounds.split(",")}
    feedback_path = Path(args.feedback_file)

    # Load feedback memory
    data = json.loads(feedback_path.read_text())
    records_raw = data["records"]

    # Find candidates: IS-passing, in target rounds, self_corr not yet measured
    targets = [
        (i, r) for i, r in enumerate(records_raw)
        if r.get("round_id") in target_rounds
        and r.get("status") == "PASS"
        and not r.get("failed_checks")
        and r.get("sharpe", 0) >= 1.25
        and r.get("fitness", 0) >= 1.0
        and r.get("self_corr", -1.0) < 0
        and r.get("alpha_link", "")
    ]

    if not targets:
        logger.info("✅ No alphas to back-fill — all already have readiness data.")
        return

    logger.info(f"🔍 Found {len(targets)} alphas to probe (rounds {sorted(target_rounds)})")
    for _, r in targets:
        logger.info(f"   round={r['round_id']} sharpe={r['sharpe']:.2f} {r['alpha_link']}")

    if args.dry_run:
        logger.info("\n[DRY RUN] — no changes written")
        return

    # Connect to WQ Brain
    client = WQBrainClient(credentials_path=args.credentials)

    promoted = []
    demoted = []

    for idx, (raw_idx, raw) in enumerate(targets):
        alpha_id = extract_alpha_id(raw["alpha_link"])
        logger.info(
            f"\n[{idx+1}/{len(targets)}] Probing {alpha_id} "
            f"(round {raw['round_id']}, sharpe={raw['sharpe']:.2f})"
        )
        try:
            readiness = client.check_submission_readiness(
                alpha_id=alpha_id,
                max_self_corr=args.max_self_corr,
                min_performance_gain=args.min_perf_gain,
            )
        except Exception as e:
            logger.warning(f"   ⚠️ Probe failed: {e} — skipping")
            continue

        old_bucket = raw["bucket"]

        raw["self_corr"] = readiness.get("self_corr", -1.0)
        raw["performance_before"] = readiness.get("performance_before", 0.0)
        raw["performance_after"] = readiness.get("performance_after", 0.0)
        raw["performance_gain"] = readiness.get("performance_gain", 0.0)
        raw["submission_ready"] = readiness.get("ready", False)

        # Re-classify with real data
        rec = FeedbackRecord(**{
            k: v for k, v in raw.items()
            if k in FeedbackRecord.__dataclass_fields__
        })
        new_bucket = rec.classify()
        raw["bucket"] = new_bucket

        logger.info(
            f"   📊 self_corr={raw['self_corr']:.3f}, "
            f"perf_gain={raw['performance_gain']:.0f} → "
            f"{old_bucket} → {new_bucket}"
        )

        if old_bucket != new_bucket:
            if new_bucket == "landed":
                promoted.append(raw)
            else:
                demoted.append(raw)

    # Write back
    feedback_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"\n💾 Saved updated feedback_memory.json")

    logger.info(f"\n{'─'*60}")
    logger.info(f"📊 Back-fill complete:")
    logger.info(f"   ✅ Promoted to landed:            {len(promoted)}")
    logger.info(f"   🚫 Confirmed landed_but_correlated: {len(demoted)}")
    logger.info(f"   (unchanged: {len(targets) - len(promoted) - len(demoted)})")

    if promoted:
        logger.info("\n🌱 Newly promoted to landed — consider adding to seed pool:")
        for r in promoted:
            logger.info(
                f"   [{r['round_id']}] sharpe={r['sharpe']:.2f} "
                f"self_corr={r['self_corr']:.3f} "
                f"perf_gain={r['performance_gain']:.0f}  "
                f"{r['alpha_link']}"
            )


if __name__ == "__main__":
    main()
