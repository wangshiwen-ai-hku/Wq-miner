"""
Feedback Memory: tracks simulation results, failure patterns, and
guides the next generation of alpha candidates.

This is the "brain" that prevents the evolutionary loop from
repeating failed ideas or converging on self-correlated families.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Record of a single candidate evaluation."""
    expression: str
    tag_pair: str
    mutation_mode: str
    hypothesis: str
    parent_a: str
    parent_b: str

    # Simulation results
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns: float = 0.0
    status: str = "PENDING"          # PASS / FAIL / ERROR / PENDING
    failed_checks: List[str] = field(default_factory=list)
    alpha_link: str = ""

    # Submission-gate metrics (only populated for IS-passing alphas via
    # check_submission_readiness — sentinel -1.0 means "not measured")
    self_corr: float = -1.0
    performance_before: float = 0.0
    performance_after: float = 0.0
    performance_gain: float = 0.0
    submission_ready: bool = False

    # Classification
    bucket: str = ""                 # see classify()
    timestamp: str = ""
    round_id: int = 0

    def classify(self) -> str:
        """Classify into feedback bucket."""
        if self.status == "ERROR":
            self.bucket = "error"
        elif self.status == "PASS" and not self.failed_checks:
            if self.sharpe >= 1.25 and self.fitness >= 1.0:
                # Pool admission is decoupled from auto-submission:
                #   - Pool bar: self_corr < 0.7 (matches platform's hard gate)
                #   - Auto-submit bar (in agent.py): stricter 0.6 + perf_gain ≥ 150
                # Seed pool tolerates 0.6 ≤ self_corr < 0.7 because min-corr
                # sampling enforces diversity at pairing time.
                # When self_corr wasn't measured (-1, API hiccup), fall back
                # to perf_gain as a proxy — strong perf overrides unknown corr.
                if 0 <= self.self_corr < 0.7:
                    self.bucket = "landed"
                else:
                    # self_corr >= 0.7 (too correlated) or
                    # self_corr == -1 (measurement failed after retries) → reject
                    self.bucket = "landed_but_correlated"
            else:
                self.bucket = "strong_not_landed"
        elif "SELF_CORRELATION" in self.failed_checks:
            self.bucket = "self_corr_trap"
        elif "HIGH_TURNOVER" in [c.upper().replace(" ", "_") for c in self.failed_checks]:
            self.bucket = "high_turnover"
        elif "CONCENTRATED_WEIGHT" in [c.upper().replace(" ", "_") for c in self.failed_checks]:
            self.bucket = "weight_concentration"
        elif self.sharpe < 0.75:
            self.bucket = "low_sharpe"
        elif self.fitness < 0.6:
            self.bucket = "low_fitness"
        else:
            self.bucket = "other_fail"
        return self.bucket

    def to_dict(self) -> dict:
        return asdict(self)


class FeedbackMemory:
    """
    Persistent memory of all candidate evaluations.

    Tracks:
    - Which tag pairs / mutation modes work best
    - Which expression patterns fail repeatedly
    - Self-correlation traps to avoid
    - High-turnover patterns to penalize
    """

    def __init__(self, save_path: str = "data/feedback_memory.json"):
        self.records: List[FeedbackRecord] = []
        self.save_path = Path(save_path)
        self.current_round: int = 0

    def add_record(self, record: FeedbackRecord):
        """Add a new feedback record."""
        record.timestamp = datetime.now().isoformat()
        record.round_id = self.current_round
        record.classify()
        self.records.append(record)

    def get_context_for_prompt(self, tag_pair: str = "", max_records: int = 8) -> str:
        """
        Generate feedback context string for the LLM prompt.

        Provides:
        - Recent successful patterns
        - Known failure modes to avoid
        - Tag-pair specific history
        """
        if not self.records:
            return ""

        lines = []

        # Recent successes
        successes = [r for r in self.records if r.bucket in ("landed", "strong_not_landed")]
        if successes:
            lines.append("### Recently Successful Patterns:")
            for r in successes[-3:]:
                lines.append(
                    f"- [{r.mutation_mode}] {r.expression[:60]}... "
                    f"(Sharpe={r.sharpe:.2f}, Fitness={r.fitness:.2f})"
                )

        # Self-correlation traps (IS-level check failure)
        corr_traps = [r for r in self.records if r.bucket == "self_corr_trap"]
        if corr_traps:
            lines.append("\n### AVOID — Self-Correlation Traps:")
            for r in corr_traps[-3:]:
                lines.append(f"- {r.expression[:60]}... (correlated with existing alphas)")

        # Landed but blocked at submission gate (self_corr too high vs. user's
        # production book, or performance gain below 100). These passed IS but
        # the signal overlaps the user's existing alpha portfolio — the LLM
        # should aim to *decorrelate* from these structures.
        landed_corr = [r for r in self.records if r.bucket == "landed_but_correlated"]
        if landed_corr:
            lines.append(
                "\n### AVOID — Landed but Blocked from Seed Pool "
                "(self-corr ≥ 0.7 vs. user's prod book, or self-corr unmeasured AND perf_gain < 150):"
            )
            for r in landed_corr[-4:]:
                lines.append(
                    f"- {r.expression[:70]}... "
                    f"(self_corr={r.self_corr:.2f}, perf_gain={r.performance_gain:.0f}) — "
                    f"generate something STRUCTURALLY orthogonal to this"
                )

        # High turnover failures
        ht_fails = [r for r in self.records if r.bucket == "high_turnover"]
        if ht_fails:
            lines.append("\n### AVOID — High Turnover Patterns:")
            for r in ht_fails[-3:]:
                lines.append(f"- {r.expression[:60]}... (turnover too high)")

        # Low sharpe failures (recent)
        low_sharpe = [r for r in self.records if r.bucket == "low_sharpe"]
        if low_sharpe:
            lines.append("\n### AVOID — Low Sharpe Patterns:")
            for r in low_sharpe[-3:]:
                lines.append(f"- {r.expression[:60]}... (Sharpe={r.sharpe:.2f})")

        # Tag-pair specific history
        if tag_pair:
            tag_records = [r for r in self.records if r.tag_pair == tag_pair]
            if tag_records:
                lines.append(f"\n### History for tag pair [{tag_pair}]:")
                for r in tag_records[-3:]:
                    lines.append(
                        f"- [{r.bucket}] {r.expression[:50]}... "
                        f"(Sharpe={r.sharpe:.2f})"
                    )

        return "\n".join(lines) if lines else ""

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all recorded evaluations."""
        if not self.records:
            return {"total": 0}

        bucket_counts = Counter(r.bucket for r in self.records)
        mode_success = {}
        mode_total = {}
        tag_success = {}
        tag_total = {}

        for r in self.records:
            mode_total[r.mutation_mode] = mode_total.get(r.mutation_mode, 0) + 1
            tag_total[r.tag_pair] = tag_total.get(r.tag_pair, 0) + 1

            if r.bucket in ("landed", "strong_not_landed"):
                mode_success[r.mutation_mode] = mode_success.get(r.mutation_mode, 0) + 1
                tag_success[r.tag_pair] = tag_success.get(r.tag_pair, 0) + 1

        # Success rates
        mode_rates = {
            k: f"{mode_success.get(k, 0)}/{v} ({100*mode_success.get(k, 0)/v:.0f}%)"
            for k, v in mode_total.items()
        }
        tag_rates = {
            k: f"{tag_success.get(k, 0)}/{v} ({100*tag_success.get(k, 0)/v:.0f}%)"
            for k, v in tag_total.items()
        }

        # Best candidates
        sorted_records = sorted(
            [r for r in self.records if r.sharpe > 0],
            key=lambda r: (r.fitness, r.sharpe),
            reverse=True,
        )

        return {
            "total": len(self.records),
            "buckets": dict(bucket_counts),
            "mutation_mode_success": mode_rates,
            "tag_pair_success": tag_rates,
            "current_round": self.current_round,
            "best_candidates": [
                {
                    "expression": r.expression[:80],
                    "sharpe": r.sharpe,
                    "fitness": r.fitness,
                    "bucket": r.bucket,
                }
                for r in sorted_records[:5]
            ],
        }

    def get_best_mutation_modes(self, top_n: int = 3) -> List[str]:
        """Get the mutation modes with highest success rates."""
        if not self.records:
            return list(MUTATION_MODES_DEFAULT)

        mode_scores: Dict[str, float] = {}
        mode_counts: Dict[str, int] = {}

        for r in self.records:
            mode_counts[r.mutation_mode] = mode_counts.get(r.mutation_mode, 0) + 1
            if r.bucket in ("landed", "strong_not_landed"):
                mode_scores[r.mutation_mode] = mode_scores.get(r.mutation_mode, 0) + 1
            elif r.bucket in ("low_sharpe", "low_fitness"):
                mode_scores[r.mutation_mode] = mode_scores.get(r.mutation_mode, 0) - 0.3

        # Rank by success rate
        rated = sorted(
            mode_scores.items(),
            key=lambda x: x[1] / max(mode_counts.get(x[0], 1), 1),
            reverse=True,
        )

        return [m for m, _ in rated[:top_n]]

    def save(self):
        """Persist feedback memory to JSON."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_round": self.current_round,
            "records": [r.to_dict() for r in self.records],
        }
        self.save_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )
        logger.info(f"💾 Feedback memory saved ({len(self.records)} records)")

    def load(self) -> bool:
        """Load feedback memory from JSON."""
        if not self.save_path.exists():
            return False
        data = json.loads(self.save_path.read_text())
        self.current_round = data.get("current_round", 0)
        self.records = []
        for d in data.get("records", []):
            r = FeedbackRecord(**{
                k: v for k, v in d.items()
                if k in FeedbackRecord.__dataclass_fields__
            })
            self.records.append(r)
        logger.info(f"📂 Loaded {len(self.records)} feedback records")
        return True


# Default modes to try when there's no feedback history
MUTATION_MODES_DEFAULT = [
    "mild_corr_breaker",
    "yield_plus_improvement",
    "weak_gate_hybrid",
    "regime_hybrid",
    "conditional_activation",
]
