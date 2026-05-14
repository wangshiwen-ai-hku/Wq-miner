"""
Seed Validator: gate-keeps unverified candidate seeds before they enter
the live SeedPool.

The motivation: external seed catalogs (e.g. CrisperX alpha50) report
historical Sharpe/Fitness from years ago. Those numbers may have decayed,
and feeding them into the evolutionary loop as if they were still strong
parents will mislead the LLM and waste simulation budget on hybrids built
around dead signals.

This module runs each candidate solo on WQ Brain, captures the *current*
IS metrics, and only promotes those that still clear a configurable
quality bar. Results persist to candidate_seeds.json so subsequent runs
skip already-validated entries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .seed_pool import (
    CANDIDATE_SEEDS,
    CandidateSeed,
    SeedPool,
    family_hash,
)
from .wq_client import WQBrainClient

logger = logging.getLogger(__name__)


@dataclass
class ValidationThreshold:
    """
    Quality bar a candidate must clear to be promoted to the seed pool.

    Matches the `landed` bucket definition in FeedbackRecord.classify():
    seeds must clear the same bar as a freshly-evolved alpha that earns a
    pool slot. Anything weaker would dilute the pool and make every
    future hybrid descend from a mediocre parent.
    """
    min_sharpe: float = 1.25
    min_fitness: float = 1.0
    max_turnover: float = 70.0
    # Reject if any of these IS checks failed
    blocking_failed_checks: tuple = (
        "SELF_CORRELATION",
        "CONCENTRATED_WEIGHT",
        "LOW_SUB_UNIVERSE_SHARPE",
    )


class SeedValidator:
    """
    Validates unverified candidate seeds against live WQ Brain and
    promotes survivors into a SeedPool.
    """

    def __init__(
        self,
        wq_client: Optional[WQBrainClient] = None,
        save_path: str = "data/candidate_seeds.json",
        threshold: Optional[ValidationThreshold] = None,
    ):
        # wq_client may be None when this validator is used purely for
        # queueing (e.g. from the research agent). validate_pending()
        # requires a real client; queueing methods do not.
        self.wq_client = wq_client
        self.save_path = Path(save_path)
        self.threshold = threshold or ValidationThreshold()
        self.candidates: List[CandidateSeed] = []

    # ── Persistence ──────────────────────────────────────────────────

    def load_or_init(self) -> None:
        """Load existing candidate_seeds.json or initialize from CANDIDATE_SEEDS."""
        if self.save_path.exists():
            data = json.loads(self.save_path.read_text())
            self.candidates = [CandidateSeed(**d) for d in data]
            logger.info(
                f"📂 Loaded {len(self.candidates)} candidates "
                f"({self._count_by_status()})"
            )
            # Merge in any newly-added CANDIDATE_SEEDS entries that aren't
            # in the saved file yet (e.g. when the developer adds new ones)
            existing_exprs = {c.expression for c in self.candidates}
            new_count = 0
            for idx, (tag, expr, notes) in enumerate(CANDIDATE_SEEDS, start=1):
                if expr in existing_exprs:
                    continue
                self.candidates.append(CandidateSeed(
                    candidate_id=f"CAND_{len(self.candidates) + 1:03d}",
                    tag=tag,
                    expression=expr,
                    notes=notes,
                ))
                new_count += 1
            if new_count:
                logger.info(f"   ➕ {new_count} new candidates appended from CANDIDATE_SEEDS")
                self.save()
        else:
            self.candidates = [
                CandidateSeed(
                    candidate_id=f"CAND_{idx:03d}",
                    tag=tag,
                    expression=expr,
                    notes=notes,
                )
                for idx, (tag, expr, notes) in enumerate(CANDIDATE_SEEDS, start=1)
            ]
            logger.info(f"🆕 Initialized {len(self.candidates)} candidates from CANDIDATE_SEEDS")
            self.save()

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(
            json.dumps(
                [c.to_dict() for c in self.candidates],
                indent=2,
                ensure_ascii=False,
            )
        )

    def _count_by_status(self) -> str:
        counts = {}
        for c in self.candidates:
            counts[c.status] = counts.get(c.status, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in counts.items())

    def status_counts(self) -> dict:
        """Return {status: count} dict for programmatic use."""
        counts = {}
        for c in self.candidates:
            counts[c.status] = counts.get(c.status, 0) + 1
        return counts

    def pending_count(self) -> int:
        return sum(1 for c in self.candidates if c.status == "pending")

    # ── External intake ─────────────────────────────────────────────

    def add_candidate(self, tag: str, expression: str, notes: str = "",
                      source: str = "external") -> bool:
        """
        Append a new candidate to the pool. Used by external producers
        (research agent, manual curation, etc.) to grow the pool without
        editing CANDIDATE_SEEDS in source.

        Deduplicates by exact expression match. Returns True if added,
        False if it was already in the pool.

        Note: this does NOT validate immediately — it just queues the
        candidate as `pending`. Validation happens in batch later via
        validate_pending().
        """
        # Ensure pool is loaded (lets external callers add without first
        # calling load_or_init explicitly)
        if not self.candidates and self.save_path.exists():
            self.load_or_init()

        expr_norm = expression.strip()
        for c in self.candidates:
            if c.expression.strip() == expr_norm:
                logger.debug(f"   ↩️ Candidate already in pool: {expr_norm[:60]}")
                return False

        next_id = f"CAND_{len(self.candidates) + 1:03d}"
        candidate = CandidateSeed(
            candidate_id=next_id,
            tag=tag,
            expression=expr_norm,
            notes=f"[{source}] {notes}" if notes else f"[{source}]",
        )
        self.candidates.append(candidate)
        self.save()
        logger.info(
            f"   ➕ Queued candidate {next_id} [{tag}] from {source}: "
            f"{expr_norm[:60]}..."
        )
        return True

    def add_candidates_bulk(self, items: List[dict], source: str = "external") -> int:
        """
        Bulk version. Each item: {"tag": str, "expression": str, "notes": str}.
        Returns the number of newly-added (deduped) candidates.
        """
        added = 0
        for item in items:
            tag = item.get("tag", "unknown")
            expr = item.get("expression", "").strip()
            notes = item.get("notes") or item.get("hypothesis", "")
            if not expr:
                continue
            if self.add_candidate(tag, expr, notes, source=source):
                added += 1
        return added

    # ── Validation pipeline ──────────────────────────────────────────

    def pending_candidates(self) -> List[CandidateSeed]:
        return [c for c in self.candidates if c.status == "pending"]

    def validate_pending(self, max_concurrent: int = 3) -> int:
        """
        Run naked simulations for all pending candidates, update their
        status, and persist. Returns the count of newly-verified seeds.
        """
        if self.wq_client is None:
            raise RuntimeError(
                "validate_pending() needs a WQBrainClient — this validator "
                "was constructed in queue-only mode."
            )
        pending = self.pending_candidates()
        if not pending:
            logger.info("✅ No pending candidates to validate")
            return 0

        logger.info(
            f"\n🧪 Validating {len(pending)} pending candidates "
            f"(threshold: sharpe≥{self.threshold.min_sharpe}, "
            f"fitness≥{self.threshold.min_fitness}, "
            f"turnover≤{self.threshold.max_turnover})..."
        )

        expressions = [c.expression for c in pending]
        results = self.wq_client.simulate_batch(
            expressions=expressions,
            max_concurrent=max_concurrent,
            timeout_per_sim=300.0,
        )

        verified_count = 0
        for candidate, sim_result in zip(pending, results):
            candidate.validated_at = datetime.now().isoformat()

            if sim_result.status == "ERROR":
                candidate.status = "error"
                candidate.rejection_reason = sim_result.error_message[:200]
                logger.warning(
                    f"   ⚠️ {candidate.candidate_id} {candidate.tag} ERROR: "
                    f"{candidate.rejection_reason}"
                )
                continue

            candidate.validation_sharpe = sim_result.sharpe
            candidate.validation_fitness = sim_result.fitness
            candidate.validation_turnover = sim_result.turnover
            candidate.validation_failed_checks = list(sim_result.failed_checks)

            reason = self._evaluate(sim_result.sharpe, sim_result.fitness,
                                    sim_result.turnover, sim_result.failed_checks)
            if reason is None:
                candidate.status = "verified"
                verified_count += 1
                logger.info(
                    f"   ✅ {candidate.candidate_id} [{candidate.tag}] "
                    f"PROMOTED — sharpe={sim_result.sharpe:.2f}, "
                    f"fitness={sim_result.fitness:.2f}, "
                    f"turnover={sim_result.turnover:.1f}%"
                )
            else:
                candidate.status = "rejected"
                candidate.rejection_reason = reason
                logger.info(
                    f"   ❌ {candidate.candidate_id} [{candidate.tag}] "
                    f"REJECTED — {reason} "
                    f"(sharpe={sim_result.sharpe:.2f}, "
                    f"fitness={sim_result.fitness:.2f})"
                )

        self.save()
        logger.info(
            f"\n📊 Validation complete: "
            f"{verified_count} promoted / "
            f"{len(pending) - verified_count} rejected"
        )
        return verified_count

    def _evaluate(self, sharpe: float, fitness: float, turnover: float,
                  failed_checks: List[str]) -> Optional[str]:
        """Return a rejection reason string, or None if the seed passes."""
        if sharpe < self.threshold.min_sharpe:
            return f"sharpe {sharpe:.2f} < {self.threshold.min_sharpe}"
        if fitness < self.threshold.min_fitness:
            return f"fitness {fitness:.2f} < {self.threshold.min_fitness}"
        if turnover > self.threshold.max_turnover:
            return f"turnover {turnover:.1f}% > {self.threshold.max_turnover}%"
        blocking = [c for c in failed_checks if c in self.threshold.blocking_failed_checks]
        if blocking:
            return f"blocking IS checks failed: {', '.join(blocking)}"
        return None

    # ── Promotion ───────────────────────────────────────────────────

    def promote_verified_into(self, seed_pool: SeedPool) -> int:
        """
        Inject all `verified` candidates into the provided seed pool.
        Uses the live validation metrics as the seed score, so the
        sampler treats them on equal footing with PROVEN_SEEDS.

        Returns the count of seeds actually added (de-duplicated by family).
        """
        from .seed_pool import Seed

        added = 0
        for c in self.candidates:
            if c.status != "verified":
                continue
            score = 0.5 * c.validation_sharpe + 0.5 * c.validation_fitness
            seed = Seed(
                seed_id=c.candidate_id,
                tag=c.tag,
                expression=c.expression,
                family_hash=family_hash(c.expression),
                score=score,
                source="validated",
                metadata={
                    "notes": c.notes,
                    "validation_sharpe": c.validation_sharpe,
                    "validation_fitness": c.validation_fitness,
                    "validation_turnover": c.validation_turnover,
                    "validated_at": c.validated_at,
                },
            )
            if seed_pool.add_seed(seed):
                added += 1
        if added:
            logger.info(f"🌱 Promoted {added} validated seeds into the live pool")
        return added
