"""
Seed Pool: manages the alpha seed inventory with family hashing,
scoring, and diversity-aware sampling.

Seeds come from:
- User's proven alpha expressions (worldquant_alpha_experience.md)
- Past simulation results
- WorldQuant Alpha 101 library
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Seed:
    """A single alpha seed expression."""
    seed_id: str
    tag: str                    # e.g. "fundamental", "price", "analyst", "option"
    expression: str
    family_hash: str
    score: float                # historical quality score
    source: str = "manual"      # "manual" / "evolved" / "alpha101"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Pair:
    """An A-B parent pair for hybridization."""
    pair_id: str
    a: Seed
    b: Seed
    tag_pair: str


# ── Expression normalization and family hashing ──────────────────────

def normalize_expression(expr: str) -> str:
    """Normalize expression text to reduce superficial duplicates."""
    expr = expr.lower().strip()
    expr = re.sub(r"\s+", "", expr)
    # Replace numeric constants with N, but preserve operator names
    expr = re.sub(r"\d+\.\d+|\d+", "N", expr)
    return expr


def short_hash(text: str, length: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def family_hash(expr: str) -> str:
    return short_hash(normalize_expression(expr))


# ── Tag pair targets for diversity ───────────────────────────────────
#
# Each entry is (tag_a, tag_b, weight). The sampler picks pairs with
# probability proportional to weight, so rare/cross-domain combinations
# (analyst, MDF, FND6, option, news) get oversampled relative to the
# "fundamental × price/volume" combos that have dominated past rounds.
#
# Weights are skipped at runtime when one side has no available seeds —
# the sampler degrades gracefully to whatever is actually populated.

TAG_PAIR_TARGETS: List[Tuple[str, str, float]] = [
    # ── Common cross-domain (lower weight — already explored heavily) ──
    ("fundamental", "price", 1.0),
    ("price", "fundamental", 1.0),
    ("fundamental", "volume", 1.0),
    ("volume", "fundamental", 1.0),

    # ── Underexplored cross-domain (high weight) ──
    ("analyst", "price", 3.0),
    ("analyst", "fundamental", 3.0),
    ("analyst", "volume", 3.0),
    ("fundamental", "option", 3.0),
    ("option", "fundamental", 3.0),
    ("price", "option", 2.5),
    ("option", "price", 2.5),
    ("news", "price", 2.5),
    ("news", "fundamental", 2.5),
    ("price", "volume", 1.5),
    ("volume", "price", 1.5),

    # ── MDF/FND6 cross-domain (high weight — orthogonal data dimension) ──
    ("fundamental_mdf", "price", 3.0),
    ("fundamental_mdf", "volume", 3.0),
    ("fundamental_mdf", "fundamental", 3.0),
    ("fundamental_mdf", "analyst", 3.0),
    ("fundamental_fnd6", "price", 3.0),
    ("fundamental_fnd6", "fundamental_mdf", 2.5),
    ("fundamental_fnd6", "volume", 3.0),
    ("price_volume", "fundamental", 2.0),
    ("price_volume", "fundamental_mdf", 2.5),

    # ── Same-domain (low weight — for fine-tuning within family) ──
    ("fundamental", "fundamental", 0.5),
    ("price", "price", 0.5),
    ("fundamental_mdf", "fundamental_mdf", 0.5),
]


# ── Seed pool from proven experience ─────────────────────────────────

# High-quality seeds from user's worldquant_alpha_experience.md
PROVEN_SEEDS: List[Tuple[str, str, float, str]] = [
    # (tag, expression, quality_score, notes)

    # ── USER PROVEN / HIGH QUALITY ──
    ("fundamental", "ts_rank(operating_income/cap, 252)",
     1.42, "Operating Earnings Yield — Sharpe 1.42, Fitness 1.06"),

    ("fundamental", "-ts_rank(fn_liab_fair_val_l1_a, 126)",
     1.38, "Liability Fair Value Deterioration — Sharpe 1.38, Fitness 1.14"),

    ("fundamental", "group_rank(ts_rank(operating_cashflow_reported_value / cap, 252), subindustry)",
     1.75, "Cashflow Yield subindustry ranked — Sharpe 1.75, self-corr 0.55"),

    ("fundamental",
     "group_rank(ts_rank(operating_cashflow_reported_value / cap, 252) + "
     "ts_rank(ts_delta(operating_cashflow_reported_value, 252), 252), subindustry)",
     1.50, "Cashflow Yield + YoY Improvement — Sharpe 1.50, Fitness 1.07"),

    ("analyst",
     "group_rank(ts_rank(est_eps / close, 252) + ts_rank(ts_delta(est_eps, 126), 126), subindustry)",
     1.92, "Analyst EPS Yield + Revision — Sharpe 1.92, Fitness 1.39"),

    # ── NEAR-MISS (good signal, needs refinement) ──
    ("fundamental",
     "group_rank(ts_rank(operating_income/cap, 252) + "
     "0.5 * ts_rank(ts_delta(operating_income/cap, 252), 126), subindustry)",
     1.92, "Operating Earnings Yield + Improvement — self-corr 0.83, needs decorrelation"),

    # ── STRUCTURAL TEMPLATES (proven structures to reuse) ──
    ("price", "rank(ts_delta(close, 5))", 0.80, "Price momentum template"),
    ("price", "ts_rank(returns, 20)", 0.85, "Returns ranking template"),
    ("volume", "rank(volume / ts_mean(volume, 20))", 0.75, "Relative volume template"),
    ("volume", "ts_rank(volume, 10)", 0.70, "Volume ranking template"),
    ("option", "rank(implied_volatility_call_120 - implied_volatility_put_120)",
     0.65, "IV skew template"),

    # ═════════════════════════════════════════════════════════════════
    # FROM WQ-Brain/commands.py — Alpha101 paper expressions
    # ═════════════════════════════════════════════════════════════════

    # Price-volume correlation
    ("price_volume", "(-1 * ts_corr(rank(open), rank(volume), 10))",
     0.70, "Alpha101: Open-volume correlation"),
    ("price_volume",
     "(-1 * rank(ts_covariance(rank(close), rank(volume), 5)))",
     0.65, "Alpha101: Close-volume covariance"),
    ("price_volume",
     "(sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1)))",
     0.60, "Alpha101: Volume-price reversal"),

    # VWAP signals
    ("price", "(rank((vwap - close)) / rank((vwap + close)))",
     0.65, "Alpha101: VWAP deviation ratio"),
    ("price", "(((high * low)^0.5) - vwap)",
     0.60, "Alpha101: Geometric mean vs VWAP"),

    # Momentum / reversal
    ("price",
     "((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))",
     0.75, "Alpha101: Multi-dimensional momentum"),
    ("price",
     "(-1 * ts_delta((((close - low) - (high - close)) / (close - low)), 9))",
     0.55, "Alpha101: Williams %R delta"),

    # Group neutralized signals
    ("price",
     "(-1 * Ts_Rank(ts_decay_linear(ts_corr(group_neutralize(vwap, sector), volume, 4), 8), 6))",
     0.70, "Alpha101: Sector-neutral VWAP-volume"),
]


# ═════════════════════════════════════════════════════════════════════
# CANDIDATE SEEDS — UNVERIFIED, must pass naked-simulation gate before
# being promoted into the live SeedPool. These come from CrisperX
# alpha50 (publication is several years old; the recorded Sharpe/Fitness
# may have decayed). The seed validator runs each one solo and keeps
# only those that still clear an IS quality threshold today.
# ═════════════════════════════════════════════════════════════════════

CANDIDATE_SEEDS: List[Tuple[str, str, str]] = [
    # (tag, expression, notes — NO historical score, we don't trust it)

    # MDF (market-derived fundamental) fields — new data dimension
    # ("fundamental_mdf", "rank(mdf_pva)",
    #  "CrisperX #0: Enterprise value"),
    # ("fundamental_mdf", "-rank(mdf_ite_q)",
    #  "CrisperX #1: Quarterly income tax"),
    # ("fundamental_mdf", "-rank(mdf_rnd)",
    #  "CrisperX #6: R&D intensity"),
    # ("fundamental_mdf", "-rank(mdf_avi)",
    #  "CrisperX #8: Anti-volatility"),
    # ("fundamental_mdf", "ts_zscore(mdf_gry, 26)",
    #  "CrisperX #21: Growth yield z-score"),
    # ("fundamental_mdf", "-rank(mdf_opi)",
    #  "CrisperX #36: Operating profitability"),
    # ("fundamental_mdf", "rank(mdf_rds)",
    #  "CrisperX #35: R&D to sales"),
    # ("fundamental_mdf", "-rank(mdf_sq5)",
    #  "CrisperX #48: Short-term quality"),
    # ("fundamental_mdf",
    #  "-ts_av_diff(mdf_eg3, 250)*ts_corr(mdf_eg3, mdf_sg3, 250)",
    #  "CrisperX #31: Earnings-sales corr"),
    # ("fundamental_mdf", "ts_mean(mdf_bet, 10)",
    #  "CrisperX #23: Beta smoothed"),

    # # FND6 (fundamental data v6) fields
    # ("fundamental_fnd6", "rank(fnd6_dcvt)",
    #  "CrisperX #11: Debt conversion"),
    # ("fundamental_fnd6", "ts_mean(fnd6_newqv1300_invfgq, 10)",
    #  "CrisperX #10: Finished goods inventory"),
    # ("fundamental_fnd6", "ts_mean(fnd6_optvol, 10)",
    #  "CrisperX #24: Option volume"),
    # ("fundamental_fnd6", "-rank(fnd6_newa1v1300_epspi)",
    #  "CrisperX #2: EPS surprise"),

    # # Analyst family (FAM) fields
    # ("analyst", "group_rank(fam_est_eps_rank, sector)",
    #  "CrisperX #30: Analyst EPS rank in sector"),

    # # Traditional fundamental fields
    # ("fundamental", "rank(current_ratio)",
    #  "CrisperX #17: Current ratio"),
    # ("fundamental", "-rank(return_assets)",
    #  "CrisperX #37: ROA reversal"),
]


@dataclass
class CandidateSeed:
    """A candidate seed pending validation against live WQ Brain."""
    candidate_id: str
    tag: str
    expression: str
    notes: str
    status: str = "pending"          # pending / verified / rejected / error
    validation_sharpe: float = 0.0
    validation_fitness: float = 0.0
    validation_turnover: float = 0.0
    validation_failed_checks: List[str] = field(default_factory=list)
    validated_at: str = ""
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SeedPool:
    """Manages the seed pool with family deduplication and persistence."""

    def __init__(self, seeds: Optional[List[Seed]] = None, save_path: str = "data/seed_pool.json"):
        self.seeds: List[Seed] = seeds or []
        self.save_path = Path(save_path)
        self._family_index: Dict[str, List[Seed]] = {}
        self._tag_index: Dict[str, List[Seed]] = {}
        self._rebuild_indices()

    def _rebuild_indices(self):
        self._family_index.clear()
        self._tag_index.clear()
        for s in self.seeds:
            self._family_index.setdefault(s.family_hash, []).append(s)
            self._tag_index.setdefault(s.tag, []).append(s)

    def add_seed(self, seed: Seed) -> bool:
        """Add a seed, returns False if family already exists."""
        if seed.family_hash in self._family_index:
            # Update score if higher
            existing = self._family_index[seed.family_hash][0]
            if seed.score > existing.score:
                existing.score = seed.score
                existing.metadata.update(seed.metadata)
            return False
        self.seeds.append(seed)
        self._family_index.setdefault(seed.family_hash, []).append(seed)
        self._tag_index.setdefault(seed.tag, []).append(seed)
        return True

    def build_from_proven(self):
        """Build seed pool from proven experience records."""
        for idx, (tag, expr, score, notes) in enumerate(PROVEN_SEEDS, start=1):
            seed = Seed(
                seed_id=f"PROVEN_{idx:03d}",
                tag=tag,
                expression=expr,
                family_hash=family_hash(expr),
                score=score,
                source="proven",
                metadata={"notes": notes},
            )
            self.add_seed(seed)

    def add_from_simulation(self, expression: str, tag: str, sharpe: float,
                             fitness: float, source: str = "evolved") -> Seed:
        """Add a seed from a simulation result."""
        # Score = weighted combination
        score = 0.5 * sharpe + 0.5 * fitness
        seed = Seed(
            seed_id=f"EVO_{len(self.seeds)+1:04d}",
            tag=tag,
            expression=expression,
            family_hash=family_hash(expression),
            score=score,
            source=source,
            metadata={"sharpe": sharpe, "fitness": fitness},
        )
        self.add_seed(seed)
        return seed

    def sample_diverse_pairs(
        self,
        n_pairs: int = 10,
        random_seed: Optional[int] = None,
    ) -> List[Pair]:
        """
        Sample diverse A-B pairs across tag families using weighted random
        selection from TAG_PAIR_TARGETS.

        Constraints:
            - A and B must come from different seed families
            - Tag-pair weights bias sampling toward underexplored combos
              (analyst/MDF/FND6/option) rather than the dominant
              fundamental×price/volume pairs
            - Avoid reusing the same parent too often
            - Skip tag pairs where either side has zero seeds available
        """
        if random_seed is not None:
            random.seed(random_seed)

        by_tag = self._tag_index
        for tag in by_tag:
            by_tag[tag] = sorted(by_tag[tag], key=lambda x: x.score, reverse=True)

        # Filter tag-pair targets to those with seeds on both sides
        available_targets = [
            (a_tag, b_tag, w)
            for (a_tag, b_tag, w) in TAG_PAIR_TARGETS
            if by_tag.get(a_tag) and by_tag.get(b_tag)
        ]
        if not available_targets:
            return []

        pairs: List[Pair] = []
        used_pair_hashes = set()
        parent_usage: Dict[str, int] = {}
        # Track how many times each tag-pair has been picked — used to
        # softly penalize repeats within a single round so a round of 8 pairs
        # doesn't end up as 8 copies of (fundamental, price).
        tag_pair_usage: Dict[Tuple[str, str], int] = {}

        def pick_from_tag(tag: str, excluded_families: set) -> Optional[Seed]:
            candidates = [
                s for s in by_tag.get(tag, [])
                if s.family_hash not in excluded_families
            ]
            if not candidates:
                return None
            weights = [
                max(0.05, s.score) / (1 + parent_usage.get(s.family_hash, 0))
                for s in candidates
            ]
            return random.choices(candidates, weights=weights, k=1)[0]

        attempts = 0
        while len(pairs) < n_pairs and attempts < 500:
            attempts += 1

            # Weighted random pick of tag pair, dampened by how often we've
            # already picked it this round.
            tp_weights = [
                w / (1 + 1.5 * tag_pair_usage.get((a, b), 0))
                for (a, b, w) in available_targets
            ]
            a_tag, b_tag, _ = random.choices(
                available_targets, weights=tp_weights, k=1
            )[0]

            a = pick_from_tag(a_tag, excluded_families=set())
            if a is None:
                continue

            b = pick_from_tag(b_tag, excluded_families={a.family_hash})
            if b is None:
                continue

            pair_hash = short_hash(a.family_hash + "::" + b.family_hash, 12)
            if pair_hash in used_pair_hashes:
                continue

            used_pair_hashes.add(pair_hash)
            parent_usage[a.family_hash] = parent_usage.get(a.family_hash, 0) + 1
            parent_usage[b.family_hash] = parent_usage.get(b.family_hash, 0) + 1
            tag_pair_usage[(a_tag, b_tag)] = tag_pair_usage.get((a_tag, b_tag), 0) + 1

            pairs.append(Pair(
                pair_id=f"P{len(pairs)+1:04d}_{pair_hash}",
                a=a,
                b=b,
                tag_pair=f"{a.tag}+{b.tag}",
            ))

        return pairs

    def save(self):
        """Persist seed pool to JSON."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self.seeds]
        self.save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load(self) -> bool:
        """Load seed pool from JSON."""
        if not self.save_path.exists():
            return False
        data = json.loads(self.save_path.read_text())
        self.seeds = [Seed(**d) for d in data]
        self._rebuild_indices()
        return True

    def __len__(self):
        return len(self.seeds)

    def __repr__(self):
        tag_counts = {}
        for s in self.seeds:
            tag_counts[s.tag] = tag_counts.get(s.tag, 0) + 1
        return f"SeedPool({len(self.seeds)} seeds: {tag_counts})"
