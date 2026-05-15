"""
Operator diversity utilities for detecting formula monoculture.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List


OPERATOR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

OVERCROWDED_OPERATORS = {
    "rank",
    "group_rank",
    "ts_rank",
    "ts_decay_linear",
    "group_neutralize",
}

UNDERUSED_OPERATOR_FAMILIES: Dict[str, List[str]] = {
    "distribution_shape": ["ts_zscore", "ts_quantile", "ts_scale", "zscore", "quantile"],
    "extreme_position": ["ts_arg_max", "ts_arg_min", "ts_max", "ts_min"],
    "relationship": ["ts_corr", "ts_covariance", "ts_regression", "vector_neut"],
    "stability": ["ts_std_dev", "ts_av_diff", "ts_skewness", "ts_kurtosis"],
    "state_change": ["days_from_last_change", "last_diff_value", "hump", "jump_decay"],
    "cross_sectional_bounds": ["group_zscore", "group_scale", "winsorize", "normalize", "bucket"],
}


def extract_operators(expression: str) -> List[str]:
    """Return lower-case function/operator names used in a Fast Expression."""
    return [m.group(1).lower() for m in OPERATOR_RE.finditer(expression or "")]


def operator_counts(expressions: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for expression in expressions:
        counts.update(extract_operators(expression))
    return counts


def summarize_operator_usage(records: Iterable[Any], limit: int = 60) -> Dict[str, Any]:
    recent = list(records)[-limit:]
    counts = operator_counts(getattr(r, "expression", "") for r in recent)
    total = sum(counts.values())
    if total == 0:
        return {
            "total": 0,
            "top": [],
            "overcrowded": {},
            "underused_families": UNDERUSED_OPERATOR_FAMILIES,
        }

    overcrowded = {
        op: {
            "count": counts[op],
            "share": round(counts[op] / total, 3),
        }
        for op in sorted(OVERCROWDED_OPERATORS)
        if counts[op] > 0
    }
    return {
        "total": total,
        "top": counts.most_common(10),
        "overcrowded": overcrowded,
        "underused_families": UNDERUSED_OPERATOR_FAMILIES,
    }


def format_operator_usage_for_prompt(records: Iterable[Any], limit: int = 60) -> str:
    summary = summarize_operator_usage(records, limit=limit)
    if summary["total"] == 0:
        return ""

    lines = ["### Operator Diversity Pressure:"]
    top = ", ".join(f"{op}={count}" for op, count in summary["top"][:8])
    lines.append(f"- Recent operator counts: {top}")
    if summary["overcrowded"]:
        crowded = ", ".join(
            f"{op} {data['share']:.0%}"
            for op, data in summary["overcrowded"].items()
        )
        lines.append(f"- Crowded operators to ration this round: {crowded}")
    lines.append(
        "- Prefer at least one underused family when valid: "
        "distribution_shape, extreme_position, relationship, stability, "
        "state_change, or cross_sectional_bounds."
    )
    return "\n".join(lines)
