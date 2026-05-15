"""
Prompt Evolver: versioned, feedback-driven system prompt and mode library.

The prompt library is intentionally stored as append-only JSON. Each evolved
version contains the full prompt/modes snapshot plus a unified diff from its
parent so changes are reviewable like a git patch.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """A full prompt/modes snapshot plus its patch history metadata."""

    version_id: str
    parent_id: Optional[str]
    created_at: str
    reason: str
    system_prompt: str
    mutation_modes: Dict[str, str]
    diff: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "reason": self.reason,
            "system_prompt": self.system_prompt,
            "mutation_modes": self.mutation_modes,
            "diff": self.diff,
            "metrics": self.metrics,
        }


class PromptLibrary:
    """Persistent, append-only prompt/mode version store."""

    def __init__(
        self,
        save_path: str = "data/prompt_versions.json",
        default_system_prompt: str = "",
        default_mutation_modes: Optional[Dict[str, str]] = None,
    ):
        self.save_path = Path(save_path)
        self.default_system_prompt = default_system_prompt
        self.default_mutation_modes = dict(default_mutation_modes or {})
        self.current_version_id = ""
        self.versions: List[PromptVersion] = []
        self.load_or_init()

    @property
    def current(self) -> PromptVersion:
        for version in reversed(self.versions):
            if version.version_id == self.current_version_id:
                return version
        return self.versions[-1]

    @property
    def system_prompt(self) -> str:
        return self._apply_runtime_hygiene(self.current.system_prompt)

    @property
    def mutation_modes(self) -> Dict[str, str]:
        return self._apply_mode_hygiene(self.current.mutation_modes)

    @staticmethod
    def _apply_runtime_hygiene(prompt: str) -> str:
        """Apply non-negotiable runtime guardrails to loaded historical prompts."""
        prompt = prompt.replace(
            "- Turnover control: hump(x, hump=0.01), jump_decay(x, d, sensitivity=0.5, force=0.1), ts_target_tvr_decay(x, target_tvr=0.1)",
            "- Turnover control: hump(x, hump=0.01), jump_decay(x, d, sensitivity=0.5, force=0.1)",
        )
        prompt = prompt.replace(
            "Use hump(), ts_decay_linear, or ts_target_tvr_decay for smoothing",
            "Use hump(), jump_decay(), or moderate-window ts_decay_linear only when needed for smoothing. Do NOT use ts_target_tvr_decay; it is inaccessible in this environment",
        )
        prompt = prompt.replace("ts_regression/vector_neut", "ts_corr/ts_covariance")
        prompt = prompt.replace("ts_corr/ts_covariance/ts_regression/vector_neut", "ts_corr/ts_covariance")
        prompt = prompt.replace("ts_corr/ts_covariance/ts_regression, ", "ts_corr/ts_covariance, ")
        prompt = prompt.replace(
            "- Utility: days_from_last_change(x), last_diff_value(x, d), vector_neut(x, y), ts_step(1)",
            "- Utility: days_from_last_change(x), last_diff_value(x, d), ts_step(1)",
        )
        prompt = prompt.replace("Prefer vector_neut(), group_zscore(), ts_regression residuals,", "Prefer group_zscore() or ts_corr/ts_covariance relationships,")
        prompt = re.sub(
            r"\n\n## Runtime Operator Diversity Guardrail\n(?:- .*\n?)+",
            "\n",
            prompt,
        ).rstrip()
        prompt = "\n".join(
            line for line in prompt.splitlines()
            if not PromptLibrary._is_operator_frequency_rule(line.strip())
        )
        if "## Runtime AVOID ERRORS" not in prompt:
            prompt = prompt.rstrip() + (
                "\n\n## Runtime AVOID ERRORS\n"
                "- Do NOT use vector_neut; live WQ simulation reports it as inaccessible or unknown.\n"
                "- For relationship operators use ts_corr(x, y, d) or ts_covariance(x, y, d). Avoid complex ts_regression signatures unless already proven by simulation.\n"
            )
        return prompt

    @staticmethod
    def _apply_mode_hygiene(modes: Dict[str, str]) -> Dict[str, str]:
        cleaned = dict(modes)
        if "relationship_residual" in cleaned:
            cleaned["relationship_residual"] = (
                "Make B a relationship component rather than a direct additive modifier. "
                "Use ts_corr(x, y, d) or ts_covariance(x, y, d) with exactly 3 inputs. "
                "Do not use vector_neut; live simulation reports it as inaccessible. "
                "Avoid complex ts_regression signatures unless already proven by simulation."
            )
        if "orthogonal_residual_rewrite" in cleaned:
            cleaned["orthogonal_residual_rewrite"] = (
                "Rewrite A into an orthogonal representation before B enters. Preserve A's "
                "economic thesis, but change at least two of: field transform, time horizon, "
                "group treatment, sign convention, activation logic. Prefer group_zscore(), "
                "ts_corr/ts_covariance relationships, or acceleration/change terms over direct A * (1 + k*B)."
            )
        if "operator_family_rotation" in cleaned:
            cleaned["operator_family_rotation"] = (
                "Generate a structurally novel child by rotating the operator family "
                "or signal mechanism. You may use common operators when they fit the "
                "idea, but prefer a different structural axis than the parent: "
                "distribution shape, relationship, extreme position, stability, "
                "state change, cross-sectional bounds, horizon, or activation logic."
            )
        return cleaned

    def load_or_init(self) -> None:
        if self.save_path.exists():
            data = json.loads(self.save_path.read_text())
            self.current_version_id = data.get("current_version_id", "")
            self.versions = [
                PromptVersion(**v) for v in data.get("versions", [])
            ]
            if self.versions:
                if not self.current_version_id:
                    self.current_version_id = self.versions[-1].version_id
                logger.info(
                    "📜 Prompt library loaded: %s (%d versions)",
                    self.current_version_id,
                    len(self.versions),
                )
                return

        initial = PromptVersion(
            version_id="prompt-v0001",
            parent_id=None,
            created_at=datetime.now().isoformat(),
            reason="Bootstrap from alpha_agent.llm_hybridizer defaults.",
            system_prompt=self.default_system_prompt,
            mutation_modes=dict(self.default_mutation_modes),
            diff="",
            metrics={},
        )
        self.versions = [initial]
        self.current_version_id = initial.version_id
        self.save()
        logger.info("📜 Prompt library initialized: %s", self.current_version_id)

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_version_id": self.current_version_id,
            "versions": [v.to_dict() for v in self.versions],
        }
        self.save_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def evolve_from_feedback(
        self,
        records: Sequence[Any],
        round_id: int,
        min_records: int = 4,
        evolution_interval: int = 2,
    ) -> Optional[PromptVersion]:
        """
        Create a new prompt version from recent feedback.

        This is deliberately rule-based instead of LLM-written: it makes prompt
        evolution reproducible and audit-friendly. The LLM still gets richer
        instructions next round, but the patch itself is deterministic.
        """
        evolution_interval = max(1, evolution_interval)
        if round_id % evolution_interval != 0:
            logger.info(
                "🧬 Prompt evolution skipped: waiting for %d-round interval "
                "(current round %d)",
                evolution_interval,
                round_id,
            )
            return None

        recent_rounds = {
            r for r in range(round_id - evolution_interval + 1, round_id + 1)
            if r > 0
        }
        recent = [r for r in records if getattr(r, "round_id", 0) in recent_rounds]
        if len(recent) < min_records:
            logger.info(
                "🧬 Prompt evolution skipped: only %d evaluated records in rounds %s",
                len(recent),
                sorted(recent_rounds),
            )
            return None

        buckets = Counter(getattr(r, "bucket", "") for r in recent)
        failed_checks = Counter(
            check
            for r in recent
            for check in getattr(r, "failed_checks", [])
        )
        simulation_errors = [
            getattr(r, "error_message", "")
            for r in recent
            if getattr(r, "status", "") == "ERROR" and getattr(r, "error_message", "")
        ]
        mode_counts = Counter(getattr(r, "mutation_mode", "") for r in recent)
        reasons: List[str] = []
        additions: List[str] = []
        new_modes = dict(self.current.mutation_modes)
        patch_ops: List[Dict[str, str]] = []

        avoid_errors: List[str] = []
        inaccessible_ops = self._extract_inaccessible_operators(simulation_errors)
        bad_arity_errors = self._extract_arity_errors(simulation_errors)
        if inaccessible_ops:
            ops_text = ", ".join(sorted(inaccessible_ops))
            reasons.append(f"inaccessible operators detected: {ops_text}")
            for op in sorted(inaccessible_ops):
                avoid_errors.append(
                    f"- Do NOT use `{op}`; WQ simulation reported it as inaccessible or unknown."
                )
        if bad_arity_errors:
            reasons.append(f"{len(bad_arity_errors)} operator arity errors detected")
            avoid_errors.extend(
                f"- Avoid malformed function calls: {message[:160]}"
                for message in bad_arity_errors
            )
            additions.append(
                "- Before returning an expression, verify every function call has the exact number of required inputs; when unsure, choose a simpler two-argument construction."
            )

        correlated = buckets.get("landed_but_correlated", 0) + buckets.get("self_corr_trap", 0)
        if correlated:
            reasons.append(f"{correlated} candidates were structurally useful but correlation-blocked")
            additions.extend([
                "- Treat self-correlation as a first-class failure, not a post-processing issue.",
                "- Do not preserve Parent A syntactically when recent landed alphas are correlation-blocked; preserve only its economic intent.",
                "- Force at least one orthogonal transformation: residualize, cross-sectional neutralize at a different group, change the time horizon, or switch from level to change/acceleration.",
                "- Avoid reusing the exact wrappers seen in correlated winners: ts_decay_linear(group_rank(...)), group_neutralize(A * (1 + k * rank(B))), and simple A + k*B blends.",
            ])
            self._upsert_mode(
                new_modes,
                patch_ops,
                "orthogonal_residual_rewrite",
                (
                    "Rewrite A into an orthogonal representation before B enters. "
                    "Preserve A's economic thesis, but change at least two of: field "
                    "transform, time horizon, group treatment, sign convention, activation "
                    "logic. Prefer group_zscore(), ts_corr/ts_covariance relationships, "
                    "or acceleration/change terms over direct A * (1 + k*B)."
                ),
            )
            self._upsert_mode(
                new_modes,
                patch_ops,
                "dataset_pivot_same_thesis",
                (
                    "Keep the same economic idea as A while pivoting to a sibling "
                    "dataset/field family when available. Example pivots: operating_income "
                    "-> operating_cashflow_reported_value/current_ratio/return_assets; "
                    "est_eps -> fam_est_eps_rank or ts_delta(est_eps,126); price reversal "
                    "-> volatility/range/liquidity state. The child must not contain "
                    "Parent A as a literal subtree."
                ),
            )

        high_turnover = max(
            buckets.get("high_turnover", 0),
            failed_checks.get("HIGH_TURNOVER", 0),
        )
        if high_turnover:
            reasons.append(f"{high_turnover} high-turnover signals need slower state changes")
            additions.extend([
                "- For price/volume inputs, separate fast entry signal from slow holding state; smooth the final output with hump() or ts_decay_linear(..., 10-20).",
                "- Do not use 5-10 day deltas/ranks as the main signal unless the final expression is explicitly turnover-controlled.",
            ])
            self._upsert_mode(
                new_modes,
                patch_ops,
                "slow_state_activation",
                "Convert a fast parent into a slower state machine: use B to define a 63-252 day regime/quality state, activate a smoothed version of A, and apply hump() or ts_decay_linear(..., 10-20) to the final expression.",
            )

        concentration = max(
            buckets.get("weight_concentration", 0),
            failed_checks.get("CONCENTRATED_WEIGHT", 0),
        )
        if concentration:
            reasons.append(f"{concentration} candidates hit weight concentration")
            additions.extend([
                "- Any ratio, division, or sparse fundamental/MDF/FND6 field must be bounded with rank/group_rank/winsorize before multiplication.",
                "- Prefer group_rank or group_zscore before final neutralization; avoid raw ratios inside multiplicative modifiers.",
            ])
            self._upsert_mode(
                new_modes,
                patch_ops,
                "bounded_cross_sectional_rebuild",
                "Rebuild the child from bounded components only: rank/group_rank/winsorize every raw ratio, use additive or if_else composition instead of raw multiplication, then neutralize/rank at subindustry or industry.",
            )

        weak_perf = buckets.get("low_fitness", 0) + buckets.get("low_sharpe", 0)
        if weak_perf >= max(2, len(recent) // 3):
            reasons.append(f"{weak_perf} low-performance candidates show stale templates")
            additions.extend([
                "- Do not default to weak_gate_hybrid or mild_corr_breaker when they only shrink exposure; require a distinct economic mechanism.",
                "- Prefer new mechanism combinations: quality x liquidity stress, analyst revision x dispersion, option skew x valuation, cashflow quality x price range state.",
            ])
            self._upsert_mode(
                new_modes,
                patch_ops,
                "mechanism_switch",
                "Generate a child whose mechanism is not a linear blend: choose one of confirmation, contradiction, timing, residualization, or regime segmentation. Explain the mechanism in the hypothesis and keep B influence indirect when direct addition would be template-like.",
            )

        if not additions and not avoid_errors and new_modes == self.current.mutation_modes:
            logger.info("🧬 Prompt evolution skipped: no actionable failure pattern detected")
            return None

        current_prompt = self.current.system_prompt
        active_rules = self._merge_active_rules(
            self._extract_adaptive_rules(current_prompt),
            additions,
        )
        active_error_rules = self._merge_active_rules(
            self._extract_avoid_errors(current_prompt),
            avoid_errors,
            max_rules=16,
        )
        block = self._build_evolution_block(
            round_id,
            buckets,
            mode_counts,
            active_rules,
            patch_ops,
            active_error_rules,
        )
        new_prompt = self._replace_or_append_block(current_prompt, block)
        if new_prompt == current_prompt and new_modes == self.current.mutation_modes:
            logger.info("🧬 Prompt evolution skipped: patch would be identical")
            return None

        parent = self.current
        version_id = f"prompt-v{len(self.versions) + 1:04d}"
        reason = "; ".join(reasons)
        diff = self._make_diff(parent.system_prompt, new_prompt, parent.mutation_modes, new_modes)
        version = PromptVersion(
            version_id=version_id,
            parent_id=parent.version_id,
            created_at=datetime.now().isoformat(),
            reason=reason,
            system_prompt=new_prompt,
            mutation_modes=new_modes,
            diff=diff,
            metrics={
                "round_id": round_id,
                "buckets": dict(buckets),
                "failed_checks": dict(failed_checks),
                "simulation_errors": simulation_errors,
                "mutation_modes": dict(mode_counts),
                "patch_ops": patch_ops,
            },
        )
        self.versions.append(version)
        self.current_version_id = version.version_id
        self.save()
        logger.info("🧬 Prompt evolved: %s -> %s", parent.version_id, version.version_id)
        logger.info("   Reason: %s", reason)
        return version

    @staticmethod
    def _build_evolution_block(
        round_id: int,
        buckets: Counter,
        mode_counts: Counter,
        active_rules: List[str],
        patch_ops: List[Dict[str, str]],
        avoid_errors: List[str],
    ) -> str:
        unique_rules = list(dict.fromkeys(active_rules))
        op_lines = [
            f"- {op['op']} mode `{op['mode']}`: {op['reason']}"
            for op in patch_ops
        ] or ["- retain active modes: no mode patch required"]
        lines = [
            "## Evolution Memory",
            f"Last updated from simulation round {round_id}.",
            f"Recent bucket distribution: {dict(buckets)}.",
            f"Recent mode usage: {dict(mode_counts)}.",
            "",
            "### Prompt Patch Operations",
            *op_lines,
            "",
            "### AVOID ERRORS",
            *(avoid_errors or ["- No active simulation syntax/operator errors recorded."]),
            "",
            "### Adaptive Rules",
            *unique_rules,
            "",
            "### Diversity Requirements",
            "- Every generated expression must differ from Parent A in at least two structural axes: operator family, lookback horizon, group operation, activation/conditioning, or dataset field family.",
            "- If the mode is exploratory, the child must not contain Parent A as a literal subtree; rewrite the idea instead of wrapping the old formula.",
            "- Prefer one compact orthogonal idea over stacking many familiar wrappers.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _replace_or_append_block(prompt: str, block: str) -> str:
        marker = "## Evolution Memory"
        idx = prompt.find(marker)
        if idx == -1:
            return prompt.rstrip() + "\n\n" + block + "\n"

        prefix = prompt[:idx].rstrip()
        return prefix + "\n\n" + block + "\n"

    @staticmethod
    def _extract_adaptive_rules(prompt: str) -> List[str]:
        start_marker = "### Adaptive Rules"
        end_marker = "### Diversity Requirements"
        start = prompt.find(start_marker)
        if start == -1:
            return []
        start += len(start_marker)
        end = prompt.find(end_marker, start)
        block = prompt[start:end if end != -1 else len(prompt)]
        return [
            line.strip()
            for line in block.splitlines()
            if line.strip().startswith("- ")
        ]

    @staticmethod
    def _extract_avoid_errors(prompt: str) -> List[str]:
        start_marker = "### AVOID ERRORS"
        end_marker = "### Adaptive Rules"
        start = prompt.find(start_marker)
        if start == -1:
            return []
        start += len(start_marker)
        end = prompt.find(end_marker, start)
        block = prompt[start:end if end != -1 else len(prompt)]
        return [
            line.strip()
            for line in block.splitlines()
            if line.strip().startswith("- ")
            and "No active simulation" not in line
        ]

    @staticmethod
    def _merge_active_rules(existing: List[str], additions: List[str], max_rules: int = 18) -> List[str]:
        merged = [
            rule for rule in list(dict.fromkeys(existing + additions))
            if not PromptLibrary._is_operator_frequency_rule(rule)
        ]
        if len(merged) <= max_rules:
            return merged
        return merged[-max_rules:]

    @staticmethod
    def _is_operator_frequency_rule(rule: str) -> bool:
        text = rule.lower()
        markers = (
            "ration crowded",
            "crowded defaults",
            "crowded default",
            "crowded operator",
            "non-crowded operator",
            "underused operator",
            "default rank-decay wrappers",
            "avoid default rank-decay",
            "avoid default rank/decay",
            "at most two",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _upsert_mode(
        modes: Dict[str, str],
        patch_ops: List[Dict[str, str]],
        name: str,
        description: str,
    ) -> None:
        if name in modes:
            return
        modes[name] = description
        patch_ops.append({
            "op": "add",
            "mode": name,
            "reason": "feedback pattern requires this additional active strategy",
        })

    @staticmethod
    def _tighten_mode(
        modes: Dict[str, str],
        patch_ops: List[Dict[str, str]],
        name: str,
        suffix: str,
    ) -> None:
        return

    @staticmethod
    def _deprecate_bad_modes(
        recent: Sequence[Any],
        modes: Dict[str, str],
        patch_ops: List[Dict[str, str]],
    ) -> None:
        return

    @staticmethod
    def _extract_inaccessible_operators(errors: List[str]) -> set:
        operators = set()
        for error in errors:
            match = re.search(
                r'inaccessible or unknown operator\s+"([^"]+)"',
                error,
                flags=re.IGNORECASE,
            )
            if match:
                operators.add(match.group(1))
        return operators

    @staticmethod
    def _extract_arity_errors(errors: List[str]) -> List[str]:
        return [
            error for error in errors
            if "Invalid number of inputs" in error
        ]

    @staticmethod
    def _remove_operator_from_modes(
        modes: Dict[str, str],
        patch_ops: List[Dict[str, str]],
        operator: str,
    ) -> None:
        return

    @staticmethod
    def _make_diff(
        old_prompt: str,
        new_prompt: str,
        old_modes: Dict[str, str],
        new_modes: Dict[str, str],
    ) -> str:
        old_text = (
            ["# system_prompt\n"]
            + old_prompt.splitlines(keepends=True)
            + ["\n# mutation_modes.json\n"]
            + json.dumps(old_modes, indent=2, sort_keys=True).splitlines(keepends=True)
        )
        new_text = (
            ["# system_prompt\n"]
            + new_prompt.splitlines(keepends=True)
            + ["\n# mutation_modes.json\n"]
            + json.dumps(new_modes, indent=2, sort_keys=True).splitlines(keepends=True)
        )
        return "".join(
            difflib.unified_diff(
                old_text,
                new_text,
                fromfile="previous_prompt",
                tofile="next_prompt",
            )
        )
