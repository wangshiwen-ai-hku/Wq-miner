"""
Alpha Mining Agent: the main orchestrator.

Runs the evolutionary loop:
    seed pool → diverse pairing → LLM hybridization → WQ simulation → feedback → repeat

Usage:
    python -m alpha_agent.agent [--rounds N] [--pairs-per-round N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .wq_client import WQBrainClient, SimulationResult
from .seed_pool import SeedPool, Pair
from .seed_validator import SeedValidator, ValidationThreshold
from .llm_hybridizer import GeminiHybridizer, HybridCandidate, MUTATION_MODES
from .feedback_memory import FeedbackMemory, FeedbackRecord, MUTATION_MODES_DEFAULT

logger = logging.getLogger(__name__)


class AlphaMiningAgent:
    """
    Evolutionary alpha mining agent.

    Orchestrates the full loop:
    1. Build/load seed pool from proven experience
    2. Sample diverse A-B parent pairs
    3. Generate hybrid candidates via Gemini
    4. Simulate on WorldQuant Brain
    5. Record feedback and update seed pool
    6. Repeat with guided evolution
    """

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        data_dir: str = "data",
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
        pairs_per_round: int = 6,
        modes_per_pair: int = 2,
        dry_run: bool = False,
        fresh: bool = False,
        validate_candidates: bool = True,
        validation_min_batch: int = 10,
        force_validate_now: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.pairs_per_round = pairs_per_round
        self.modes_per_pair = modes_per_pair

        # Initialize components
        logger.info("=" * 60)
        logger.info("🧬 Evolutionary Alpha Mining Agent")
        logger.info("=" * 60)

        # Seed pool
        self.seed_pool = SeedPool(save_path=str(self.data_dir / "seed_pool.json"))
        if fresh:
            logger.info("🔄 FRESH mode — clearing saved state")
            self.seed_pool.build_from_proven()
            self.seed_pool.save()
        elif not self.seed_pool.load():
            logger.info("Building seed pool from proven experience...")
            self.seed_pool.build_from_proven()
            self.seed_pool.save()
        else:
            logger.info("♻️ Resumed seed pool from previous session")
        logger.info(f"📊 {self.seed_pool}")

        # Feedback memory
        self.feedback = FeedbackMemory(save_path=str(self.data_dir / "feedback_memory.json"))
        if fresh:
            logger.info("🔄 Feedback memory cleared")
        else:
            self.feedback.load()
        logger.info(f"🧠 Feedback memory: {len(self.feedback.records)} records")

        # LLM hybridizer
        try:
            self.hybridizer = GeminiHybridizer(
                api_key=gemini_api_key,
                model_name=gemini_model,
            )
            self._use_llm = True
        except ValueError as e:
            if dry_run:
                logger.warning(f"⚠️ {e} — using template fallback in dry-run mode")
                self.hybridizer = None
                self._use_llm = False
            else:
                raise

        # WQ Brain client
        if not dry_run:
            self.wq_client = WQBrainClient(credentials_path=credentials_path)
            self.wq_client.login()
        else:
            self.wq_client = None
            logger.info("🏜️ DRY RUN mode — skipping WQ Brain simulation")

        # Seed validator — gates unverified candidates (from CrisperX list,
        # research agent, or manual adds) behind a live IS-quality bar
        # before any seed enters the live pool.
        #
        # Batched validation: we don't validate every time the agent starts.
        # The pool accumulates until pending >= validation_min_batch (or the
        # user forces with force_validate_now). This decouples cheap
        # discovery (research agent) from expensive validation (WQ sim).
        if validate_candidates and not dry_run:
            self._stage_banner("🚪 SEED VALIDATION GATE")
            self.validator = SeedValidator(
                wq_client=self.wq_client,
                save_path=str(self.data_dir / "candidate_seeds.json"),
            )
            self.validator.load_or_init()
            self._log_candidate_pool_status()

            pending_n = self.validator.pending_count()
            if pending_n == 0:
                logger.info("✅ No pending candidates — skipping validation")
            elif force_validate_now or pending_n >= validation_min_batch:
                trigger = "FORCED" if force_validate_now else f"pending {pending_n} ≥ batch {validation_min_batch}"
                logger.info(f"🧪 Validation triggered ({trigger})")
                self.validator.validate_pending()
            else:
                logger.info(
                    f"⏸️ Validation deferred — pending {pending_n} < batch "
                    f"threshold {validation_min_batch}. "
                    f"Re-run with --validate-now to force, or let candidates "
                    f"accumulate from the research agent."
                )

            promoted = self.validator.promote_verified_into(self.seed_pool)
            if promoted:
                self.seed_pool.save()
            logger.info(f"📊 Seed pool after validation gate: {self.seed_pool}")
        else:
            self.validator = None
            if validate_candidates and dry_run:
                logger.info("🏜️ Skipping seed validation in dry-run mode")

    @staticmethod
    def _stage_banner(text: str):
        logger.info("")
        logger.info("=" * 64)
        logger.info(text)
        logger.info("=" * 64)

    def _log_candidate_pool_status(self):
        counts = self.validator.status_counts()
        if not counts:
            logger.info("📂 Candidate pool: empty")
            return
        parts = [f"{k}={v}" for k, v in sorted(counts.items())]
        logger.info(f"📂 Candidate pool ({len(self.validator.candidates)} total): {', '.join(parts)}")

    def run_round(self) -> pd.DataFrame:
        """
        Run a single evolutionary round.

        Returns:
            DataFrame with all candidate results from this round.
        """
        self.feedback.current_round += 1
        round_id = self.feedback.current_round
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 ROUND {round_id}")
        logger.info(f"{'='*60}")

        # Step 1: Sample diverse parent pairs
        logger.info(f"\n📐 Sampling {self.pairs_per_round} diverse A-B pairs...")
        pairs = self.seed_pool.sample_diverse_pairs(
            n_pairs=self.pairs_per_round,
            random_seed=int(time.time()) % 10000,
        )

        if not pairs:
            logger.error("❌ No valid pairs could be sampled!")
            return pd.DataFrame()

        logger.info(f"   Got {len(pairs)} pairs:")
        for p in pairs:
            logger.info(
                f"   {p.pair_id}: [{p.tag_pair}] "
                f"A={p.a.expression[:40]}... | B={p.b.expression[:40]}..."
            )

        # Step 2: Determine mutation modes — exploit best historical modes
        # for (modes_per_pair - 1) slots, but ALWAYS reserve 1 slot for a
        # random exploratory mode the recent history hasn't favored. This
        # prevents the loop from collapsing onto the same 1–2 modes
        # (which was happening: every round was mild_corr_breaker +
        # yield_plus_improvement, both "preservation-heavy" mutations
        # that inherit parent A's structure → high prod-book correlation).
        all_modes = list(MUTATION_MODES.keys())

        if self.feedback.records and self.modes_per_pair > 1:
            exploit_n = max(1, self.modes_per_pair - 1)
            best_modes = self.feedback.get_best_mutation_modes(top_n=exploit_n)
            # Pad with first defaults if history is too thin
            for m in MUTATION_MODES_DEFAULT:
                if m not in best_modes and len(best_modes) < exploit_n:
                    best_modes.append(m)

            # Reserve 1 exploratory slot — pick a random mode NOT in best_modes
            explore_candidates = [m for m in all_modes if m not in best_modes]
            if explore_candidates:
                best_modes.append(random.choice(explore_candidates))
        elif self.feedback.records:
            # Only 1 slot — still alternate exploit/explore by coin flip
            top1 = self.feedback.get_best_mutation_modes(top_n=1)
            if random.random() < 0.5 and top1:
                best_modes = top1
            else:
                best_modes = [random.choice(all_modes)]
        else:
            best_modes = MUTATION_MODES_DEFAULT[:self.modes_per_pair]

        logger.info(f"\n🎯 Mutation modes for this round: {best_modes}")

        # Step 3: Generate candidates
        logger.info(f"\n✨ Generating hybrid candidates...")
        all_candidates: List[HybridCandidate] = []

        for pair in pairs:
            candidates = []

            if self._use_llm and self.hybridizer:
                feedback_ctx = self.feedback.get_context_for_prompt(
                    tag_pair=pair.tag_pair, max_records=6
                )
                candidates = self.hybridizer.generate_batch(
                    parent_a=pair.a.expression,
                    parent_b=pair.b.expression,
                    tag_pair=pair.tag_pair,
                    pair_id=pair.pair_id,
                    modes=best_modes,
                    feedback_context=feedback_ctx,
                )

            # Fallback: if LLM produced 0 candidates, use templates
            if not candidates:
                logger.info(
                    f"   ⚡ LLM returned 0 candidates for {pair.pair_id}, "
                    f"falling back to template generation"
                )
                candidates = self._template_generate(pair, best_modes)

            all_candidates.extend(candidates)
            logger.info(
                f"   Pair {pair.pair_id}: generated {len(candidates)} candidates"
            )

        logger.info(f"\n📦 Total candidates: {len(all_candidates)}")

        if not all_candidates:
            logger.error("❌ No candidates generated!")
            return pd.DataFrame()

        # Step 4: Simulate on WQ Brain
        logger.info(f"\n🧪 Simulating {len(all_candidates)} candidates...")

        if self.dry_run:
            # Serial fake simulation for testing
            sim_results = [
                self._fake_simulate(c) for c in all_candidates
            ]
        else:
            # ── PARALLEL batch simulation ──
            sim_results = self.wq_client.simulate_batch(
                expressions=[c.expression for c in all_candidates],
                max_concurrent=3,
                timeout_per_sim=300.0,
            )

        # Step 4b: Process results and record feedback
        results = []
        for i, (candidate, sim_result) in enumerate(
            zip(all_candidates, sim_results)
        ):
            logger.info(
                f"\n--- Candidate {i+1}/{len(all_candidates)} ---"
            )
            logger.info(f"   Expression: {candidate.expression}")
            logger.info(f"   Hypothesis: {candidate.hypothesis}")
            logger.info(f"   Mode: {candidate.mutation_mode}")

            # Record feedback
            record = FeedbackRecord(
                expression=candidate.expression,
                tag_pair=candidate.tag_pair,
                mutation_mode=candidate.mutation_mode,
                hypothesis=candidate.hypothesis,
                parent_a=candidate.parent_a_expr,
                parent_b=candidate.parent_b_expr,
                sharpe=sim_result.sharpe,
                fitness=sim_result.fitness,
                turnover=sim_result.turnover,
                returns=sim_result.returns,
                status=sim_result.status,
                failed_checks=sim_result.failed_checks,
                alpha_link=sim_result.alpha_link,
            )
            record.classify()

            # For IS-passing alphas, also probe the submission gates
            # (self-correlation + performance gain). This is a few extra API
            # calls — we already wanted these values for auto-submit, now we
            # just persist them so the LLM can learn to decorrelate. After
            # this we re-classify so `landed_but_correlated` is populated
            # *before* the record goes into feedback memory.
            readiness = None
            _is_candidate = (
                sim_result.status == "PASS"
                and not sim_result.failed_checks
                and sim_result.sharpe >= 1.25
                and sim_result.fitness >= 1.0
            )
            if (
                not self.dry_run
                and sim_result.alpha_id
                and _is_candidate
            ):
                logger.info(
                    f"   🔍 Probing submission gates for {sim_result.alpha_id}..."
                )
                try:
                    readiness = self.wq_client.check_submission_readiness(
                        alpha_id=sim_result.alpha_id,
                        max_self_corr=0.6,
                        min_performance_gain=150.0,
                    )
                    record.self_corr = readiness.get("self_corr", -1.0)
                    record.performance_before = readiness.get("performance_before", 0.0)
                    record.performance_after = readiness.get("performance_after", 0.0)
                    record.performance_gain = readiness.get("performance_gain", 0.0)
                    record.submission_ready = readiness.get("ready", False)
                    record.classify()  # may demote to landed_but_correlated
                    logger.info(
                        f"   📊 Gates: self_corr={record.self_corr:.3f}, "
                        f"perf_gain={record.performance_gain:.1f}, "
                        f"→ bucket={record.bucket}"
                    )
                except Exception as e:
                    logger.warning(f"   ⚠️ Readiness probe failed: {e}")

            # Only add to feedback memory if it's a valid evaluation (not an API/concurrency error)
            if record.bucket != "error":
                self.feedback.add_record(record)

            results.append({
                "round": round_id,
                "expression": candidate.expression,
                "tag_pair": candidate.tag_pair,
                "mutation_mode": candidate.mutation_mode,
                "hypothesis": candidate.hypothesis,
                "sharpe": sim_result.sharpe,
                "fitness": sim_result.fitness,
                "turnover": sim_result.turnover,
                "returns": sim_result.returns,
                "status": sim_result.status,
                "bucket": record.bucket,
                "failed_checks": "; ".join(sim_result.failed_checks),
                "alpha_link": sim_result.alpha_link,
            })

            logger.info(f"   Result: {sim_result.summary()}")
            logger.info(f"   Bucket: {record.bucket}")

            # STRICT seed-pool gate: ONLY `landed` enters the seed pool.
            #
            # Explicitly excluded:
            #   - strong_not_landed: passes IS checks but sharpe/fitness too
            #     weak to be a useful parent (would dilute the pool)
            #   - landed_but_correlated: high-quality IS signal BUT overlaps
            #     the user's prod book — every child would inherit that
            #     correlation, defeating the decorrelation goal
            #
            # These still enter feedback memory so the LLM learns from them;
            # they just don't become future parents.
            if record.bucket == "landed":
                tag = candidate.tag_pair.split("+")[0]
                self.seed_pool.add_from_simulation(
                    expression=candidate.expression,
                    tag=tag,
                    sharpe=sim_result.sharpe,
                    fitness=sim_result.fitness,
                    source="evolved",
                )
                logger.info(f"   🌱 Added to seed pool (landed)!")
            elif record.bucket in ("strong_not_landed", "landed_but_correlated"):
                logger.info(
                    f"   🚫 Not adding to seed pool — bucket={record.bucket} "
                    f"(only `landed` enters the pool to keep it pure)"
                )

                # Auto-submit if readiness probe already showed all gates green
                if record.submission_ready and sim_result.alpha_id:
                    logger.info(
                        f"   ✅ All gates passed! Submitting...\n"
                        f"      Self-corr: {record.self_corr:.4f} (<0.7 ✓)\n"
                        f"      Perf gain: +{record.performance_gain:.1f} (>100 ✓)"
                    )
                    submitted = self.wq_client.submit_alpha(
                        sim_result.alpha_id, skip_checks=True
                    )
                    if submitted:
                        logger.info(f"   🏆 SUBMITTED SUCCESSFULLY!")
                elif readiness is not None and not record.submission_ready:
                    logger.info(
                        f"   ⏸️ Not submitting: {readiness.get('reason', 'gates failed')}"
                    )

        # Step 5: Save state
        self.seed_pool.save()
        self.feedback.save()

        # Step 6: Round summary
        result_df = pd.DataFrame(results)
        self._print_round_summary(result_df, round_id)

        # Save results CSV
        csv_path = self.data_dir / f"round_{round_id:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(csv_path, index=False)
        logger.info(f"\n💾 Results saved to: {csv_path}")

        return result_df

    def run_evolution(self, n_rounds: int = 3) -> pd.DataFrame:
        """
        Run multiple evolutionary rounds.

        Args:
            n_rounds: Number of evolution rounds to run.

        Returns:
            Concatenated DataFrame of all round results.
        """
        all_results = []

        for i in range(n_rounds):
            logger.info(f"\n{'#'*60}")
            logger.info(f"# EVOLUTION ROUND {i+1}/{n_rounds}")
            logger.info(f"{'#'*60}")

            round_df = self.run_round()
            all_results.append(round_df)

            # Print running statistics
            stats = self.feedback.get_statistics()
            logger.info(f"\n📈 Running Statistics:")
            logger.info(f"   Total evaluations: {stats['total']}")
            logger.info(f"   Buckets: {stats.get('buckets', {})}")

            if i < n_rounds - 1:
                logger.info(f"\n⏳ Waiting 10s before next round...")
                time.sleep(10)

        final_df = pd.concat(all_results, ignore_index=True)

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 EVOLUTION COMPLETE — {n_rounds} rounds")
        logger.info(f"{'='*60}")
        self._print_final_summary(final_df)

        return final_df

    def run_continuous(
        self,
        interval_minutes: int = 30,
        max_rounds: int = 0,
    ):
        """
        Run the agent continuously in an infinite loop.
        Crash-resilient: catches all exceptions per round,
        saves state, and retries after the interval.

        Args:
            interval_minutes: Minutes to wait between rounds.
            max_rounds: Stop after N rounds (0 = infinite).
        """
        import signal

        _stop = False
        _sig_count = 0

        def _handle_signal(sig, frame):
            nonlocal _stop, _sig_count
            _sig_count += 1
            if _sig_count == 1:
                logger.info(f"\n🛑 Received signal {sig}, finishing current round... (Press Ctrl+C again to force quit)")
                _stop = True
            else:
                logger.warning(f"\n🛑 Multiple signals received. FORCE QUITTING NOW!")
                sys.exit(1)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        round_count = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"♾️  CONTINUOUS MODE")
        logger.info(f"   Interval: {interval_minutes} min between rounds")
        logger.info(f"   Max rounds: {'∞' if max_rounds == 0 else max_rounds}")
        logger.info(f"   Stop: Ctrl+C or kill -TERM")
        logger.info(f"{'='*60}")

        while not _stop:
            round_count += 1
            if max_rounds > 0 and round_count > max_rounds:
                logger.info(f"🏁 Reached max rounds ({max_rounds}). Stopping.")
                break

            logger.info(f"\n{'#'*60}")
            logger.info(
                f"# CONTINUOUS ROUND {round_count}"
                f"{f'/{max_rounds}' if max_rounds else ''}"
                f" — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            logger.info(f"{'#'*60}")

            try:
                self.run_round()

                stats = self.feedback.get_statistics()
                logger.info(f"\n📈 Running totals:")
                logger.info(f"   Rounds completed: {round_count}")
                logger.info(f"   Total evaluations: {stats['total']}")
                logger.info(f"   Buckets: {stats.get('buckets', {})}")
                if stats.get("best_candidates"):
                    best = stats["best_candidates"][0]
                    logger.info(
                        f"   Best so far: Sharpe={best['sharpe']:.2f} "
                        f"Fitness={best['fitness']:.2f}"
                    )

            except KeyboardInterrupt:
                logger.info("\n🛑 KeyboardInterrupt. Saving state and exiting.")
                break
            except Exception as e:
                logger.error(f"\n❌ Round {round_count} FAILED: {e}")
                logger.error("   Saving state and continuing after interval...")
                import traceback
                traceback.print_exc()

            # Save state after every round (success or failure)
            try:
                self.seed_pool.save()
                self.feedback.save()
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

            if _stop:
                break

            # Wait for next round
            logger.info(
                f"\n💤 Sleeping {interval_minutes} min until "
                f"{(datetime.now() + __import__('datetime').timedelta(minutes=interval_minutes)).strftime('%H:%M:%S')}..."
            )
            for _ in range(interval_minutes * 60):
                if _stop:
                    break
                time.sleep(1)

        # Final state save
        self.seed_pool.save()
        self.feedback.save()
        logger.info(f"\n✅ Agent stopped. {round_count} rounds completed.")
        logger.info(f"   Seed pool: {len(self.seed_pool)} seeds")
        logger.info(f"   Feedback records: {len(self.feedback.records)}")

    def _print_round_summary(self, df: pd.DataFrame, round_id: int):
        """Print summary for a single round."""
        if df.empty:
            return

        logger.info(f"\n{'─'*50}")
        logger.info(f"📊 ROUND {round_id} SUMMARY")
        logger.info(f"{'─'*50}")

        print(f"\nStatus distribution:")
        print(df["bucket"].value_counts().to_string())

        passed = df[df["bucket"].isin(["landed", "strong_not_landed"])]
        if not passed.empty:
            print(f"\n🏆 Best candidates this round:")
            print(
                passed.sort_values("fitness", ascending=False)[
                    ["expression", "sharpe", "fitness", "turnover", "bucket"]
                ].to_string(index=False)
            )

        print(f"\nMutation mode performance:")
        for mode in df["mutation_mode"].unique():
            mode_df = df[df["mutation_mode"] == mode]
            good = mode_df[mode_df["bucket"].isin(["landed", "strong_not_landed"])]
            print(f"  {mode}: {len(good)}/{len(mode_df)} passed")

    def _print_final_summary(self, df: pd.DataFrame):
        """Print final evolution summary."""
        if df.empty:
            return

        stats = self.feedback.get_statistics()

        print(f"\n{'='*60}")
        print(f"📈 FINAL EVOLUTION REPORT")
        print(f"{'='*60}")
        print(f"Total candidates evaluated: {len(df)}")
        print(f"Seed pool size: {len(self.seed_pool)}")
        print(f"\nBucket distribution:")
        print(df["bucket"].value_counts().to_string())

        if stats.get("best_candidates"):
            print(f"\n🏆 Top 5 candidates across all rounds:")
            for i, c in enumerate(stats["best_candidates"], 1):
                print(
                    f"  {i}. Sharpe={c['sharpe']:.2f} Fitness={c['fitness']:.2f} "
                    f"[{c['bucket']}] {c['expression']}"
                )

        print(f"\nMutation mode success rates:")
        for mode, rate in stats.get("mutation_mode_success", {}).items():
            print(f"  {mode}: {rate}")

        print(f"\nTag pair success rates:")
        for tag, rate in stats.get("tag_pair_success", {}).items():
            print(f"  {tag}: {rate}")

    @staticmethod
    def _template_generate(pair: Pair, modes: List[str]) -> List[HybridCandidate]:
        """Template-based fallback when LLM is not available."""
        a = pair.a.expression
        b = pair.b.expression
        templates = {
            "weak_gate_hybrid": (
                f"trade_when(rank({b}) > 0.55, {a}, -1)",
                "Use B as a weak gate while preserving A as the main signal."
            ),
            "regime_hybrid": (
                f"if_else(ts_rank({b}, 20) > 0.5, {a}, 0.5 * {a})",
                "Use B as a regime condition to reduce unstable exposure."
            ),
            "mild_corr_breaker": (
                f"group_neutralize({a} * (1 + 0.1 * rank({b})), sector)",
                "Use B as a mild modifier to break correlation without destroying A."
            ),
            "yield_plus_improvement": (
                f"group_rank({a} + 0.5 * rank({b}), subindustry)",
                "Combine A with a weak B component in subindustry-ranked output."
            ),
            "conditional_activation": (
                f"if_else(rank({b}) > 0.3, group_rank({a}, subindustry), 0)",
                "Activate A only when B exceeds a threshold."
            ),
        }
        candidates = []
        for mode in modes:
            if mode in templates:
                expr, hyp = templates[mode]
                candidates.append(HybridCandidate(
                    expression=expr,
                    mutation_mode=mode,
                    hypothesis=hyp,
                    parent_a_expr=a,
                    parent_b_expr=b,
                    tag_pair=pair.tag_pair,
                    pair_id=pair.pair_id,
                ))
        return candidates

    @staticmethod
    def _fake_simulate(candidate: HybridCandidate) -> SimulationResult:
        """Fake simulation for dry-run / testing mode."""
        random.seed(hash(candidate.expression) % 2**32)

        tag_pair = candidate.tag_pair
        mode = candidate.mutation_mode

        base = 0.5
        if "fundamental" in tag_pair:
            base += 0.30
        if "analyst" in tag_pair:
            base += 0.25
        if "price" in tag_pair:
            base += 0.15
        if "volume" in tag_pair:
            base += 0.10

        if mode in ("yield_plus_improvement", "mild_corr_breaker"):
            base += 0.20
        if mode == "regime_hybrid":
            base += 0.10

        sharpe = round(random.gauss(base, 0.45), 3)
        fitness = round(sharpe * random.uniform(0.5, 1.5), 3)
        turnover = round(random.uniform(3, 50), 2)

        failed = []
        if sharpe < 0.75:
            failed.append("LOW_SHARPE")
        if fitness < 0.6:
            failed.append("LOW_FITNESS")
        if turnover > 50:
            failed.append("HIGH_TURNOVER")
        if random.random() < 0.15:
            failed.append("SELF_CORRELATION")

        return SimulationResult(
            expression=candidate.expression,
            sharpe=sharpe,
            fitness=fitness,
            turnover=turnover,
            returns=round(random.uniform(2, 12), 2),
            status="PASS" if not failed else "FAIL",
            failed_checks=failed,
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evolutionary Alpha Mining Agent"
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="Number of evolutionary rounds (default: 2)"
    )
    parser.add_argument(
        "--pairs-per-round", type=int, default=6,
        help="Number of A-B parent pairs per round (default: 6)"
    )
    parser.add_argument(
        "--modes-per-pair", type=int, default=2,
        help="Number of mutation modes per pair (default: 2)"
    )
    parser.add_argument(
        "--credentials", type=str, default="credentials.json",
        help="Path to WQ Brain credentials JSON"
    )
    parser.add_argument(
        "--gemini-model", type=str, default="gemini-3-flash-preview",
        help="Gemini model name (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run with fake simulation (no WQ Brain calls)"
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Run continuously (infinite loop with interval between rounds)"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Minutes between rounds in continuous mode (default: 30)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start fresh — clear saved seed pool and feedback memory"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the candidate seed validation gate on startup "
             "(use --fresh + this combo to re-validate from scratch later)"
    )
    parser.add_argument(
        "--validation-min-batch", type=int, default=10,
        help="Only run validation when pending candidates >= this number "
             "(default: 10). Lets the research agent accumulate candidates "
             "before paying simulation cost."
    )
    parser.add_argument(
        "--validate-now", action="store_true",
        help="Force validation of all pending candidates this startup, "
             "regardless of batch threshold."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    from .color_log import apply_color_logging
    apply_color_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file="data/agent.log" if args.continuous else None,
        noisy_loggers=[
            "httpx", "httpcore", "urllib3", "google.auth",
            "google.genai", "google.api_core",
        ],
    )

    # Run the agent
    agent = AlphaMiningAgent(
        credentials_path=args.credentials,
        gemini_model=args.gemini_model,
        pairs_per_round=args.pairs_per_round,
        modes_per_pair=args.modes_per_pair,
        dry_run=args.dry_run,
        fresh=args.fresh,
        validate_candidates=not args.skip_validation,
        validation_min_batch=args.validation_min_batch,
        force_validate_now=args.validate_now,
    )

    if args.continuous:
        # Infinite loop mode
        agent.run_continuous(
            interval_minutes=args.interval,
            max_rounds=args.rounds if args.rounds != 2 else 0,
        )
    else:
        # Fixed rounds mode
        result_df = agent.run_evolution(n_rounds=args.rounds)

        if not result_df.empty:
            print(f"\n✅ Evolution complete. {len(result_df)} candidates evaluated.")
        else:
            print(f"\n⚠️ No results generated.")


if __name__ == "__main__":
    main()

