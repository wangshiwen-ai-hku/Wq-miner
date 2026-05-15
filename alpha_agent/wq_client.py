"""
WorldQuant Brain API Client.

Adapted from WQ-Brain by RussellDash332.
Handles authentication, simulation, result checking, and alpha submission.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ModuleNotFoundError:
    requests = None
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Structured result from a WQ Brain simulation."""
    expression: str
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns: float = 0.0
    drawdown: float = 0.0
    margin: float = 0.0
    status: str = "UNKNOWN"          # PASS / FAIL / ERROR
    alpha_id: str = ""
    alpha_link: str = ""
    checks: Dict[str, Any] = field(default_factory=dict)
    failed_checks: List[str] = field(default_factory=list)
    error_message: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed_all(self) -> bool:
        return self.status == "PASS" and len(self.failed_checks) == 0

    def summary(self) -> str:
        return (
            f"Sharpe={self.sharpe:.2f} | Fitness={self.fitness:.2f} | "
            f"Turnover={self.turnover:.1f}% | Status={self.status} | "
            f"Failed={', '.join(self.failed_checks) if self.failed_checks else 'None'}"
        )


# Default simulation settings matching user's proven configuration
DEFAULT_SETTINGS = {
    "nanHandling": "ON",
    "instrumentType": "EQUITY",
    "delay": 1,
    "universe": "TOP3000",
    "truncation": 0.08,
    "unitHandling": "VERIFY",
    "pasteurization": "ON",
    "region": "USA",
    "language": "FASTEXPR",
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "visualization": False,
}


class WQBrainClient:
    """
    Client for WorldQuant Brain API.

    Handles login, simulation submission, result polling, and alpha checks.
    """

    BASE_URL = "https://api.worldquantbrain.com"

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        settings: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        poll_interval: float = 10.0,
    ):
        if requests is None:
            raise ModuleNotFoundError(
                "requests is required for WQ Brain API access. "
                "Install project dependencies with `pip install -r requirements.txt`."
            )
        self.session = requests.Session()
        self.credentials_path = Path(credentials_path)
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self._logged_in = False

    def login(self) -> bool:
        """Authenticate with WQ Brain."""
        creds = json.loads(self.credentials_path.read_text())
        email, password = creds["email"], creds["password"]
        self.session.auth = (email, password)

        r = self.session.post(f"{self.BASE_URL}/authentication")
        data = r.json()

        if "user" in data:
            self._logged_in = True
            logger.info("✅ Logged in to WQ Brain successfully.")
            return True

        if "inquiry" in data:
            logger.warning(
                f"⚠️ Biometric auth required at: "
                f"{r.url}/persona?inquiry={data['inquiry']}"
            )
            input("Complete biometric auth, then press Enter to continue...")
            self.session.post(f"{r.url}/persona", json=data)
            self._logged_in = True
            return True

        logger.error(f"❌ Login failed: {data}")
        return False

    def _ensure_login(self):
        if not self._logged_in:
            if not self.login():
                raise RuntimeError("Cannot connect to WQ Brain")

    def _safe_get(self, url: str, **kwargs) -> requests.Response:
        """GET with retry."""
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, **kwargs)
                return r
            except Exception as e:
                logger.warning(f"GET retry {attempt+1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt)
        raise ConnectionError(f"Failed GET after {self.max_retries} retries: {url}")

    def _safe_post(self, url: str, **kwargs) -> requests.Response:
        """POST with retry."""
        for attempt in range(self.max_retries):
            try:
                r = self.session.post(url, **kwargs)
                return r
            except Exception as e:
                logger.warning(f"POST retry {attempt+1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt)
        raise ConnectionError(f"Failed POST after {self.max_retries} retries: {url}")

    def simulate(
        self,
        expression: str,
        settings_override: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """
        Submit an alpha expression for simulation and wait for results.

        Args:
            expression: The alpha formula to simulate.
            settings_override: Optional dict to override default settings.

        Returns:
            SimulationResult with all metrics.
        """
        self._ensure_login()

        sim_settings = {**self.settings, **(settings_override or {})}
        result = SimulationResult(expression=expression, settings=sim_settings)

        # --- Step 1: Submit simulation ---
        logger.info(f"📤 Submitting simulation: {expression[:80]}...")

        for attempt in range(self.max_retries * 5):
            try:
                r = self._safe_post(
                    f"{self.BASE_URL}/simulations",
                    json={
                        "regular": expression,
                        "type": "REGULAR",
                        "settings": sim_settings,
                    },
                )

                if "Location" in r.headers:
                    sim_url = r.headers["Location"]
                    break
                else:
                    resp_data = r.json()
                    if "credentials" in str(resp_data):
                        logger.error("Login expired, re-authenticating...")
                        self._logged_in = False
                        self._ensure_login()
                    elif "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in str(resp_data):
                        logger.warning(f"Concurrency limit hit, waiting 15s to retry (attempt {attempt+1})...")
                        time.sleep(15)
                        continue
                    else:
                        result.status = "ERROR"
                        result.error_message = str(resp_data)
                        logger.error(f"Simulation rejected: {resp_data}")
                        return result
            except Exception as e:
                logger.warning(f"Submit retry {attempt+1}: {e}")
                time.sleep(5)
        else:
            result.status = "ERROR"
            result.error_message = "Failed to submit simulation"
            return result

        # --- Step 2: Poll for completion ---
        logger.info(f"⏳ Polling simulation: {sim_url}")
        alpha_link = None

        while True:
            try:
                r = self._safe_get(sim_url).json()
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(self.poll_interval)
                continue

            if "alpha" in r:
                alpha_link = r["alpha"]
                break

            if "progress" in r:
                progress = int(100 * r["progress"])
                logger.info(f"  ... simulation progress: {progress}%")
            elif "message" in r:
                result.status = "ERROR"
                result.error_message = r["message"]
                logger.error(f"Simulation error: {r['message']}")
                return result

            time.sleep(self.poll_interval)

        # --- Step 3: Fetch alpha details ---
        r = self._safe_get(f"{self.BASE_URL}/alphas/{alpha_link}").json()

        result.alpha_id = alpha_link
        result.alpha_link = f"https://platform.worldquantbrain.com/alpha/{alpha_link}"
        result.sharpe = r.get("is", {}).get("sharpe", 0.0)
        result.fitness = r.get("is", {}).get("fitness", 0.0)
        result.turnover = round(100 * r.get("is", {}).get("turnover", 0.0), 2)
        result.returns = r.get("is", {}).get("returns", 0.0)
        result.drawdown = r.get("is", {}).get("drawdown", 0.0)
        result.margin = r.get("is", {}).get("margin", 0.0)

        # Parse checks
        checks = r.get("is", {}).get("checks", [])
        result.checks = {c["name"]: c for c in checks}
        result.failed_checks = [
            c["name"] for c in checks if c.get("result") in ("FAIL", "ERROR")
        ]
        result.status = "PASS" if not result.failed_checks else "FAIL"

        logger.info(f"✅ Simulation complete: {result.summary()}")
        logger.info(f"   Link: {result.alpha_link}")

        return result

    def submit_async(self, expression: str,
                     settings_override: Optional[Dict[str, Any]] = None,
                     ) -> Optional[str]:
        """
        Submit a simulation and return the poll URL immediately
        (does NOT wait for result).

        Returns:
            The simulation URL to poll, or None if submission failed.
        """
        self._ensure_login()
        sim_settings = {**self.settings, **(settings_override or {})}

        for attempt in range(self.max_retries * 5):
            try:
                r = self._safe_post(
                    f"{self.BASE_URL}/simulations",
                    json={
                        "regular": expression,
                        "type": "REGULAR",
                        "settings": sim_settings,
                    },
                )

                if "Location" in r.headers:
                    sim_url = r.headers["Location"]
                    logger.info(f"📤 Submitted: {expression[:60]}... → {sim_url}")
                    return sim_url
                else:
                    resp_data = r.json()
                    if "credentials" in str(resp_data):
                        logger.error("Login expired, re-authenticating...")
                        self._logged_in = False
                        self._ensure_login()
                    elif "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in str(resp_data):
                        logger.warning(f"Concurrency limit hit, waiting 15s to retry (attempt {attempt+1})...")
                        time.sleep(15)
                        continue
                    else:
                        logger.error(f"Submission rejected: {resp_data}")
                        return None
            except Exception as e:
                logger.warning(f"Submit retry {attempt+1}: {e}")
                time.sleep(5)

        return None

    def poll_result(self, expression: str, sim_url: str,
                    timeout: float = 300.0,
                    ) -> SimulationResult:
        """
        Poll a previously submitted simulation until it completes.

        Args:
            expression: The original expression (for the result).
            sim_url: The simulation URL from submit_async.
            timeout: Maximum seconds to wait.

        Returns:
            SimulationResult.
        """
        result = SimulationResult(expression=expression, settings=self.settings)
        start = time.time()

        while time.time() - start < timeout:
            try:
                r = self._safe_get(sim_url).json()
            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(self.poll_interval)
                continue

            if "alpha" in r:
                alpha_link = r["alpha"]
                return self._fetch_alpha_details(result, alpha_link)

            if "progress" in r:
                progress = int(100 * r["progress"])
                elapsed = int(time.time() - start)
                logger.debug(
                    f"   [{expression[:40]}...] {progress}% ({elapsed}s)"
                )
            elif "message" in r:
                result.status = "ERROR"
                result.error_message = r["message"]
                logger.error(f"Simulation error: {r['message']}")
                return result

            time.sleep(self.poll_interval)

        result.status = "ERROR"
        result.error_message = f"Timeout after {timeout}s"
        return result

    def _fetch_alpha_details(self, result: SimulationResult,
                              alpha_link: str) -> SimulationResult:
        """Fetch full alpha metrics from a completed simulation."""
        r = self._safe_get(f"{self.BASE_URL}/alphas/{alpha_link}").json()

        result.alpha_id = alpha_link
        result.alpha_link = (
            f"https://platform.worldquantbrain.com/alpha/{alpha_link}"
        )
        result.sharpe = r.get("is", {}).get("sharpe", 0.0)
        result.fitness = r.get("is", {}).get("fitness", 0.0)
        result.turnover = round(100 * r.get("is", {}).get("turnover", 0.0), 2)
        result.returns = r.get("is", {}).get("returns", 0.0)
        result.drawdown = r.get("is", {}).get("drawdown", 0.0)
        result.margin = r.get("is", {}).get("margin", 0.0)

        checks = r.get("is", {}).get("checks", [])
        result.checks = {c["name"]: c for c in checks}
        result.failed_checks = [
            c["name"] for c in checks if c.get("result") in ("FAIL", "ERROR")
        ]
        result.status = "PASS" if not result.failed_checks else "FAIL"

        logger.info(f"✅ {result.expression[:50]}... → {result.summary()}")
        return result

    def simulate_batch(
        self,
        expressions: List[str],
        max_concurrent: int = 3,
        timeout_per_sim: float = 300.0,
    ) -> List[SimulationResult]:
        """
        Submit and poll multiple simulations in parallel.

        Flow:
        1. Use ThreadPoolExecutor to run both submit and poll
        2. This naturally controls both submission concurrency and polling concurrency.

        Args:
            expressions: List of alpha expressions to simulate.
            max_concurrent: Max concurrent simulations (don't hammer the API).
            timeout_per_sim: Timeout per individual simulation.

        Returns:
            List of SimulationResults in same order as input.
        """
        self._ensure_login()

        logger.info(f"\n🚀 Batch running {len(expressions)} simulations (max {max_concurrent} concurrent)...")
        results = [None] * len(expressions)

        def _run_one(idx: int, expr: str) -> tuple:
            # Random jitter to prevent perfectly synchronized submissions initially
            time.sleep(idx * 0.5 % 2.0)
            
            sim_url = self.submit_async(expr)
            if not sim_url:
                return idx, SimulationResult(
                    expression=expr,
                    status="ERROR",
                    error_message="Failed to submit",
                )
            
            result = self.poll_result(expr, sim_url, timeout=timeout_per_sim)
            return idx, result

        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {}
            for idx, expr in enumerate(expressions):
                f = pool.submit(_run_one, idx, expr)
                futures[f] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, result = future.result()
                    results[idx] = result
                    done_count = sum(1 for r in results if r is not None)
                    logger.info(
                        f"   📊 [{done_count}/{len(expressions)}] completed"
                    )
                except Exception as e:
                    results[idx] = SimulationResult(
                        expression=expressions[idx],
                        status="ERROR",
                        error_message=str(e),
                    )
                    logger.error(f"   ❌ Run error for #{idx}: {e}")

        return results

    def check_alpha(self, alpha_id: str) -> Dict[str, Any]:
        """Run full check on an alpha (including self-correlation)."""
        self._ensure_login()

        r = self._safe_get(f"{self.BASE_URL}/alphas/{alpha_id}/check")
        while True:
            if r.content:
                try:
                    return r.json()
                except Exception:
                    pass
            time.sleep(5)
            r = self._safe_get(f"{self.BASE_URL}/alphas/{alpha_id}/check")

    def get_self_correlation(self, alpha_id: str, max_attempts: int = 8) -> float:
        """
        Get the self-correlation value for an alpha via the dedicated endpoint.
        Retries with increasing backoff — the value may take time to compute
        after simulation completes.
        """
        self._ensure_login()

        for attempt in range(max_attempts):
            wait = min(5 * (attempt + 1), 30)   # 5s, 10s, 15s … capped at 30s
            r = self._safe_get(
                f"{self.BASE_URL}/alphas/{alpha_id}/correlations/self"
            )
            if r.content:
                try:
                    records = r.json().get("records", [])
                    if records:
                        value = max(record[5] for record in records)
                        logger.debug(
                            f"   self_corr via /correlations/self "
                            f"(attempt {attempt+1}): {value:.4f}"
                        )
                        return value
                except Exception as e:
                    logger.warning(f"   Correlation parse error (attempt {attempt+1}): {e}")
            if attempt < max_attempts - 1:
                logger.debug(
                    f"   self_corr not ready yet (attempt {attempt+1}/{max_attempts}), "
                    f"waiting {wait}s..."
                )
                time.sleep(wait)

        logger.warning(f"   ⚠️ Could not retrieve self_corr after {max_attempts} attempts")
        return -1.0

    def _get_team_id(self) -> Optional[str]:
        """Get the user's active team ID (required for performance check)."""
        self._ensure_login()
        try:
            r = self._safe_get(
                f"{self.BASE_URL}/users/self/teams",
                params={
                    "status": "ACTIVE",
                    "members.self.status": "ACCEPTED",
                    "order": "-dateCreated",
                },
            ).json()
            results = r.get("results", [])
            if results:
                team_id = results[0]["id"]
                logger.debug(f"Team ID: {team_id}")
                return team_id
        except Exception as e:
            logger.warning(f"Failed to get team ID: {e}")
        return None

    def check_submission_readiness(
        self,
        alpha_id: str,
        max_self_corr: float = 0.7,
        min_performance_gain: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Run all pre-submission checks for an alpha.

        Checks:
        1. All non-self-correlation IS checks pass (from /alphas/{id}/check)
        2. Performance after > before + min_performance_gain

        Self-correlation is recorded when present in the first /check response,
        but it is not retried and does not block readiness. In practice the
        performance contribution endpoint is the cheaper, stronger gate for
        whether a PASS alpha is useful enough to enter the seed pool.

        Returns:
            Dict with keys:
            - ready: bool
            - self_corr: float
            - performance_before: float
            - performance_after: float
            - performance_gain: float
            - all_checks_passed: bool
            - reason: str (if not ready)
        """
        self._ensure_login()

        result = {
            "ready": False,
            "self_corr": -1.0,
            "performance_before": 0.0,
            "performance_after": 0.0,
            "performance_gain": 0.0,
            "all_checks_passed": False,
            "reason": "",
        }
        reasons = []
        performance_available = False

        # ── Step 1: Run /check and verify all IS checks pass ──
        logger.info(f"   🔍 Running IS checks for {alpha_id}...")
        checks = []
        for attempt in range(self.max_retries):
            check_r = self._safe_get(
                f"{self.BASE_URL}/alphas/{alpha_id}/check"
            )
            if check_r.content:
                try:
                    checks = check_r.json().get("is", {}).get("checks", [])
                    if checks:
                        break
                except Exception:
                    pass
            time.sleep(5)

        # Check all non-self-correlation IS checks pass. SELF_CORRELATION is
        # noisy/slow to populate after simulation, so performance gain below
        # decides whether a PASS alpha is useful for the seed pool.
        failed = [
            c["name"] for c in checks
            if c.get("result") in ("FAIL", "ERROR")
            and c.get("name") != "SELF_CORRELATION"
        ]
        if not checks:
            reasons.append("Failed to retrieve IS checks")
        elif failed:
            reasons.append(f"IS checks failed: {', '.join(failed)}")
        else:
            result["all_checks_passed"] = True

        # Extract self-correlation from the initial IS check results only.
        # Do not retry /check and do not call /correlations/self here; if
        # self-corr is missing, leave -1.0 and continue to performance.
        def _extract_self_corr_from_checks(chks) -> float:
            for c in chks:
                if c["name"] == "SELF_CORRELATION":
                    v = c.get("value")
                    if v is not None:
                        return float(v)
            return -1.0

        result["self_corr"] = _extract_self_corr_from_checks(checks)
        if result["self_corr"] >= 0:
            logger.info(f"   📊 Self-correlation: {result['self_corr']:.4f} (recorded only)")
        else:
            logger.info("   📊 Self-correlation: unavailable in initial /check; skipping retry")

        # ── Step 2: Check performance (before vs after) ──
        team_id = self._get_team_id()
        if team_id is None:
            reasons.append("Cannot get team ID for performance check")
        else:
            logger.info(f"   📈 Checking performance contribution...")
            for attempt in range(self.max_retries + 2):
                perf_r = self._safe_get(
                    f"{self.BASE_URL}/teams/{team_id}/alphas/{alpha_id}"
                    f"/before-and-after-performance"
                )
                if perf_r.content:
                    try:
                        score = perf_r.json().get("score", {})
                        if "before" in score and "after" in score:
                            result["performance_before"] = float(score["before"])
                            result["performance_after"] = float(score["after"])
                            result["performance_gain"] = (
                                result["performance_after"]
                                - result["performance_before"]
                            )
                            performance_available = True
                            break
                    except Exception:
                        pass
                time.sleep(5)
            else:
                reasons.append("Failed to retrieve performance data")

            logger.info(
                f"   📈 Performance: {result['performance_before']:.1f} → "
                f"{result['performance_after']:.1f} "
                f"(+{result['performance_gain']:.1f})"
            )

        if performance_available and result["performance_gain"] < min_performance_gain:
            reasons.append(
                f"Performance gain {result['performance_gain']:.1f} "
                f"< {min_performance_gain}"
            )

        # ── All checks passed ──
        result["ready"] = not reasons
        result["reason"] = "; ".join(reasons)
        return result

    def submit_alpha(self, alpha_id: str, skip_checks: bool = False) -> bool:
        """
        Submit an alpha for production, with pre-submission safety checks.

        Checks before submitting:
        1. All non-self-correlation IS checks must pass
        2. Performance contribution > 100

        Self-correlation is recorded when immediately available, but it is not
        retried here; the platform still enforces its own final submission
        rules when the alpha is submitted.

        Args:
            alpha_id: The alpha ID to submit.
            skip_checks: If True, skip pre-submission checks (not recommended).

        Returns:
            True if successfully submitted.
        """
        self._ensure_login()

        if not skip_checks:
            logger.info(f"\n🔒 Pre-submission safety checks for {alpha_id}...")

            readiness = self.check_submission_readiness(alpha_id)

            if not readiness["ready"]:
                logger.warning(
                    f"   ❌ NOT READY to submit: {readiness['reason']}"
                )
                logger.info(
                    f"   Self-corr: {readiness['self_corr']:.4f} | "
                    f"Perf gain: {readiness['performance_gain']:.1f} | "
                    f"Checks passed: {readiness['all_checks_passed']}"
                )
                return False

            logger.info(
                f"   ✅ All pre-submit gates passed!\n"
                f"   Self-corr: {readiness['self_corr']:.4f} (<0.7 ✓)\n"
                f"   Perf gain: +{readiness['performance_gain']:.1f} (>100 ✓)"
            )

        # Actually submit
        self._safe_post(f"{self.BASE_URL}/alphas/{alpha_id}/submit")
        logger.info(f"📬 Submitting alpha {alpha_id}...")

        while True:
            r = self._safe_get(f"{self.BASE_URL}/alphas/{alpha_id}/submit")
            if r.status_code == 404:
                logger.info("Alpha already submitted.")
                return False
            if r.content:
                try:
                    checks = r.json().get("is", {}).get("checks", [])
                    for check in checks:
                        if check["name"] == "SELF_CORRELATION":
                            passed = check["result"] == "PASS"
                            logger.info(
                                f"Submit result: {'✅ PASS' if passed else '❌ FAIL'} "
                                f"(self-corr check)"
                            )
                            return passed
                    break
                except Exception:
                    pass
            time.sleep(5)

        return False

    def get_existing_alphas(
        self,
        limit: int = 50,
        min_sharpe: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Fetch existing unsubmitted alphas for correlation avoidance."""
        self._ensure_login()

        url = (
            f"{self.BASE_URL}/users/self/alphas"
            f"?limit={limit}&offset=0"
            f"&stage=IS%1fOS"
            f"&is.sharpe%3E={min_sharpe}"
            f"&order=-dateCreated"
            f"&hidden=false"
        )

        r = self._safe_get(url).json()
        results = r.get("results", [])

        alphas = []
        for a in results:
            alphas.append({
                "id": a["id"],
                "expression": a.get("regular", {}).get("code", ""),
                "sharpe": a.get("is", {}).get("sharpe", 0),
                "fitness": a.get("is", {}).get("fitness", 0),
                "status": a.get("status", ""),
            })

        logger.info(f"📋 Fetched {len(alphas)} existing alphas.")
        return alphas
