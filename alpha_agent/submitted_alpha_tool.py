"""Tool for recording submitted WorldQuant Brain alphas.

This module can be imported by the mining agent or run directly:

    python -m alpha_agent.submitted_alpha_tool --credentials credentials.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import http.cookiejar
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


BASE_URL = "https://api.worldquantbrain.com"
DEFAULT_OUTPUT = "data/submitted_alphas.csv"


VISIBLE_COLUMNS = [
    "id",
    "name",
    "type",
    "status",
    "stage",
    "grade",
    "dateCreated",
    "dateSubmitted",
    "dateModified",
    "alpha_url",
    "expression",
    "description",
    "operatorCount",
    "settings_instrumentType",
    "settings_region",
    "settings_universe",
    "settings_delay",
    "settings_decay",
    "settings_neutralization",
    "settings_truncation",
    "settings_pasteurization",
    "settings_unitHandling",
    "settings_nanHandling",
    "settings_maxTrade",
    "settings_maxPosition",
    "settings_language",
    "settings_visualization",
    "settings_startDate",
    "settings_endDate",
    "is_sharpe",
    "is_fitness",
    "is_turnover",
    "is_returns",
    "is_drawdown",
    "is_margin",
    "is_pnl",
    "is_bookSize",
    "is_longCount",
    "is_shortCount",
    "is_selfCorrelation",
    "is_prodCorrelation",
    "is_startDate",
    "is_checks",
    "os_startDate",
    "os_osISSharpeRatio",
    "os_preCloseSharpeRatio",
    "os_checks",
    "tags",
    "classifications",
    "competitions",
    "team_id",
    "team_name",
    "team_type",
]


class SubmittedAlphaRecorder:
    """Fetch and save the current user's submitted alphas."""

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        session: Optional[Any] = None,
    ):
        self.credentials_path = Path(credentials_path)
        self.session = session
        self.opener = None
        if session is None:
            self.opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
            )
        self._logged_in = session is not None

    def login(self) -> None:
        """Authenticate the owned session with WorldQuant Brain."""
        if self._logged_in:
            return

        creds = json.loads(self.credentials_path.read_text())
        if self.session is not None:
            self.session.auth = (creds["email"], creds["password"])
            response = self.session.post(f"{BASE_URL}/authentication", timeout=45)
            response.raise_for_status()
            data = response.json()
        else:
            auth = base64.b64encode(
                f"{creds['email']}:{creds['password']}".encode("utf-8")
            ).decode("ascii")
            data = self._urllib_json(
                f"{BASE_URL}/authentication",
                method="POST",
                headers={
                    "Authorization": f"Basic {auth}",
                    "Accept": "application/json",
                },
            )

        if "user" in data:
            self._logged_in = True
            return

        if "inquiry" in data:
            raise RuntimeError(
                "WorldQuant Brain requires biometric authentication. "
                "Complete login in the browser, then retry."
            )

        raise RuntimeError(f"WorldQuant Brain authentication failed: {data}")

    def fetch_submitted_alphas(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return all visible submitted alphas for the current user."""
        self.login()

        first_page = self._get_alpha_page(limit=limit, offset=0)
        total = int(first_page.get("count", 0))
        summaries = list(first_page.get("results", []))

        for offset in range(limit, total, limit):
            page = self._get_alpha_page(limit=limit, offset=offset)
            summaries.extend(page.get("results", []))

        rows = []
        for summary in summaries:
            alpha_id = summary["id"]
            detail = self._get_json(f"{BASE_URL}/alphas/{alpha_id}")
            rows.append(self._visible_row(detail))
        return rows

    def save_csv(
        self,
        output_path: str | Path = DEFAULT_OUTPUT,
        limit: int = 100,
    ) -> Path:
        """Fetch submitted alphas and save only visible fields to CSV."""
        rows = self.fetch_submitted_alphas(limit=limit)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=VISIBLE_COLUMNS)
            writer.writeheader()
            writer.writerows(_clean_row(row) for row in rows)

        logger.info("Saved %d submitted alphas to %s", len(rows), output)
        return output

    def _get_alpha_page(self, limit: int, offset: int) -> dict[str, Any]:
        params = {
            "limit": limit,
            "offset": offset,
            "order": "-dateSubmitted",
            "hidden": "false",
            "stage": "OS",
            "status": "ACTIVE",
        }
        if self.session is None:
            query = urllib.parse.urlencode(params)
            return self._urllib_json(f"{BASE_URL}/users/self/alphas?{query}")

        response = self.session.get(
            f"{BASE_URL}/users/self/alphas",
            params=params,
            headers={"Accept": "application/json;version=2.0"},
            timeout=45,
        )
        response.raise_for_status()
        return response.json()

    def _get_json(self, url: str) -> dict[str, Any]:
        if self.session is None:
            return self._urllib_json(url)

        response = self.session.get(
            url,
            headers={"Accept": "application/json;version=2.0"},
            timeout=45,
        )
        response.raise_for_status()
        return response.json()

    def _urllib_json(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            method=method,
            headers=headers or {"Accept": "application/json;version=2.0"},
        )
        assert self.opener is not None
        with self.opener.open(req, timeout=45) as response:
            body = response.read().decode("utf-8")
        return json.loads(body) if body else {}

    def _visible_row(self, detail: dict[str, Any]) -> dict[str, Any]:
        settings = detail.get("settings") or {}
        regular = detail.get("regular") or {}
        is_section = detail.get("is") or {}
        os_section = detail.get("os") or {}
        team = detail.get("team") or {}

        return {
            "id": detail.get("id", ""),
            "name": detail.get("name", ""),
            "type": detail.get("type", ""),
            "status": detail.get("status", ""),
            "stage": detail.get("stage", ""),
            "grade": detail.get("grade", ""),
            "dateCreated": detail.get("dateCreated", ""),
            "dateSubmitted": detail.get("dateSubmitted", ""),
            "dateModified": detail.get("dateModified", ""),
            "alpha_url": f"https://platform.worldquantbrain.com/alpha/{detail.get('id', '')}",
            "expression": regular.get("code", ""),
            "description": regular.get("description", ""),
            "operatorCount": regular.get("operatorCount", ""),
            "settings_instrumentType": settings.get("instrumentType", ""),
            "settings_region": settings.get("region", ""),
            "settings_universe": settings.get("universe", ""),
            "settings_delay": settings.get("delay", ""),
            "settings_decay": settings.get("decay", ""),
            "settings_neutralization": settings.get("neutralization", ""),
            "settings_truncation": settings.get("truncation", ""),
            "settings_pasteurization": settings.get("pasteurization", ""),
            "settings_unitHandling": settings.get("unitHandling", ""),
            "settings_nanHandling": settings.get("nanHandling", ""),
            "settings_maxTrade": settings.get("maxTrade", ""),
            "settings_maxPosition": settings.get("maxPosition", ""),
            "settings_language": settings.get("language", ""),
            "settings_visualization": settings.get("visualization", ""),
            "settings_startDate": settings.get("startDate", ""),
            "settings_endDate": settings.get("endDate", ""),
            "is_sharpe": is_section.get("sharpe", ""),
            "is_fitness": is_section.get("fitness", ""),
            "is_turnover": is_section.get("turnover", ""),
            "is_returns": is_section.get("returns", ""),
            "is_drawdown": is_section.get("drawdown", ""),
            "is_margin": is_section.get("margin", ""),
            "is_pnl": is_section.get("pnl", ""),
            "is_bookSize": is_section.get("bookSize", ""),
            "is_longCount": is_section.get("longCount", ""),
            "is_shortCount": is_section.get("shortCount", ""),
            "is_selfCorrelation": is_section.get("selfCorrelation", ""),
            "is_prodCorrelation": is_section.get("prodCorrelation", ""),
            "is_startDate": is_section.get("startDate", ""),
            "is_checks": _check_summary(is_section.get("checks", [])),
            "os_startDate": os_section.get("startDate", ""),
            "os_osISSharpeRatio": os_section.get("osISSharpeRatio", ""),
            "os_preCloseSharpeRatio": os_section.get("preCloseSharpeRatio", ""),
            "os_checks": _check_summary(os_section.get("checks", [])),
            "tags": _names(detail.get("tags", [])),
            "classifications": _names(detail.get("classifications", [])),
            "competitions": _names(detail.get("competitions", [])),
            "team_id": team.get("id", ""),
            "team_name": team.get("name", ""),
            "team_type": team.get("type", ""),
        }


def _check_summary(checks: Iterable[dict[str, Any]]) -> str:
    parts = []
    for check in checks:
        name = check.get("name", "")
        result = check.get("result", "")
        value = check.get("value")
        if value is None:
            parts.append(f"{name}:{result}")
        else:
            parts.append(f"{name}:{result}({value})")
    return "; ".join(parts)


def _names(items: Iterable[Any]) -> str:
    names = []
    for item in items:
        if isinstance(item, dict):
            names.append(str(item.get("name") or item.get("id") or ""))
        else:
            names.append(str(item))
    return "; ".join(name for name in names if name)


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_cell(row.get(key, "")) for key in VISIBLE_COLUMNS}


def _clean_cell(value: Any) -> Any:
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def save_submitted_alphas(
    credentials_path: str = "credentials.json",
    output_path: str | Path = DEFAULT_OUTPUT,
    session: Optional[Any] = None,
    limit: int = 100,
) -> Path:
    """Convenience function used by launchers before mining starts."""
    recorder = SubmittedAlphaRecorder(
        credentials_path=credentials_path,
        session=session,
    )
    return recorder.save_csv(output_path=output_path, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save submitted WorldQuant Brain alphas to CSV."
    )
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to WorldQuant Brain credentials JSON.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Page size for WorldQuant Brain API requests.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output = save_submitted_alphas(
        credentials_path=args.credentials,
        output_path=args.output,
        limit=args.limit,
    )
    print(output)


if __name__ == "__main__":
    main()
