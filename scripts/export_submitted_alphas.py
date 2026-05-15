#!/usr/bin/env python3
"""Export submitted WorldQuant Brain alphas to CSV."""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import http.cookiejar
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "https://api.worldquantbrain.com"


def request_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    retries: int = 3,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                method=method,
                headers=headers or {"Accept": "application/json;version=2.0"},
            )
            with opener.open(req, timeout=45) as resp:
                body = resp.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                last_error = exc
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Request failed: {url}") from last_error


def login(credentials_path: Path) -> urllib.request.OpenerDirector:
    creds = json.loads(credentials_path.read_text())
    auth = base64.b64encode(
        f"{creds['email']}:{creds['password']}".encode("utf-8")
    ).decode("ascii")
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )
    data = request_json(
        opener,
        f"{BASE_URL}/authentication",
        method="POST",
        headers={"Authorization": f"Basic {auth}", "Accept": "application/json"},
    )
    if not isinstance(data, dict) or "user" not in data:
        raise RuntimeError(f"Authentication did not complete: {data}")
    return opener


def get_active_team_id(opener: urllib.request.OpenerDirector) -> str:
    query = urllib.parse.urlencode(
        {
            "status": "ACTIVE",
            "members.self.status": "ACCEPTED",
            "order": "-dateCreated",
        }
    )
    data = request_json(opener, f"{BASE_URL}/users/self/teams?{query}")
    results = data.get("results", []) if isinstance(data, dict) else []
    return results[0].get("id", "") if results else ""


def fetch_submitted_alpha_summaries(
    opener: urllib.request.OpenerDirector,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    first_query = urllib.parse.urlencode(
        {
            "limit": limit,
            "offset": 0,
            "order": "-dateSubmitted",
            "hidden": "false",
            "stage": "OS",
            "status": "ACTIVE",
        }
    )
    first = request_json(opener, f"{BASE_URL}/users/self/alphas?{first_query}")
    total = int(first.get("count", 0))
    results = list(first.get("results", []))

    for offset in range(limit, total, limit):
        query = urllib.parse.urlencode(
            {
                "limit": limit,
                "offset": offset,
                "order": "-dateSubmitted",
                "hidden": "false",
                "stage": "OS",
                "status": "ACTIVE",
            }
        )
        page = request_json(opener, f"{BASE_URL}/users/self/alphas?{query}")
        results.extend(page.get("results", []))
    return results


def check_summary(section: dict[str, Any] | None) -> str:
    checks = (section or {}).get("checks", [])
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


def metric(section: dict[str, Any] | None, key: str) -> Any:
    return (section or {}).get(key, "")


def safe_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def fetch_performance(
    opener: urllib.request.OpenerDirector,
    team_id: str,
    alpha_id: str,
) -> dict[str, Any]:
    empty = {
        "performance_before": "",
        "performance_after": "",
        "performance_gain": "",
        "performance_score_json": "",
        "performance_error": "",
    }
    if not team_id:
        return {**empty, "performance_error": "No active team id found"}
    url = f"{BASE_URL}/teams/{team_id}/alphas/{alpha_id}/before-and-after-performance"
    try:
        data = request_json(opener, url, retries=2)
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", "ignore").strip()
        return {**empty, "performance_error": f"HTTP {exc.code}: {message}"}
    except Exception as exc:
        return {**empty, "performance_error": str(exc)}
    score = data.get("score", {}) if isinstance(data, dict) else {}
    before = score.get("before")
    after = score.get("after")
    out: dict[str, Any] = {
        "performance_before": before if before is not None else "",
        "performance_after": after if after is not None else "",
        "performance_score_json": safe_json(score),
        "performance_error": "",
    }
    if before is not None and after is not None:
        out["performance_gain"] = float(after) - float(before)
    else:
        out["performance_gain"] = ""
    return out


def flatten_alpha(detail: dict[str, Any], performance: dict[str, Any]) -> dict[str, Any]:
    settings = detail.get("settings") or {}
    regular = detail.get("regular") or {}
    is_section = detail.get("is") or {}
    os_section = detail.get("os") or {}
    prod_section = detail.get("prod") or {}
    train_section = detail.get("train") or {}
    test_section = detail.get("test") or {}
    row: dict[str, Any] = {
        "id": detail.get("id", ""),
        "author": detail.get("author", ""),
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
        "settings_json": safe_json(settings),
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
        "is_sharpe": metric(is_section, "sharpe"),
        "is_fitness": metric(is_section, "fitness"),
        "is_turnover": metric(is_section, "turnover"),
        "is_returns": metric(is_section, "returns"),
        "is_drawdown": metric(is_section, "drawdown"),
        "is_margin": metric(is_section, "margin"),
        "is_pnl": metric(is_section, "pnl"),
        "is_bookSize": metric(is_section, "bookSize"),
        "is_longCount": metric(is_section, "longCount"),
        "is_shortCount": metric(is_section, "shortCount"),
        "is_selfCorrelation": metric(is_section, "selfCorrelation"),
        "is_prodCorrelation": metric(is_section, "prodCorrelation"),
        "is_startDate": metric(is_section, "startDate"),
        "is_checks": check_summary(is_section),
        "os_startDate": metric(os_section, "startDate"),
        "os_osISSharpeRatio": metric(os_section, "osISSharpeRatio"),
        "os_preCloseSharpeRatio": metric(os_section, "preCloseSharpeRatio"),
        "os_checks": check_summary(os_section),
        "train_json": safe_json(train_section),
        "test_json": safe_json(test_section),
        "prod_json": safe_json(prod_section),
        "tags": safe_json(detail.get("tags")),
        "classifications": safe_json(detail.get("classifications")),
        "competitions": safe_json(detail.get("competitions")),
        "themes": safe_json(detail.get("themes")),
        "pyramids": safe_json(detail.get("pyramids")),
        "team": safe_json(detail.get("team")),
    }
    row.update(performance)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials", default="credentials.json")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    credentials_path = Path(args.credentials)
    opener = login(credentials_path)
    team_id = get_active_team_id(opener)
    summaries = fetch_submitted_alpha_summaries(opener)

    rows: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries, start=1):
        alpha_id = summary["id"]
        print(f"[{index}/{len(summaries)}] fetching {alpha_id}", flush=True)
        detail = request_json(opener, f"{BASE_URL}/alphas/{alpha_id}")
        performance = fetch_performance(opener, team_id, alpha_id)
        rows.append(flatten_alpha(detail, performance))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or f"data/submitted_alphas_{timestamp}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} submitted alphas to {output}")


if __name__ == "__main__":
    main()
