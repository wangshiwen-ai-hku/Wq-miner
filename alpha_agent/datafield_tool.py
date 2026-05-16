"""Natural-language WorldQuant Brain data-field search tool.

Example:
    python -m alpha_agent.datafield_tool -s "Analyst revision / short interest / options skew"
"""

from __future__ import annotations
import csv
import argparse
import base64
import hashlib
import http.cookiejar
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)

BASE_URL = "https://api.worldquantbrain.com"
DEFAULT_OUTPUT_DIR = "data/datafields"

FIELD_COLUMNS = [
    "query",
    "intent_summary",
    "matched_term",
    "field_name",
    "id",
    "description",
    "dataset",
    "dataset_id",
    "dataset_name",
    "category_id",
    "category_name",
    "type",
    "region",
    "universe",
    "delay",
    "rank_score",
]

DEFAULT_RETURN_COLUMNS = [
    "field_name",
    "dataset",
    "description",
    "coverage",
    "delay",
    "userCount",
    "alphaCount",
]

COLUMN_ALIASES = {
    "field": "field_name",
    "field id": "field_name",
    "field_id": "field_name",
    "field name": "field_name",
    "field_name": "field_name",
    "name": "field_name",
    "id": "field_name",
    "dataset": "dataset",
    "dataset name": "dataset",
    "dataset_name": "dataset",
    "dataset id": "dataset_id",
    "dataset_id": "dataset_id",
    "description": "description",
    "coverage": "coverage",
    "datecoverage": "dateCoverage",
    "date coverage": "dateCoverage",
    "delay": "delay",
    "usercount": "userCount",
    "user count": "userCount",
    "user_count": "userCount",
    "alphacount": "alphaCount",
    "alpha count": "alphaCount",
    "alpha_count": "alphaCount",
    "type": "type",
    "category": "category_name",
    "category_id": "category_id",
    "category id": "category_id",
    "matched_term": "matched_term",
    "matched term": "matched_term",
    "rank_score": "rank_score",
    "rank score": "rank_score",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "are",
    "as",
    "by",
    "data",
    "datafield",
    "datafields",
    "field",
    "fields",
    "find",
    "for",
    "give",
    "have",
    "hypothesis",
    "i",
    "include",
    "including",
    "in",
    "me",
    "need",
    "of",
    "on",
    "or",
    "related",
    "show",
    "stock",
    "stocks",
    "that",
    "the",
    "to",
    "top3000",
    "with",
}

INTENT_PROMPT = """You convert an alpha research hypothesis into WorldQuant Brain data-field search queries.

Return strict JSON only:
{
  "intent_summary": "one sentence summary of the user's field need",
  "search_queries": ["short, concrete data-field search strings"],
  "exclude_queries": ["concepts that should be filtered out"],
  "return_columns": ["field_name", "dataset", "description", "coverage", "delay", "userCount", "alphaCount"],
  "must_have_concepts": ["concepts useful for ranking returned fields"],
  "rationale": "brief reason for the query plan"
}

Rules:
- Do not invent WorldQuant field ids.
- Search queries should be concepts likely to appear in field ids/descriptions/dataset names.
- Include direct phrases from the user and finance/market synonyms when useful.
- Preserve explicit include/exclude/return instructions from the user.
- Prefer 8 to 30 search queries.
- Avoid generic words such as alpha, signal, factor, market, stock, data, field.
"""


class DataFieldSearcher:
    """Search WorldQuant Brain data-fields from a natural-language prompt."""

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        intent_mode: str = "auto",
        intent_model: str = "gemini-2.5-flash",
    ):
        self.credentials_path = Path(credentials_path)
        self.intent_mode = intent_mode
        self.intent_model = intent_model
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self._logged_in = False

    def login(self) -> None:
        if self._logged_in:
            return
        creds = json.loads(self.credentials_path.read_text())
        auth = base64.b64encode(
            f"{creds['email']}:{creds['password']}".encode("utf-8")
        ).decode("ascii")
        data = self._request_json(
            f"{BASE_URL}/authentication",
            method="POST",
            headers={"Authorization": f"Basic {auth}", "Accept": "application/json"},
        )
        if not isinstance(data, dict) or "user" not in data:
            raise RuntimeError(f"WorldQuant Brain authentication failed: {data}")
        self._logged_in = True

    def search_with_plan(
        self,
        prompt: str,
        *,
        instrument_type: str = "EQUITY",
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        limit_per_term: int = 20,
        max_terms: int = 24,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.login()
        plan = parse_intent(
            prompt,
            max_terms=max_terms,
            mode=self.intent_mode,
            model=self.intent_model,
        )
        terms = plan["search_queries"]
        logger.info("Intent: %s", plan.get("intent_summary", ""))
        logger.info("Search terms: %s", ", ".join(terms))
        if plan.get("exclude_queries"):
            logger.info("Exclude terms: %s", ", ".join(plan["exclude_queries"]))

        fields_by_id: dict[str, dict[str, Any]] = {}
        matched_terms: dict[str, list[str]] = {}
        for term in terms:
            results = self._search_term(
                term,
                instrument_type=instrument_type,
                region=region,
                universe=universe,
                delay=delay,
                limit=limit_per_term,
            )
            logger.info("%r returned %d fields", term, len(results))
            for item in results:
                field_id = item.get("id")
                if not field_id:
                    continue
                fields_by_id[field_id] = item
                matched_terms.setdefault(field_id, []).append(term)

        rows = [
            field_to_row(
                field,
                query=prompt,
                intent_summary=plan.get("intent_summary", ""),
                matched_term=", ".join(dict.fromkeys(matched_terms.get(field_id, []))),
                region=region,
                universe=universe,
                delay=delay,
                rank_score=score_field(field, plan),
            )
            for field_id, field in fields_by_id.items()
        ]
        rows = [row for row in rows if not should_exclude_row(row, plan)]
        rows.sort(key=lambda row: float(row["rank_score"] or 0), reverse=True)
        plan["matched_field_count"] = len(rows)
        return rows, plan

    def search(self, prompt: str, **kwargs: Any) -> list[dict[str, Any]]:
        rows, _ = self.search_with_plan(prompt, **kwargs)
        return rows

    def _search_term(
        self,
        term: str,
        *,
        instrument_type: str,
        region: str,
        universe: str,
        delay: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        params = {
            "instrumentType": instrument_type,
            "region": region,
            "delay": delay,
            "universe": universe,
            "limit": limit,
            "offset": 0,
            "search": term,
        }
        data = self._request_json(f"{BASE_URL}/data-fields?{urllib.parse.urlencode(params)}")
        if isinstance(data, dict):
            results = data.get("results", [])
        elif isinstance(data, list):
            results = data
        else:
            results = []
        return [item for item in results if isinstance(item, dict)]

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ) -> Any:
        request = urllib.request.Request(
            url,
            method=method,
            headers=headers or {"Accept": "application/json;version=2.0"},
        )
        with self.opener.open(request, timeout=45) as response:
            body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def parse_intent(
    prompt: str,
    *,
    max_terms: int = 24,
    mode: str = "auto",
    model: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    if mode not in {"auto", "llm", "heuristic"}:
        raise ValueError("intent mode must be one of: auto, llm, heuristic")

    structured = parse_structured_prompt(prompt, max_terms=max_terms)
    if structured:
        return structured

    if mode in {"auto", "llm"}:
        plan = parse_intent_with_llm(prompt, max_terms=max_terms, model=model)
        if plan:
            plan["intent_mode"] = "llm"
            return plan
        if mode == "llm":
            raise RuntimeError("LLM intent parsing failed and --intent-mode=llm was requested")

    return parse_intent_heuristic(prompt, max_terms=max_terms)


def parse_intent_with_llm(
    prompt: str,
    *,
    max_terms: int,
    model: str,
) -> dict[str, Any] | None:
    api_key = _load_env_value("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.info("No GEMINI_API_KEY available; using heuristic intent parsing")
        return None

    try:
        from google import genai
    except Exception as e:
        logger.info("google-genai unavailable (%s); using heuristic intent parsing", e)
        return None

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=(
                f"{INTENT_PROMPT}\n\n"
                f"User alpha/data-field request:\n{prompt}\n\n"
                f"Return at most {max_terms} search_queries."
            ),
        )
        text = getattr(response, "text", "") or ""
        data = _parse_json_object(text)
        return normalize_plan(data, prompt=prompt, max_terms=max_terms)
    except Exception as e:
        logger.warning("LLM intent parsing failed: %s", e)
        return None


def parse_intent_heuristic(prompt: str, max_terms: int = 24) -> dict[str, Any]:
    terms = extract_search_terms(prompt, max_terms=max_terms)
    return {
        "intent_mode": "heuristic",
        "intent_summary": _clean_text(prompt)[:240],
        "search_queries": terms,
        "exclude_queries": [],
        "return_columns": list(DEFAULT_RETURN_COLUMNS),
        "must_have_concepts": terms[:12],
        "rationale": "Fallback generic phrase extraction; no domain-specific mapping table was used.",
    }


def parse_structured_prompt(prompt: str, max_terms: int = 24) -> dict[str, Any] | None:
    include_text = _extract_section(
        prompt,
        start=r"find\s+datafields?\s+related\s+to",
        end=r"exclude\s+datafields?\s+related\s+to|return",
    )
    exclude_text = _extract_section(
        prompt,
        start=r"exclude\s+datafields?\s+related\s+to",
        end=r"return",
    )
    return_text = _extract_section(prompt, start=r"return", end=None)
    if not include_text and not exclude_text and not return_text:
        return None

    include_terms = [
        term for term in _parse_term_list(include_text)
        if not _too_generic(term)
    ][:max_terms]
    exclude_terms = [
        term for term in _parse_term_list(exclude_text)
        if not _too_generic(term)
    ]
    return_columns = normalize_return_columns(_parse_term_list(return_text))
    if not include_terms:
        include_terms = extract_search_terms(prompt, max_terms=max_terms)

    return {
        "intent_mode": "structured",
        "intent_summary": "Find datafields related to: " + ", ".join(include_terms),
        "search_queries": include_terms,
        "exclude_queries": exclude_terms,
        "return_columns": return_columns,
        "must_have_concepts": include_terms[:24],
        "rationale": "Parsed explicit Find/Exclude/Return sections from the prompt.",
    }


def normalize_plan(data: Any, *, prompt: str, max_terms: int) -> dict[str, Any]:
    if not isinstance(data, dict):
        return parse_intent_heuristic(prompt, max_terms=max_terms)

    raw_queries = data.get("search_queries") or data.get("queries") or []
    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]
    queries = []
    for item in raw_queries:
        if not isinstance(item, str):
            continue
        item = _clean_text(item).lower()
        if item and not _too_generic(item):
            queries.append(item)

    if not queries:
        queries = extract_search_terms(prompt, max_terms=max_terms)

    raw_concepts = data.get("must_have_concepts") or data.get("concepts") or []
    if isinstance(raw_concepts, str):
        raw_concepts = [raw_concepts]
    concepts = [
        _clean_text(item).lower()
        for item in raw_concepts
        if isinstance(item, str) and not _too_generic(item)
    ]

    raw_excludes = data.get("exclude_queries") or data.get("exclude") or []
    if isinstance(raw_excludes, str):
        raw_excludes = [raw_excludes]
    excludes = [
        _clean_text(item).lower()
        for item in raw_excludes
        if isinstance(item, str) and not _too_generic(item)
    ]

    raw_columns = data.get("return_columns") or data.get("columns") or []
    if isinstance(raw_columns, str):
        raw_columns = _parse_term_list(raw_columns)

    return {
        "intent_summary": _clean_text(str(data.get("intent_summary") or prompt))[:500],
        "search_queries": list(dict.fromkeys(queries))[:max_terms],
        "exclude_queries": list(dict.fromkeys(excludes)),
        "return_columns": normalize_return_columns(raw_columns),
        "must_have_concepts": list(dict.fromkeys(concepts or queries[:12]))[:24],
        "rationale": _clean_text(str(data.get("rationale") or "")),
    }


def extract_search_terms(prompt: str, max_terms: int = 24) -> list[str]:
    normalized = _clean_text(prompt).lower()
    focused = _focus_requested_fields(normalized)
    working_text = focused or normalized
    chunks = [
        _strip_prompt_filler(chunk.strip(" -_:"))
        for chunk in re.split(r"[/,;，；、\n]+|\band\b|\bor\b", working_text)
    ]
    chunks = [chunk for chunk in chunks if chunk]

    phrase_terms: list[str] = []
    token_terms: list[str] = []
    for chunk in chunks:
        useful_tokens = _useful_tokens(chunk)
        if 1 <= len(useful_tokens) <= 7:
            phrase_terms.append(" ".join(useful_tokens))
        tokens = re.findall(r"[a-z][a-z0-9_-]+", chunk)
        for token in tokens:
            if token not in STOPWORDS and len(token) > 2:
                token_terms.append(token)

    # Include compact phrases from adjacent useful tokens.
    tokens = _useful_tokens(working_text)
    adjacent_terms = []
    for left, right in zip(tokens, tokens[1:]):
        adjacent_terms.append(f"{left} {right}")

    whole_query = " ".join(_useful_tokens(working_text))
    terms = []
    if 2 <= len(_useful_tokens(whole_query)) <= 10:
        terms.append(whole_query)
    terms.extend(phrase_terms + token_terms + adjacent_terms)
    return list(dict.fromkeys(term for term in terms if term))[:max_terms]


def field_to_row(
    field: dict[str, Any],
    *,
    query: str,
    intent_summary: str,
    matched_term: str,
    region: str,
    universe: str,
    delay: int,
    rank_score: float,
) -> dict[str, Any]:
    dataset = field.get("dataset") or {}
    category = field.get("category") or {}
    field_id = field.get("id", "")
    dataset_id = dataset.get("id", "")
    dataset_name = dataset.get("name", "")
    return {
        "query": query,
        "intent_summary": intent_summary,
        "matched_term": matched_term,
        "field_name": field_id,
        "id": field_id,
        "description": field.get("description", ""),
        "dataset": dataset_name or dataset_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "category_id": category.get("id", ""),
        "category_name": category.get("name", ""),
        "type": field.get("type", ""),
        "region": region,
        "universe": universe,
        "coverage": field.get("coverage", ""),
        "dateCoverage": field.get("dateCoverage", ""),
        "delay": field.get("delay", delay),
        "userCount": field.get("userCount", ""),
        "alphaCount": field.get("alphaCount", ""),
        "rank_score": round(rank_score, 4),
    }


def should_exclude_row(row: dict[str, Any], plan: dict[str, Any]) -> bool:
    excludes = [str(term).lower() for term in plan.get("exclude_queries", []) if term]
    if not excludes:
        return False
    include_terms = [str(term).lower() for term in plan.get("search_queries", [])]
    haystack = row_haystack(row)
    for term in excludes:
        if term not in haystack and not all(token in haystack for token in _useful_tokens(term)):
            continue
        # If the user explicitly asked for a larger phrase containing this
        # exclude term, preserve matches to that include phrase. Example:
        # include "target price", exclude "price".
        protected = any(
            term in include and include != term and include in haystack
            for include in include_terms
        )
        if not protected:
            return True
    return False


def row_haystack(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, "")).lower()
        for key in (
            "field_name",
            "description",
            "dataset",
            "dataset_id",
            "dataset_name",
            "category_id",
            "category_name",
        )
    )


def score_field(field: dict[str, Any], plan: dict[str, Any]) -> float:
    dataset = field.get("dataset") or {}
    category = field.get("category") or {}
    haystack = " ".join(
        str(value).lower()
        for value in (
            field.get("id", ""),
            field.get("description", ""),
            dataset.get("id", ""),
            dataset.get("name", ""),
            category.get("id", ""),
            category.get("name", ""),
        )
    )
    field_id = str(field.get("id", "")).lower()
    score = 0.0
    for term in plan.get("search_queries", []):
        if term in field_id:
            score += 3.0
        elif term in haystack:
            score += 1.25
        else:
            token_hits = sum(1 for token in _useful_tokens(term) if token in haystack)
            score += 0.25 * token_hits
    for concept in plan.get("must_have_concepts", []):
        if concept in haystack:
            score += 0.75
    return score


def save_csv(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    fieldnames: list[str] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or FIELD_COLUMNS
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_clean_row(row, columns) for row in rows)
    return output


def save_json(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    output_path: str | Path,
    *,
    return_columns: list[str],
    request: dict[str, Any],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [_clean_row(row, return_columns) for row in rows]
    payload = {
        "request": request,
        "plan": plan,
        "summary": {
            "matched_field_count": len(fields),
            "returned_columns": return_columns,
        },
        "fields": fields,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def save_plan(plan: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def default_output_path(prompt: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:60] or "query"
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return Path(DEFAULT_OUTPUT_DIR) / f"{slug}_{digest}.json"


def _clean_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {key: _clean_cell(row.get(key, "")) for key in columns}


def _clean_cell(value: Any) -> Any:
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_json_object(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_section(prompt: str, *, start: str, end: str | None) -> str:
    flags = re.I | re.S | re.M
    end_pattern = f"(?=^\\s*(?:{end})\\s*:?)" if end else r"\Z"
    pattern = rf"^\s*{start}\s*:?\s*(.*?){end_pattern}"
    match = re.search(pattern, prompt, flags=flags)
    return match.group(1).strip() if match else ""


def _parse_term_list(text: str) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    bracket_match = re.search(r"\[(.*?)\]", text)
    if bracket_match:
        text = bracket_match.group(1)
    text = re.sub(r"^[\-\*\d\.\s]+", "", text)
    parts = re.split(r"[,，;；、\n]+", text)
    terms = []
    for part in parts:
        part = part.strip(" []'\"`")
        if not part:
            continue
        cleaned = _clean_text(part).lower()
        if cleaned:
            terms.append(cleaned)
    return list(dict.fromkeys(terms))


def normalize_return_columns(columns: Iterable[Any]) -> list[str]:
    normalized = []
    for column in columns:
        if not isinstance(column, str):
            continue
        key = _clean_text(column).lower()
        key = key.replace("-", " ")
        mapped = COLUMN_ALIASES.get(key) or COLUMN_ALIASES.get(key.replace(" ", "_"))
        if mapped:
            normalized.append(mapped)
    if not normalized:
        normalized = list(DEFAULT_RETURN_COLUMNS)
    return list(dict.fromkeys(normalized))


def _load_env_value(key: str, env_path: str = ".env") -> str:
    if not Path(env_path).exists():
        return ""
    try:
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _too_generic(term: str) -> bool:
    tokens = _useful_tokens(term)
    return not tokens


def _useful_tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z][a-z0-9_-]+", text)
        if token not in STOPWORDS and len(token) > 2 and not token.isdigit()
    ]


def _strip_prompt_filler(text: str) -> str:
    text = re.sub(
        r"^(please\s+)?(find|search|show|give|list|get)\s+"
        r"(me\s+)?(data\s*)?(fields?|datafields?)?\s*(about|for|related to)?\s*",
        "",
        text,
    )
    text = re.sub(r"\b(for\s+)?usa\s+top3000\s+delay\s+\d+\b", "", text)
    return text.strip(" -_:")


def _focus_requested_fields(text: str) -> str:
    patterns = [
        r"(?:find|search|show|list|get)\s+(?:me\s+)?(?:data\s*)?(?:fields?|datafields?)\s+(?:about|for|related to)\s+(.+)",
        r"(?:need|needs|want|wants)\s+(?:data\s*)?(?:fields?|datafields?)\s+(?:about|for|related to)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def print_summary(rows: list[dict[str, Any]], max_rows: int = 20) -> None:
    print(f"matched_fields={len(rows)}")
    for row in rows[:max_rows]:
        desc = row["description"][:100]
        print(
            f"- {row['field_name']} | {row['dataset']} | coverage={row.get('coverage', '')} | "
            f"userCount={row.get('userCount', '')} | alphaCount={row.get('alphaCount', '')} | {desc}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search WorldQuant Brain data-fields from a natural-language prompt. "
            "Recommended prompt pattern: -s \"Find datafields related to: "
            "[include terms]\\n\\nExclude datafields related to: [exclude terms]"
            "\\n\\nReturn: field_name, dataset, description, coverage, delay, "
            "userCount, alphaCount\""
        )
    )
    parser.add_argument(
        "-s",
        "--search",
        required=True,
        help="Natural-language or structured Find/Exclude/Return query.",
    )
    parser.add_argument("--credentials", default="credentials.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--instrument-type", default="EQUITY")
    parser.add_argument("--region", default="USA")
    parser.add_argument("--universe", default="TOP3000")
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--limit-per-term", type=int, default=20)
    parser.add_argument("--max-terms", type=int, default=48)
    parser.add_argument(
        "--intent-mode",
        choices=["auto", "llm", "heuristic"],
        default="auto",
        help="Intent parser mode. auto uses LLM when available, then heuristic fallback.",
    )
    parser.add_argument("--intent-model", default="gemini-2.5-flash")
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument("--show", type=int, default=20, help="Number of matches to print.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    searcher = DataFieldSearcher(
        credentials_path=args.credentials,
        intent_mode=args.intent_mode,
        intent_model=args.intent_model,
    )
    rows, plan = searcher.search_with_plan(
        args.search,
        instrument_type=args.instrument_type,
        region=args.region,
        universe=args.universe,
        delay=args.delay,
        limit_per_term=args.limit_per_term,
        max_terms=args.max_terms,
    )
    return_columns = plan.get("return_columns") or DEFAULT_RETURN_COLUMNS
    output = Path(args.output) if args.output else default_output_path(args.search)
    if args.format == "csv" and not args.output:
        output = output.with_suffix(".csv")

    request = {
        "search": args.search,
        "instrument_type": args.instrument_type,
        "region": args.region,
        "universe": args.universe,
        "delay": args.delay,
        "limit_per_term": args.limit_per_term,
        "max_terms": args.max_terms,
        "intent_mode": args.intent_mode,
        "intent_model": args.intent_model,
    }
    if args.format == "json":
        output = save_json(
            rows,
            plan,
            output,
            return_columns=return_columns,
            request=request,
        )
        plan_output = output
    else:
        output = save_csv(rows, output, fieldnames=return_columns)
        plan_output = output.with_suffix(".plan.json")
        save_plan(plan, plan_output)
    print_summary(rows, max_rows=args.show)
    print(f"intent_mode={plan.get('intent_mode', '')}")
    print(f"intent_summary={plan.get('intent_summary', '')}")
    if args.format == "csv":
        print(f"plan={plan_output}")
    print(output)


if __name__ == "__main__":
    main()
