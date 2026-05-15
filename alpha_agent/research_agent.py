"""
Research Agent: a parallel discovery agent that reads WorldQuant Brain
documentation, generates hypotheses, and proposes seed alpha candidates.

It does NOT consume WQ simulation budget. Output flows into two places:
  1. `data/research_report.html` — accumulating HTML research report
  2. `data/candidate_seeds.json` — structured candidate pool, queued for
     batch validation by SeedValidator before any candidate is allowed to
     enter the live SeedPool

This is the "discovery" half of the discover → accumulate → validate
pipeline. Keep this agent cheap (LLM-only); let SeedValidator pay the
WQ simulation cost in batched bursts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from html.parser import HTMLParser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus, urlencode, urljoin

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return False

from .seed_validator import SeedValidator
from .wq_client import WQBrainClient

load_dotenv()

logger = logging.getLogger(__name__)


# Topic suggestions for autonomous-mode rotation
DEFAULT_TOPICS = [
    "WorldQuant Brain 基本面数据(Fundamental & MDF)结合量价的 Alpha 挖掘策略",
    "WorldQuant Brain Analyst 数据 (est_eps, fam_*) 与价格反应的 Alpha 设计",
    "WorldQuant Brain Option 数据 (implied_volatility, IV skew) 的 Alpha 思路",
    "WorldQuant Brain News 情感数据与短期价格反转的 Alpha",
    "WorldQuant Brain FND6 财务字段 (debt, inventory, EPS surprise) 的 Alpha 挖掘",
    "WorldQuant Brain 跨域信号融合：group_neutralize / vector_neut 的高级用法",
    "WorldQuant Brain 高换手控制：hump, jump_decay, ts_target_tvr_decay 的实战",
]


FASTEXPR_OPERATORS = {
    "abs", "bucket", "densify", "group_mean", "group_neutralize",
    "group_rank", "group_scale", "group_sum", "group_zscore", "hump",
    "if_else", "jump_decay", "last_diff_value", "log", "max", "min",
    "normalize", "pasteurize", "power", "quantile", "rank", "scale",
    "sigmoid", "sign", "signed_power", "sqrt", "trade_when", "ts_arg_max",
    "ts_arg_min", "ts_av_diff", "ts_backfill", "ts_corr", "ts_covariance",
    "ts_decay_linear", "ts_delta", "ts_kurtosis", "ts_max", "ts_mean",
    "ts_min", "ts_product", "ts_quantile", "ts_rank", "ts_regression",
    "ts_scale", "ts_skewness", "ts_std_dev", "ts_step", "ts_sum",
    "ts_zscore", "vector_neut", "winsorize", "zscore",
}

FASTEXPR_GROUPS_AND_CONSTANTS = {
    "true", "false", "nan", "market", "sector", "industry", "subindustry",
    "dense", "range", "std", "rettype", "lag", "hump", "target_tvr",
    "sensitivity", "force",
}

COMMON_PRICE_FIELDS = {
    "open", "high", "low", "close", "vwap", "returns", "volume", "adv20",
    "cap",
}

TOPIC_SEARCH_TERMS = {
    "news": [
        "news", "sentiment", "headline", "article", "buzz", "media",
        "social", "event", "press", "nlp", "web", "story",
    ],
    "analyst": ["analyst", "estimate", "eps", "rating", "recommendation"],
    "option": ["option", "implied", "volatility", "put", "call"],
    "fundamental": ["fundamental", "income", "cashflow", "assets", "debt"],
    "mdf": ["mdf"],
    "fnd6": ["fnd6"],
}

WQ_SUPPORT_BASE_URL = "https://support.worldquantbrain.com"
WQ_PLATFORM_BASE_URL = "https://platform.worldquantbrain.com"


class _SupportPageParser(HTMLParser):
    """Tiny HTML parser for support search/link discovery."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: List[Dict[str, str]] = []
        self.assets: List[str] = []
        self.text_parts: List[str] = []
        self._current_href = ""
        self._current_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple]):
        attrs_dict = dict(attrs)
        if tag in ("script", "link"):
            src = attrs_dict.get("src") or attrs_dict.get("href") or ""
            if src:
                self.assets.append(urljoin(self.base_url, src))
        if tag != "a":
            return
        href = attrs_dict.get("href", "")
        if href:
            self._current_href = urljoin(self.base_url, href)
            self._current_text = []

    def handle_data(self, data: str):
        data = data.strip()
        if not data:
            return
        self.text_parts.append(data)
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str):
        if tag == "a" and self._current_href:
            text = " ".join(self._current_text).strip()
            if text:
                self.links.append({"url": self._current_href, "text": text})
            self._current_href = ""
            self._current_text = []

    @property
    def text(self) -> str:
        return " ".join(self.text_parts)


SYSTEM_INSTRUCTION = """
你是一个顶级的量化研究员（Quantitative Researcher），专精 WorldQuant Brain Alpha 挖掘平台。
你的任务是基于给定主题，产出两类成果：

1. 一份精美的中文 HTML 研究报告，详细解释金融逻辑、数据字段、算子用法。
2. 一组**结构化的 Seed Alpha 候选**，每个都包含完整的 Fast Expression 公式、所属数据 tag、金融假设。

**Seed Alpha 必须满足**：
- 使用合法的 WorldQuant Fast Expression 算子（rank, ts_rank, group_rank, ts_delta,
  group_neutralize, if_else, trade_when, ts_decay_linear, ts_corr 等）
- 数据字段必须真实存在（fundamental: operating_income, est_eps, current_ratio 等；
  MDF: mdf_pva, mdf_avi, mdf_opi 等；FND6: fnd6_dcvt, fnd6_optvol 等；
  Analyst: fam_est_eps_rank；Option: implied_volatility_call_120/put_120）
- 表达式简洁可执行，不要嵌套过深
- 低频字段（fundamental, mdf_*, fnd6_*）请用 126/252 天窗口的 ts_rank
- 避免显然会过拟合或权重集中的形式（裸 ratio 必须 rank 包裹）
- tag 必须是以下之一: fundamental, fundamental_mdf, fundamental_fnd6, analyst,
  option, news, price, volume, price_volume
- 避免错误：Unexpected character ':' near "undamental:current_r". <linkToCommonErrorMessages>Learn more</linkToCommonErrorMessages>
- 如果用户提示中提供了 “Live WorldQuant data fields”，Seed Alpha 只能使用其中列出的真实字段；不要发明字段。
- 对 news 字段：优先使用平台返回的 news/sentiment/buzz/headline 类字段，配合价格反应、成交量确认或情绪反转逻辑。

**输出格式（严格 JSON）**:
{
  "html_report": "<完整 HTML 字符串，body 内容即可，不需要 html/head 包裹>",
  "seed_candidates": [
    {
      "tag": "fundamental_mdf",
      "expression": "group_rank(ts_rank(mdf_oey, 252), subindustry)",
      "hypothesis": "operational earnings yield 在 subindustry 内排名能捕捉相对价值"
    },
    ...
  ]
}

请给出 5-10 个 seed candidates，每个都要有清晰的金融假设。不要用 ```json 包裹，直接返回 JSON。
"""


class WQFieldCatalog:
    """Small live data-field cache used to ground the research agent."""

    def __init__(
        self,
        credentials_path: str = "credentials.json",
        data_dir: str = "data",
        max_fields_per_query: int = 80,
        max_search_rounds: int = 3,
        min_fields: int = 1,
        client: Optional[WQBrainClient] = None,
    ):
        self.credentials_path = credentials_path
        self.cache_path = Path(data_dir) / "wq_field_catalog.json"
        self.max_fields_per_query = max_fields_per_query
        self.max_search_rounds = max_search_rounds
        self.min_fields = min_fields
        self.client = client or WQBrainClient(credentials_path=credentials_path, max_retries=2)
        self.fields_by_id: Dict[str, Dict[str, Any]] = {}
        self.datasets_by_id: Dict[str, Dict[str, Any]] = {}
        self.web_context: Dict[str, Any] = {}
        self._web_pages: Dict[str, Dict[str, Any]] = {}
        self.connected = False

    def load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            data = json.loads(self.cache_path.read_text())
            self.fields_by_id = {
                item["id"]: item for item in data.get("fields", [])
                if item.get("id")
            }
            logger.info(
                "📂 Loaded cached WQ field catalog: %d fields",
                len(self.fields_by_id),
            )
            return bool(self.fields_by_id)
        except Exception as e:
            logger.warning("⚠️ Failed to load WQ field cache: %s", e)
            return False

    def connect(self) -> bool:
        try:
            self.connected = self.client.login()
        except Exception as e:
            logger.warning("⚠️ WQ field grounding login failed: %s", e)
            self.connected = False
        return self.connected

    def refresh_for_topic(self, topic: str) -> Dict[str, Any]:
        if not self.connected and not self.connect():
            self.load_cache()
            return {
                "connected": False,
                "fields": list(self.fields_by_id.values()),
                "reason": "WQ login failed; using cache if available",
            }

        seed_terms = self._terms_for_topic(topic)
        self.web_context = self._crawl_platform_learn(topic, seed_terms)
        terms = list(dict.fromkeys(seed_terms + self.web_context.get("terms", [])))
        is_news_topic = self._is_news_topic(topic)
        search_log = []
        dataset_hints = self.web_context.get("dataset_hints", [])
        field_hints = self.web_context.get("field_hints", [])

        if is_news_topic:
            category_log, category_terms = self._discover_category("news")
            search_log.extend(category_log)
            terms = list(dict.fromkeys(terms + category_terms))

        for dataset_id in dataset_hints:
            search_log.append(self._fetch_fields_for_dataset(dataset_id))
        for field_id in field_hints:
            search_log.extend(self._fetch_fields_variants(search=field_id))

        if is_news_topic and any(self._looks_like_news_field(f) for f in self.fields_by_id.values()):
            logger.info("📰 Field discovery succeeded from platform documentation hints")
            self._save_cache()
            fields = list(self.fields_by_id.values())
            return {
                "connected": True,
                "fields": fields,
                "terms": terms,
                "web_context": self.web_context,
                "search_log": search_log,
                "reason": "",
            }

        if is_news_topic and not any(self._looks_like_news_field(f) for f in self.fields_by_id.values()):
            logger.warning(
                "⚠️ Platform data browser/category crawl found no usable news fields; "
                "aborting strict news grounding before blind keyword guessing."
            )
            self._save_cache()
            return {
                "connected": False,
                "fields": list(self.fields_by_id.values()),
                "terms": terms,
                "web_context": self.web_context,
                "search_log": search_log,
                "reason": (
                    "Platform data browser and category API discovery found no usable "
                    "news fields; blind keyword guessing disabled for news topics"
                ),
            }

        for round_idx in range(self.max_search_rounds):
            before = len(self.fields_by_id)
            round_terms = self._expand_terms(terms, round_idx)
            logger.info(
                "🕵️ WQ field discovery round %d/%d terms=%s",
                round_idx + 1,
                self.max_search_rounds,
                round_terms,
            )
            for term in round_terms:
                search_log.extend(self._fetch_fields_variants(search=term))
                search_log.extend(self._fetch_datasets_and_fields(search=term))

            if is_news_topic:
                news_count = sum(
                    1 for f in self.fields_by_id.values()
                    if self._looks_like_news_field(f)
                )
                if news_count >= self.min_fields:
                    break
            elif len(self.fields_by_id) >= self.min_fields:
                break

            if len(self.fields_by_id) == before and round_idx == self.max_search_rounds - 2:
                # Last broad pass: some WQ endpoints ignore `search`; pull a
                # generic page and filter locally instead of giving up early.
                search_log.extend(self._fetch_fields_variants(search=""))

        self._save_cache()
        fields = list(self.fields_by_id.values())
        if is_news_topic:
            news_fields = [f for f in fields if self._looks_like_news_field(f)]
            fields_for_status = news_fields
        else:
            fields_for_status = fields
        return {
            "connected": bool(fields_for_status),
            "fields": fields,
            "terms": terms,
            "web_context": self.web_context,
            "search_log": search_log,
            "reason": "" if fields_for_status else "WQ discovery exhausted all query strategies without usable topic fields",
        }

    def format_for_prompt(self, topic: str, max_fields: int = 120) -> str:
        fields = list(self.fields_by_id.values())
        if not fields:
            return (
                "## Live WorldQuant data fields\n"
                "No live field catalog is available. Be conservative and use only well-known fields from the system instruction."
            )

        topic_lower = topic.lower()
        if self._is_news_topic(topic):
            preferred = [
                f for f in fields
                if self._looks_like_news_field(f)
            ]
        else:
            preferred = fields

        if not preferred:
            preferred = fields

        preferred = preferred[:max_fields]
        lines = [
            "## Live WorldQuant data fields",
            "Use ONLY these live platform-returned data fields plus standard price/volume fields. Do not invent fields.",
        ]
        web_lines = self._format_web_context()
        if web_lines:
            lines.extend(["", *web_lines, ""])
        for field in preferred:
            fid = field.get("id", "")
            desc = field.get("description") or field.get("name") or ""
            dataset = (
                field.get("dataset", {}).get("id")
                if isinstance(field.get("dataset"), dict)
                else field.get("dataset")
            )
            suffix = f" [{dataset}]" if dataset else ""
            desc = str(desc).replace("\n", " ")[:100]
            lines.append(f"- {fid}{suffix}: {desc}")
        return "\n".join(lines)

    def _format_web_context(self) -> List[str]:
        if not self.web_context:
            return []
        lines = ["## WorldQuant platform/context pages used for topic selection"]
        for link in self.web_context.get("links", [])[:5]:
            lines.append(f"- {link['title']}: {link['url']}")
            snippet = link.get("snippet", "")
            if snippet:
                lines.append(f"  snippet: {snippet[:220]}")
        terms = self.web_context.get("terms", [])
        if terms:
            lines.append(f"- Extracted search terms: {', '.join(terms[:30])}")
        return lines

    def known_field_ids(self) -> Set[str]:
        ids = set(self.fields_by_id) | COMMON_PRICE_FIELDS
        return ids | {field_id.lower() for field_id in ids}

    @staticmethod
    def _looks_like_news_field(field: Dict[str, Any]) -> bool:
        text = json.dumps(field, ensure_ascii=False).lower()
        return any(term in text for term in TOPIC_SEARCH_TERMS["news"])

    @staticmethod
    def _is_news_topic(topic: str) -> bool:
        topic_lower = topic.lower()
        return "news" in topic_lower or "新闻" in topic or "情感" in topic

    @staticmethod
    def _terms_for_topic(topic: str) -> List[str]:
        topic_lower = topic.lower()
        terms: List[str] = []
        for key, values in TOPIC_SEARCH_TERMS.items():
            if key in topic_lower or key.upper() in topic or key in topic:
                terms.extend(values)
        if "新闻" in topic or "情感" in topic:
            terms.extend(TOPIC_SEARCH_TERMS["news"])
        if not terms:
            terms = ["news", "analyst", "fundamental", "option", "mdf", "fnd6"]
        return list(dict.fromkeys(terms))

    @staticmethod
    def _expand_terms(terms: List[str], round_idx: int) -> List[str]:
        if round_idx == 0:
            return terms[:6]
        if round_idx == 1:
            expanded = list(terms)
            for term in terms:
                expanded.extend([f"{term}_score", f"{term}_rank", f"{term}_volume"])
            return list(dict.fromkeys(expanded))[:18]
        expanded = list(terms)
        expanded.extend(["news", "sentiment", "media", "social", "event", "web", "nlp", ""])
        return list(dict.fromkeys(expanded))

    def _crawl_platform_learn(self, topic: str, seed_terms: List[str]) -> Dict[str, Any]:
        frontier = list(self._platform_doc_seed_urls(topic))
        seen = set()
        relevant_pages: List[Dict[str, Any]] = []
        extracted_terms: List[str] = []
        dataset_hints: List[str] = []
        field_hints: List[str] = []
        crawl_log: List[str] = []
        max_pages = 30

        while frontier and len(seen) < max_pages:
            url = frontier.pop(0)
            if url in seen:
                continue
            seen.add(url)
            html = self._fetch_html(url)
            crawl_log.append(f"{url}:bytes={len(html)}")
            if not html:
                continue

            parser = _SupportPageParser(url)
            parser.feed(html)
            page_text = parser.text
            asset_texts = []
            for asset_url in parser.assets[:10]:
                if asset_url in seen:
                    continue
                asset_text = self._fetch_html(asset_url)
                crawl_log.append(f"{asset_url}:bytes={len(asset_text)}")
                if asset_text:
                    asset_texts.append(asset_text)

            combined_text = " ".join([page_text, *asset_texts])
            score = self._text_score(topic, seed_terms, combined_text + " " + url)
            new_paths = []
            for text_blob in [html, *asset_texts]:
                new_paths.extend(self._extract_platform_learn_paths(text_blob))
            for path in new_paths:
                child_url = urljoin(WQ_PLATFORM_BASE_URL, path)
                if child_url not in seen and child_url not in frontier:
                    frontier.append(child_url)

            page_terms = self._extract_terms_from_text(combined_text, seed_terms)
            page_datasets = self._extract_dataset_hints(combined_text)
            page_fields = self._extract_field_hints(combined_text)
            extracted_terms.extend(page_terms)
            dataset_hints.extend(page_datasets)
            field_hints.extend(page_fields)

            if score > 0 or page_datasets or page_fields:
                relevant_pages.append({
                    "url": url,
                    "title": self._title_from_url(url),
                    "score": str(score),
                    "snippet": self._best_snippet(combined_text, seed_terms),
                })

            if self._is_news_topic(topic) and (page_datasets or page_fields):
                if any("news" in item.lower() or "sent" in item.lower() for item in page_datasets + page_fields):
                    break

        relevant_pages.sort(key=lambda item: int(item.get("score", "0")), reverse=True)
        extracted_terms = list(dict.fromkeys(extracted_terms))
        dataset_hints = list(dict.fromkeys(dataset_hints))
        field_hints = list(dict.fromkeys(field_hints))
        logger.info(
            "🌐 WQ platform crawl visited %d pages; selected=%d terms=%s dataset_hints=%s field_hints=%s",
            len(seen),
            len(relevant_pages[:5]),
            extracted_terms[:20],
            dataset_hints[:10],
            field_hints[:10],
        )
        for page in relevant_pages[:5]:
            logger.info(
                "   📄 platform page score=%s url=%s snippet=%s",
                page.get("score", "0"),
                page.get("url", ""),
                page.get("snippet", "")[:180],
            )
        return {
            "links": relevant_pages[:5],
            "terms": extracted_terms,
            "dataset_hints": dataset_hints,
            "field_hints": field_hints,
            "crawl_log": crawl_log,
        }

    @staticmethod
    def _platform_doc_seed_urls(topic: str) -> List[str]:
        urls = [
            f"{WQ_PLATFORM_BASE_URL}/learn/documentation",
            f"{WQ_PLATFORM_BASE_URL}/learn/data-and-operators",
        ]
        if WQFieldCatalog._is_news_topic(topic):
            urls.extend([
                f"{WQ_PLATFORM_BASE_URL}/data/data-sets?category=news",
                f"{WQ_PLATFORM_BASE_URL}/data/data-fields?category=news",
                f"{WQ_PLATFORM_BASE_URL}/data/data-sets",
                f"{WQ_PLATFORM_BASE_URL}/data/data-fields",
                f"{WQ_PLATFORM_BASE_URL}/learn/documentation/data/news",
                f"{WQ_PLATFORM_BASE_URL}/learn/documentation/data",
                f"{WQ_PLATFORM_BASE_URL}/learn/documentation/datasets",
            ])
        urls.append(f"{WQ_PLATFORM_BASE_URL}/learn/documentation/examples/19-alpha-examples")
        return list(dict.fromkeys(urls))

    @staticmethod
    def _extract_platform_learn_paths(text: str) -> List[str]:
        paths = re.findall(r'["\'](/(?:learn|data)/[^"\']+)["\']', text or "")
        paths.extend(re.findall(r'["\']((?:learn|data)/[^"\']+)["\']', text or ""))
        clean = []
        for path in paths:
            if path.startswith(("learn/", "data/")):
                path = "/" + path
            path = path.split("\\", 1)[0]
            if len(path) < 120:
                clean.append(path)
        return list(dict.fromkeys(clean))

    @staticmethod
    def _support_search_urls(query: str) -> List[str]:
        q = quote_plus(query)
        return [
            f"{WQ_SUPPORT_BASE_URL}/hc/en-us/search?query={q}",
            f"{WQ_SUPPORT_BASE_URL}/hc/en-us/community/search?query={q}",
        ]

    def _fetch_html(self, url: str) -> str:
        try:
            response = self.client._safe_get(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
                timeout=30,
            )
            if not response.content:
                return ""
            text = response.text
            content_type = response.headers.get("Content-Type", "").lower()
            is_text_asset = any(
                marker in content_type
                for marker in ("javascript", "json", "text/css", "text/plain")
            ) or "/static/" in url
            if not is_text_asset and "<html" not in text.lower() and "<a " not in text.lower():
                return ""
            return text
        except Exception as e:
            logger.debug("WQ web fetch failed for %s: %s", url, e)
            return ""

    @staticmethod
    def _link_score(topic: str, seed_terms: List[str], link: Dict[str, str]) -> int:
        haystack = f"{link.get('text', '')} {link.get('url', '')}".lower()
        score = 0
        for term in seed_terms:
            if term and term.lower() in haystack:
                score += 3
        for token in re.findall(r"[a-zA-Z]{3,}", topic.lower()):
            if token in haystack:
                score += 1
        return score

    @staticmethod
    def _url_score(topic: str, seed_terms: List[str], url: str) -> int:
        haystack = url.lower()
        score = 0
        for term in seed_terms:
            if term and term.lower() in haystack:
                score += 4
        for token in re.findall(r"[a-zA-Z]{3,}", topic.lower()):
            if token in haystack:
                score += 1
        return score

    @staticmethod
    def _text_score(topic: str, seed_terms: List[str], text: str) -> int:
        haystack = (text or "").lower()
        score = 0
        for term in seed_terms:
            if term and term.lower() in haystack:
                score += 4
        for token in re.findall(r"[a-zA-Z]{3,}", topic.lower()):
            if token in haystack:
                score += 1
        return score

    @staticmethod
    def _title_from_url(url: str) -> str:
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        return f"Platform Learn: {tail.replace('-', ' ') or 'documentation'}"

    @staticmethod
    def _best_snippet(text: str, terms: List[str], window: int = 260) -> str:
        clean = re.sub(r"\s+", " ", text or "").strip()
        if not clean:
            return ""
        lower = clean.lower()
        positions = [
            lower.find(term.lower())
            for term in terms
            if term and lower.find(term.lower()) >= 0
        ]
        start = max(0, min(positions) - 80) if positions else 0
        return clean[start:start + window]

    @staticmethod
    def _extract_terms_from_text(text: str, seed_terms: List[str]) -> List[str]:
        lower = (text or "").lower()
        terms = []
        for term in seed_terms:
            if term.lower() in lower:
                terms.append(term)
        for token in re.findall(r"\b[a-z][a-z0-9_]{2,}\b", lower):
            if any(marker in token for marker in ("news", "sent", "buzz", "headline", "article", "media", "social", "event")):
                terms.append(token)
        return list(dict.fromkeys(terms))[:40]

    @staticmethod
    def _extract_dataset_hints(text: str) -> List[str]:
        hints = []
        patterns = [
            r'"dataset\.id"\s*:\s*"([^"]+)"',
            r'"datasetId"\s*:\s*"([^"]+)"',
            r'"dataset"\s*:\s*\{[^{}]*"id"\s*:\s*"([^"]+)"',
            r'\bdataset[=:]\s*["\']?([a-zA-Z0-9_./-]+)',
        ]
        for pattern in patterns:
            hints.extend(re.findall(pattern, text or "", flags=re.IGNORECASE))
        return list(dict.fromkeys(
            h.strip().strip("'\"")
            for h in hints
            if 2 <= len(h.strip().strip("'\"")) <= 80
        ))

    @staticmethod
    def _extract_field_hints(text: str) -> List[str]:
        hints = []
        for pattern in (
            r'"id"\s*:\s*"([a-zA-Z][a-zA-Z0-9_]{2,})"',
            r'"field"\s*:\s*"([a-zA-Z][a-zA-Z0-9_]{2,})"',
            r'"dataField"\s*:\s*"([a-zA-Z][a-zA-Z0-9_]{2,})"',
        ):
            hints.extend(re.findall(pattern, text or ""))
        hints.extend(
            token for token in re.findall(r"\b[a-z][a-z0-9_]{3,}\b", (text or "").lower())
            if any(marker in token for marker in ("news", "sent", "buzz", "headline", "article", "media", "social", "event"))
        )
        blocked = FASTEXPR_OPERATORS | FASTEXPR_GROUPS_AND_CONSTANTS
        return list(dict.fromkeys(
            h for h in hints
            if h.lower() not in blocked and len(h) <= 80
        ))[:80]

    def _discover_category(self, category: str) -> tuple[List[str], List[str]]:
        """Ground a topic from the platform data browser before keyword search.

        The `/data/...` pages are SPA routes, so useful identifiers may come
        from either bundled JS or authenticated API calls with category params.
        """
        logs: List[str] = []
        terms: List[str] = []
        urls = [
            f"{WQ_PLATFORM_BASE_URL}/data/data-sets?category={category}",
            f"{WQ_PLATFORM_BASE_URL}/data/data-fields?category={category}",
            f"{WQ_PLATFORM_BASE_URL}/data/data-sets",
            f"{WQ_PLATFORM_BASE_URL}/data/data-fields",
        ]
        dataset_hints: List[str] = []
        field_hints: List[str] = []

        for url in urls:
            html = self._fetch_html(url)
            logs.append(f"platform-data:{url}:bytes={len(html)}")
            if not html:
                continue
            parser = _SupportPageParser(url)
            parser.feed(html)
            asset_texts = []
            for asset_url in parser.assets[:12]:
                asset_text = self._fetch_html(asset_url)
                logs.append(f"platform-data-asset:{asset_url}:bytes={len(asset_text)}")
                if asset_text:
                    asset_texts.append(asset_text)
            combined_text = " ".join([parser.text, html, *asset_texts])
            terms.extend(self._extract_terms_from_text(combined_text, TOPIC_SEARCH_TERMS.get(category, [category])))
            dataset_hints.extend(self._extract_dataset_hints(combined_text))
            field_hints.extend(self._extract_field_hints(combined_text))

        dataset_hints = list(dict.fromkeys(dataset_hints))
        field_hints = list(dict.fromkeys(field_hints))
        for dataset_id in dataset_hints:
            if category.lower() in dataset_id.lower() or "sent" in dataset_id.lower():
                logs.append(self._fetch_fields_for_dataset(dataset_id))
        for field_id in field_hints:
            if category.lower() in field_id.lower() or "sent" in field_id.lower():
                logs.extend(self._fetch_fields_variants(search=field_id))

        dataset_param_variants = [
            {"category": category},
            {"category": category.upper()},
            {"category.id": category},
            {"dataCategory": category},
            {"type": category},
            {"search": category, "category": category},
        ]
        for params in dataset_param_variants:
            logs.extend(self._fetch_datasets_by_params(
                params,
                label=self._param_label("category-data-sets", params),
                filter_text="",
            ))

        field_param_variants = [
            {"category": category},
            {"category": category.upper()},
            {"category.id": category},
            {"dataset.category": category},
            {"dataCategory": category},
            {"type": category},
            {"search": category, "category": category},
        ]
        for params in field_param_variants:
            logs.append(self._fetch_fields_by_extra_params(params, label=self._param_label("category-data-fields", params)))

        logger.info(
            "📰 WQ category discovery category=%s dataset_hints=%s field_hints=%s live_news_fields=%d",
            category,
            dataset_hints[:10],
            field_hints[:10],
            sum(1 for f in self.fields_by_id.values() if self._looks_like_news_field(f)),
        )
        return logs, list(dict.fromkeys(terms))

    def _fetch_fields_by_extra_params(self, extra_params: Dict[str, Any], label: str) -> str:
        params = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "delay": 1,
            "universe": "TOP3000",
            "limit": self.max_fields_per_query,
            "offset": 0,
        }
        params.update(extra_params)
        return self._fetch_fields(params=params, label=label)

    def _fetch_fields_variants(self, search: str, offset: int = 0) -> List[str]:
        base_params = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "delay": 1,
            "universe": "TOP3000",
            "limit": self.max_fields_per_query,
            "offset": offset,
        }
        variants = []
        for key in ("search", "query", "name"):
            params = dict(base_params)
            if search:
                params[key] = search
            variants.append(params)
        if not search:
            variants.append(base_params)

        logs = []
        for params in variants:
            logs.append(self._fetch_fields(params=params, label=params.get("search") or params.get("query") or params.get("name") or "<all>"))
        return logs

    def _fetch_fields(self, params: Dict[str, Any], label: str) -> str:
        url = f"{self.client.BASE_URL}/data-fields?{urlencode(params)}"
        try:
            response = self.client._safe_get(
                url,
                headers={"Accept": "application/json;version=2.0"},
                timeout=30,
            )
            if not response.content:
                return
            payload = response.json()
            results = self._extract_results(payload)
            for item in results:
                if not isinstance(item, dict):
                    continue
                field_id = item.get("id")
                if field_id:
                    self.fields_by_id[field_id] = item
            logger.info("🔎 WQ data-fields %s returned %d fields", label, len(results))
            return f"data-fields:{label}:{len(results)}"
        except Exception as e:
            logger.warning("⚠️ WQ data-fields query failed for %r: %s", label, e)
            return f"data-fields:{label}:error:{e}"

    def _fetch_fields_for_dataset(self, dataset_id: str) -> str:
        params = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "delay": 1,
            "universe": "TOP3000",
            "limit": self.max_fields_per_query,
            "dataset.id": dataset_id,
        }
        return self._fetch_fields(params=params, label=f"dataset:{dataset_id}")

    def _fetch_datasets_and_fields(self, search: str) -> List[str]:
        params = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "delay": 1,
            "universe": "TOP3000",
            "limit": self.max_fields_per_query,
        }
        if search:
            params["search"] = search
        return self._fetch_datasets_by_params(params, label=f"search={search or '<all>'}", filter_text=search)

    def _fetch_datasets_by_params(
        self,
        params: Dict[str, Any],
        label: str,
        filter_text: Optional[str] = None,
    ) -> List[str]:
        request_params = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "delay": 1,
            "universe": "TOP3000",
            "limit": self.max_fields_per_query,
        }
        request_params.update(params)
        if filter_text is None:
            filter_text = str(params.get("search") or params.get("category") or "")
        url = f"{self.client.BASE_URL}/data-sets?{urlencode(request_params)}"
        logs = []
        try:
            response = self.client._safe_get(
                url,
                headers={"Accept": "application/json;version=2.0"},
                timeout=30,
            )
            payload = response.json() if response.content else {}
            datasets = self._extract_results(payload)
            for dataset in datasets:
                dataset_id = dataset.get("id")
                if not dataset_id:
                    continue
                self.datasets_by_id[dataset_id] = dataset
                if filter_text and filter_text.lower() not in json.dumps(dataset, ensure_ascii=False).lower():
                    continue
                field_params = {
                    "instrumentType": "EQUITY",
                    "region": "USA",
                    "delay": 1,
                    "universe": "TOP3000",
                    "limit": self.max_fields_per_query,
                    "dataset.id": dataset_id,
                }
                logs.append(self._fetch_fields(field_params, label=f"dataset:{dataset_id}"))
            logger.info("📚 WQ data-sets %s returned %d datasets", label, len(datasets))
            logs.append(f"data-sets:{label}:{len(datasets)}")
        except Exception as e:
            logger.warning("⚠️ WQ data-sets query failed for %r: %s", label, e)
            logs.append(f"data-sets:{label}:error:{e}")
        return logs

    @staticmethod
    def _param_label(prefix: str, params: Dict[str, Any]) -> str:
        parts = [f"{key}={value}" for key, value in sorted(params.items())]
        return f"{prefix}({','.join(parts)})"

    @staticmethod
    def _extract_results(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("results", "data", "fields"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            if payload.get("id"):
                return [payload]
        return []

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now().isoformat(),
                    "fields": list(self.fields_by_id.values()),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


class ResearchAgent:
    """
    LLM-driven research agent that produces hypotheses + structured seed
    candidates. Candidates flow into the SeedValidator's pending pool;
    they do NOT bypass the validation gate.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        data_dir: str = "data",
        validator: Optional[SeedValidator] = None,
        credentials_path: str = "credentials.json",
        use_wq_grounding: bool = True,
        allow_ungrounded_candidates: bool = False,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in .env.")

        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.html_path = self.data_dir / "research_report.html"
        self.md_path = self.data_dir / "seeds_and_hypotheses.md"
        self.allow_ungrounded_candidates = allow_ungrounded_candidates
        self.field_catalog: Optional[WQFieldCatalog] = None
        if use_wq_grounding:
            self.field_catalog = WQFieldCatalog(
                credentials_path=credentials_path,
                data_dir=str(self.data_dir),
            )

        # The validator handles the candidate pool. We can be given one
        # (recommended — shared with the main agent) or we lazily build one
        # without a WQ client (just for queueing, no validation here).
        self.validator = validator or SeedValidator(
            wq_client=None,
            save_path=str(self.data_dir / "candidate_seeds.json"),
        )
        if self.validator.save_path.exists() and not self.validator.candidates:
            self.validator.load_or_init()

        self._init_client()

    def _init_client(self):
        from google import genai
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"🤖 Research Agent ready — model: {self.model_name}")

    # ── Public entry points ─────────────────────────────────────────

    def conduct_research(self, topic: str) -> int:
        """
        Run one research cycle for `topic`. Returns the number of NEW
        candidates queued (post-dedup) into the pool.
        """
        self._stage_banner(f"📚 RESEARCH STAGE — {topic[:55]}")
        logger.info(f"🔍 Topic: {topic}")

        field_context = self._prepare_field_context(topic)
        if field_context is None:
            logger.error("❌ Research aborted: no grounded fields found for topic")
            return 0

        raw_text = self._call_llm(topic, field_context=field_context)
        if not raw_text:
            logger.error("❌ LLM returned empty response")
            return 0

        data = self._parse_response(raw_text)
        if data is None:
            logger.error("❌ Failed to parse LLM response as JSON")
            return 0

        # ── HTML report side ──
        html_report = data.get("html_report", "").strip()
        if html_report:
            self._append_html_report(topic, html_report)
        else:
            logger.warning("⚠️ No html_report field in LLM response")

        # ── Seed candidates side ──
        candidates = data.get("seed_candidates", [])
        if not isinstance(candidates, list):
            logger.warning("⚠️ seed_candidates is not a list, skipping intake")
            return 0

        logger.info(f"📦 LLM proposed {len(candidates)} seed candidate(s)")
        candidates = self._filter_grounded_candidates(candidates)
        self._dump_markdown_summary(topic, candidates)
        added = self.validator.add_candidates_bulk(
            candidates, source=f"research:{self._topic_slug(topic)}"
        )

        # ── Stage summary ──
        self._stage_summary(topic, len(candidates), added)
        return added

    def conduct_research_batch(self, topics: List[str]) -> int:
        """Run research over multiple topics, returns total new candidates."""
        total_added = 0
        for i, topic in enumerate(topics, start=1):
            logger.info(f"\n{'#' * 60}")
            logger.info(f"# TOPIC {i}/{len(topics)}: {topic[:50]}")
            logger.info(f"{'#' * 60}")
            total_added += self.conduct_research(topic)
        return total_added

    # ── Internal helpers ────────────────────────────────────────────

    def _prepare_field_context(self, topic: str) -> Optional[str]:
        if not self.field_catalog:
            logger.warning(
                "⚠️ WQ field grounding disabled. Research output may hallucinate fields."
            )
            return ""

        status = self.field_catalog.refresh_for_topic(topic)
        field_count = len(status.get("fields", []))
        if status.get("connected") and field_count:
            logger.info(
                "✅ WQ field grounding active: %d live/cached fields loaded",
                field_count,
            )
        else:
            logger.warning("⚠️ WQ field grounding degraded: %s", status.get("reason"))
        context = self.field_catalog.format_for_prompt(topic)
        if "news" in topic.lower() or "新闻" in topic or "情感" in topic:
            news_count = sum(
                1 for f in self.field_catalog.fields_by_id.values()
                if self.field_catalog._looks_like_news_field(f)
            )
            logger.info("📰 News-like live fields available: %d", news_count)
            if news_count == 0 and not self.allow_ungrounded_candidates:
                logger.error(
                    "❌ No live/cached news-like fields found after WQ discovery. "
                    "Use --allow-ungrounded-candidates only if you explicitly want LLM-only guesses."
                )
                return None
        return context

    def _call_llm(self, topic: str, field_context: str = "") -> Optional[str]:
        from google.genai import types

        user_prompt = (
            f"请针对以下主题产出研报与 seed candidates。\n\n"
            f"主题：【{topic}】\n\n"
            f"{field_context}\n\n"
            f"请遵守系统指令中的所有约束，输出严格 JSON。"
        )

        logger.info(f"🧠 Calling {self.model_name} (this may take ~1 min)...")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                    response_mime_type="application/json",
                ),
            )
            text = response.text
            logger.info(f"📝 LLM response received ({len(text)} chars)")
            return text
        except Exception as e:
            logger.error(f"❌ Gemini API error: {e}")
            return None

    def _filter_grounded_candidates(self, candidates: List[dict]) -> List[dict]:
        if (
            not self.field_catalog
            or not self.field_catalog.fields_by_id
            or self.allow_ungrounded_candidates
        ):
            return candidates

        known_fields = self.field_catalog.known_field_ids()
        filtered = []
        rejected = 0
        for candidate in candidates:
            expr = candidate.get("expression", "")
            unknown = self._unknown_fields(expr, known_fields)
            if unknown:
                rejected += 1
                logger.warning(
                    "🚫 Rejecting ungrounded research candidate: unknown fields=%s expr=%s",
                    sorted(unknown),
                    expr[:100],
                )
                continue
            filtered.append(candidate)

        if rejected:
            logger.info(
                "🧹 Grounding filter kept %d/%d candidates",
                len(filtered),
                len(candidates),
            )
        return filtered

    @staticmethod
    def _unknown_fields(expression: str, known_fields: Set[str]) -> Set[str]:
        identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression or ""))
        unknown = set()
        for token in identifiers:
            token_lower = token.lower()
            if token_lower in FASTEXPR_OPERATORS:
                continue
            if token_lower in FASTEXPR_GROUPS_AND_CONSTANTS:
                continue
            if token in known_fields or token_lower in known_fields:
                continue
            unknown.add(token)
        return unknown

    @staticmethod
    def _parse_response(text: str) -> Optional[dict]:
        """Robust JSON parsing — strips ``` fences, finds outer object."""
        text = text.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract outermost {...}
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def _append_html_report(self, topic: str, html_content: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        section = (
            f'<section style="border-top:2px solid #888; padding-top:1em; '
            f'margin-top:2em;">\n'
            f'<h2>🧪 {topic}</h2>\n'
            f'<p style="color:#888;font-size:0.9em">研究时间: {timestamp}</p>\n'
            f'{html_content}\n'
            f'</section>\n'
        )
        if not self.html_path.exists():
            wrapper = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Alpha Research Report</title>"
                "<style>body{font-family:-apple-system,sans-serif;max-width:900px;"
                "margin:2em auto;padding:0 1em;line-height:1.6}"
                "code{background:#f4f4f4;padding:2px 4px;border-radius:3px}"
                "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px}"
                "</style></head><body>\n"
                f"<h1>📊 Alpha Research Report</h1>\n"
                f"{section}"
                "</body></html>"
            )
            self.html_path.write_text(wrapper, encoding="utf-8")
            logger.info(f"✅ Created HTML report: {self.html_path}")
        else:
            existing = self.html_path.read_text(encoding="utf-8")
            if "</body>" in existing:
                updated = existing.replace("</body>", f"{section}</body>")
            else:
                updated = existing + section
            self.html_path.write_text(updated, encoding="utf-8")
            logger.info(f"✅ Appended to HTML report: {self.html_path}")

    def _dump_markdown_summary(self, topic: str, candidates: List[dict]):
        """Human-readable summary of what the LLM proposed this cycle."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n## {topic}  \n*({timestamp})*\n\n")
            for i, c in enumerate(candidates, start=1):
                tag = c.get("tag", "?")
                expr = c.get("expression", "").strip()
                hyp = c.get("hypothesis", "").strip()
                f.write(f"### Candidate {i} `[{tag}]`\n")
                f.write(f"- **Expression**: `{expr}`\n")
                f.write(f"- **Hypothesis**: {hyp}\n\n")

    @staticmethod
    def _topic_slug(topic: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic)[:40]
        return slug.strip("_").lower() or "topic"

    @staticmethod
    def _stage_banner(text: str):
        logger.info("")
        logger.info("=" * 64)
        logger.info(text)
        logger.info("=" * 64)

    def _stage_summary(self, topic: str, proposed: int, added: int):
        deduped = proposed - added
        pending = self.validator.pending_count()
        total = len(self.validator.candidates)
        logger.info("")
        logger.info("─" * 64)
        logger.info(f"📊 Stage summary — {topic[:50]}")
        logger.info(f"   proposed:  {proposed}")
        logger.info(f"   queued:    {added} (new)")
        logger.info(f"   deduped:   {deduped} (already in pool)")
        logger.info(f"   pool size: {total} (pending={pending})")
        logger.info("─" * 64)


def _setup_logging(verbose: bool, log_file: Optional[Path] = None):
    from .color_log import apply_color_logging
    apply_color_logging(
        level=logging.DEBUG if verbose else logging.INFO,
        log_file=str(log_file) if log_file else None,
        noisy_loggers=["httpx", "httpcore", "urllib3", "google.auth",
                       "google.genai", "google.api_core"],
    )


def main():
    parser = argparse.ArgumentParser(description="Research Agent")
    parser.add_argument(
        "--topic", type=str,
        help="Specific topic to research (default: rotate through built-in list)",
    )
    parser.add_argument(
        "--topics-batch", type=int, default=0,
        help="Run N topics from the default rotation (0 = single topic only)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-pro",
        help="Gemini model name",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory for output files",
    )
    parser.add_argument(
        "--credentials", type=str, default="credentials.json",
        help="Path to WQ Brain credentials JSON for live field grounding",
    )
    parser.add_argument(
        "--no-wq-grounding", action="store_true",
        help="Disable live WQ data-field lookup; LLM may hallucinate fields",
    )
    parser.add_argument(
        "--allow-ungrounded-candidates", action="store_true",
        help="Queue candidates even if they use fields absent from live field catalog",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose, log_file=Path(args.data_dir) / "research_agent.log")

    agent = ResearchAgent(
        model_name=args.model,
        data_dir=args.data_dir,
        credentials_path=args.credentials,
        use_wq_grounding=not args.no_wq_grounding,
        allow_ungrounded_candidates=args.allow_ungrounded_candidates,
    )

    if args.topics_batch > 0:
        topics = DEFAULT_TOPICS[:args.topics_batch]
        added = agent.conduct_research_batch(topics)
    else:
        topic = args.topic or DEFAULT_TOPICS[0]
        added = agent.conduct_research(topic)

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"🏁 Research complete — {added} new candidates queued")
    logger.info(f"   Candidate pool file: {agent.validator.save_path}")
    logger.info(f"   HTML report:         {agent.html_path}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
