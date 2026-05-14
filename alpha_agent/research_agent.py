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
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .seed_validator import SeedValidator

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
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in .env.")

        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.html_path = self.data_dir / "research_report.html"
        self.md_path = self.data_dir / "seeds_and_hypotheses.md"

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

        raw_text = self._call_llm(topic)
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

    def _call_llm(self, topic: str) -> Optional[str]:
        from google.genai import types

        user_prompt = (
            f"请针对以下主题产出研报与 seed candidates。\n\n"
            f"主题：【{topic}】\n\n"
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
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose, log_file=Path(args.data_dir) / "research_agent.log")

    agent = ResearchAgent(model_name=args.model, data_dir=args.data_dir)

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
