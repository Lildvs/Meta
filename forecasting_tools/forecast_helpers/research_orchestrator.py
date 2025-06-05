from __future__ import annotations

"""Unified research orchestrator.

This helper fires several existing search capabilities in parallel and returns
merged snippets so that higher-level agents can cite them easily.

Currently supports two depth settings:
• quick  – SmartSearcher + AskNews news-summaries
• deep   – quick sources **plus** Perplexity deep research

The orchestrator is purposely thin: it delegates heavy lifting to the existing
helpers and merely merges / deduplicates results.
"""

import asyncio
import logging
import os
from typing import TypedDict, Literal, List

from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecast_helpers.asknews_searcher import (
    AskNewsSearcher,
)
from forecasting_tools.agents_and_tools.misc_tools import (
    perplexity_pro_search,  # deep research (async agent-tool function)
    perplexity_quick_search,  # unused for now but could support "medium" depth
)
from forecasting_tools.forecast_helpers.cod_summariser import compress as cod_compress

logger = logging.getLogger(__name__)

Depth = Literal["quick", "deep"]


class ResearchSnippet(TypedDict):
    source: str  # e.g. "smart_search", "asknews", "perplexity"
    text: str


def _dedupe(snippets: List[ResearchSnippet]) -> List[ResearchSnippet]:
    """Remove duplicate texts (exact match). Preserve first occurrence order."""

    seen: set[str] = set()
    deduped: list[ResearchSnippet] = []
    for s in snippets:
        if s["text"] not in seen:
            seen.add(s["text"])
            deduped.append(s)
    return deduped


async def orchestrate_research(query: str, depth: Depth = "quick") -> List[ResearchSnippet]:
    """Run the selected research tools in parallel.

    Parameters
    ----------
    query : str
        The user question or topic.
    depth : "quick" | "deep"
        How exhaustive the search should be.

    Returns
    -------
    list[ResearchSnippet]
        Merged, deduplicated snippets. Each snippet is a dict with `source` and `text`.
    """

    if not query:
        raise ValueError("Query must be non-empty")

    async def _compute() -> List[ResearchSnippet]:
        snippets: list[ResearchSnippet] = []

        smart_searcher = SmartSearcher(num_searches_to_run=1, num_sites_per_search=5)
        smart_future = smart_searcher.invoke(query)

        asknews_enabled = os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")
        ask_task = (
            AskNewsSearcher().get_formatted_news_async(query) if asknews_enabled else None
        )

        deep_task = None
        if depth == "deep":
            from forecasting_tools.forecast_helpers.tool_critic import ToolCritic
            critic = ToolCritic()
            if await critic.should_deep_search(query):
                deep_task = perplexity_pro_search(query)

        tasks: list[asyncio.Future] = [smart_future]
        if ask_task:
            tasks.append(ask_task)
        if deep_task:
            tasks.append(deep_task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                continue
            if isinstance(res, list):
                snippets.extend(res)
            else:
                src_name = ["smart_search", "asknews", "perplexity"][i]
                snippets.append({"source": src_name, "text": str(res)})

        deduped = _dedupe(snippets)

        # ----------------------------------------------------
        # Vision OCR on images collected by SmartSearcher
        # ----------------------------------------------------
        from forecasting_tools.forecast_helpers.image_ocr_searcher import ImageOcrSearcher  # noqa: WPS433

        max_imgs = int(os.getenv("MAX_IMAGES", "3"))
        if smart_searcher.images and max_imgs > 0:
            ocr = ImageOcrSearcher()
            img_tasks = [ocr.describe(u) for u in smart_searcher.images[:max_imgs]]
            ocr_results = await asyncio.gather(*img_tasks)
            for desc, url in zip(ocr_results, smart_searcher.images[:max_imgs]):
                if desc:
                    deduped.append({
                        "source": "image_ocr",
                        "text": f"{desc} ([img]({url}))",
                    })

        if os.getenv("ENABLE_COD_SUMMARY", "TRUE").upper() == "TRUE" and len(deduped) > 6:
            texts = [s["text"] for s in deduped]
            try:
                summary = await cod_compress(texts)
                deduped = [{"source": "cod_summary", "text": summary}]
            except Exception as err:  # noqa: BLE001
                logger.warning("CoD summariser failed: %s", err)
        return deduped

    from forecasting_tools.forecast_helpers.cache import EmbeddingCache

    cache_key = f"{depth}:{query.strip().lower()}"
    cache = EmbeddingCache()

    snippets = await cache.get_or_fetch(cache_key, _compute)

    logger.info("Research cache hit-ratio %.1f%%", EmbeddingCache.hit_ratio() * 100)
    return snippets