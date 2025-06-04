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

    snippets: list[ResearchSnippet] = []

    # Always run SmartSearcher (cheap) – we call its .invoke synchronously because
    # SmartSearcher already performs internal concurrency.
    smart_task = SmartSearcher(num_searches_to_run=1, num_sites_per_search=5).invoke(query)

    # Always run AskNews summaries if keys exist
    asknews_enabled = os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")
    ask_task = (
        AskNewsSearcher().get_formatted_news_async(query) if asknews_enabled else None
    )

    # Deep search optional
    deep_task = None
    if depth == "deep":
        # Perplexity pro search returns a string via misc_tools wrapper
        deep_task = perplexity_pro_search(query)

    # gather only non-None tasks
    tasks = [smart_task]
    if ask_task:
        tasks.append(ask_task)
    if deep_task:
        tasks.append(deep_task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Map back to sources.
    idx = 0
    snippets.append({"source": "smart_search", "text": str(results[idx])})
    idx += 1

    if ask_task:
        ask_result = results[idx]
        snippets.append({"source": "asknews", "text": str(ask_result)})
        idx += 1

    if deep_task:
        deep_result = results[idx]
        snippets.append({"source": "perplexity", "text": str(deep_result)})

    return _dedupe(snippets)