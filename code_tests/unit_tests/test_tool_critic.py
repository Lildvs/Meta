import os
import asyncio
from forecasting_tools.forecast_helpers.tool_critic import ToolCritic
from forecasting_tools.forecast_helpers.research_orchestrator import orchestrate_research
import types
from forecasting_tools.forecast_helpers import research_orchestrator


async def _dummy_perplexity(query: str):
    _dummy_perplexity.calls += 1  # type: ignore[attr-defined]
    return []

_dummy_perplexity.calls = 0  # type: ignore[attr-defined]


async def test_tool_critic_paths(monkeypatch):
    # Negative path – critic returns 0 so deep search skipped
    os.environ["CRITIC_MOCK"] = "FALSE"
    from forecasting_tools.agents_and_tools import misc_tools

    monkeypatch.setattr(misc_tools, "perplexity_pro_search", _dummy_perplexity, raising=True)
    monkeypatch.setattr(research_orchestrator, "perplexity_pro_search", _dummy_perplexity, raising=True)

    _dummy_perplexity.calls = 0  # reset
    res = await orchestrate_research("Who is the president of France?", depth="deep")
    assert _dummy_perplexity.calls == 0, "Deep search should have been skipped"

    # Positive path – critic returns 1.0 → deep search executed
    os.environ["CRITIC_MOCK"] = "TRUE"
    _dummy_perplexity.calls = 0
    # patch already in place
    res2 = await orchestrate_research("Latest Nvidia quarterly revenue?", depth="deep")
    assert _dummy_perplexity.calls == 1, "Deep search should have been called once"

    # Cleanup env var
    os.environ.pop("CRITIC_MOCK", None)