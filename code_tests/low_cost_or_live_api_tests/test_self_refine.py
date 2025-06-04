import asyncio
import pytest
from forecasting_tools.ai_models.agent_wrappers import AgentSdkLlm


@pytest.mark.asyncio
async def test_self_refine_triggers_once(monkeypatch):
    """Ensure that low confidence triggers a second call once."""

    call_counter = {"calls": 0}

    async def fake_get_response(self, *args, **kwargs):  # type: ignore[override]
        call_counter["calls"] += 1
        return f"dummy answer {call_counter['calls']}"

    async def fake_confidence(self, q, a):  # noqa: D401
        return 0.3  # always low confidence

    async def fake_orchestrate(query, depth="deep"):
        return [{"source": "mock", "text": "extra info"}]

    # Patch internals
    monkeypatch.setattr(AgentSdkLlm, "_assess_confidence", fake_confidence)
    monkeypatch.setattr(AgentSdkLlm, "_order_messages_for_perplexity", lambda self, m: m)
    monkeypatch.setattr(AgentSdkLlm, "_AgentSdkLlm__classcell__", None, raising=False)
    monkeypatch.setattr(AgentSdkLlm, "get_response", fake_get_response, raising=False)
    import forecasting_tools.forecast_helpers.research_orchestrator as ro
    monkeypatch.setattr(ro, "orchestrate_research", fake_orchestrate)

    llm = AgentSdkLlm(model="gpt-3.5-turbo-0125")
    result = await llm.get_response(messages=[{"role": "user", "content": "What is AI?"}])

    assert call_counter["calls"] == 2, "Should have refined once"
    assert "dummy answer 2" == result