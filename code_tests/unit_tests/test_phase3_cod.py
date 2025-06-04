import asyncio
import pytest

from forecasting_tools.forecast_helpers.cod_summariser import compress as cod_compress


def _dummy_snippet(i: int) -> dict[str, str]:
    return {
        "source": f"dummy_{i}",
        "text": f"This is some detailed content number {i}. It discusses a variety of interesting facts about item {i} and includes many supporting details."  # noqa: E501
    }


@pytest.mark.asyncio
async def test_cod_summariser_under_400_tokens(monkeypatch) -> None:  # noqa: D401
    # Monkey-patch GeneralLlm.invoke so the test does not hit real LLMs / need API keys
    async def _fake_invoke(self, prompt):  # noqa: D401
        return "This is a condensed summary [1][2][3][4][5]."

    from forecasting_tools.ai_models.general_llm import GeneralLlm

    monkeypatch.setattr(GeneralLlm, "invoke", _fake_invoke, raising=True)

    snippets = [_dummy_snippet(i) for i in range(6)]
    texts = [s["text"] for s in snippets]
    summary = await cod_compress(texts)
    assert summary, "CoD summariser returned empty string"
    approx_tokens = len(summary.split())  # word count as proxy
    assert approx_tokens < 400, f"Summary too long: ~{approx_tokens} tokens"