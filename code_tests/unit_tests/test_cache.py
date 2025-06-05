from forecasting_tools.forecast_helpers.cache import EmbeddingCache
import asyncio


async def test_embedding_cache_get_or_fetch() -> None:
    cache = EmbeddingCache()
    calls: list[int] = []

    async def _fetch():
        calls.append(1)
        return {"val": 42}

    key = "unit-test-key"
    first = await cache.get_or_fetch(key, _fetch)
    second = await cache.get_or_fetch(key, _fetch)

    assert first == second == {"val": 42}
    # fetcher should have been called exactly once
    assert len(calls) == 1