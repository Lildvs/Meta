import asyncio
import os
from types import SimpleNamespace

from forecasting_tools.forecast_helpers import research_orchestrator as ro


class DummySmart(ro.SmartSearcher):  # type: ignore[misc]
    def __init__(self, *args, **kwargs):  # noqa: D401
        super().__init__(*args, **kwargs)
        self._images = [
            "http://example.com/a.png",
            "http://example.com/b.png",
        ]

    async def invoke(self, prompt):  # noqa: D401
        return "dummy smart results"


async def _fake_ocr(url: str):  # noqa: D401
    return f"description for {url}"


async def test_vision_snippets(monkeypatch):
    # patch SmartSearcher in orchestrator context
    monkeypatch.setattr(ro, "SmartSearcher", DummySmart, raising=True)

    # patch ImageOcrSearcher.describe to avoid external calls
    from forecasting_tools.forecast_helpers import image_ocr_searcher as ios

    monkeypatch.setattr(ios.ImageOcrSearcher, "describe", staticmethod(_fake_ocr), raising=True)

    os.environ["MAX_IMAGES"] = "2"

    snippets = await ro.orchestrate_research("test vision", depth="quick")

    img_snips = [s for s in snippets if s["source"] == "image_ocr"]
    assert len(img_snips) == 2

    # cleanup
    os.environ.pop("MAX_IMAGES")