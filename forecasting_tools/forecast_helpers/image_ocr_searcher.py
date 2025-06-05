from __future__ import annotations

"""Download an image and generate a short description via vision LLM.

Uses EmbeddingCache (key = sha256(url)) so the same image is described only once.
"""

import hashlib
import logging
import os
from typing import Final

import aiohttp

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.ai_utils.openai_utils import VisionMessageData
from forecasting_tools.forecast_helpers.cache import EmbeddingCache

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_BYTES: Final[int] = 1 * 1024 * 1024  # 1 MB
_VISION_MODEL: Final[str] = os.getenv("VISION_MODEL", "gpt-4o-mini-vision")
_ENABLE_VISION: Final[bool] = os.getenv("ENABLE_VISION", "TRUE").upper() == "TRUE"
_MAX_WORDS: Final[int] = 120


class ImageOcrSearcher:  # noqa: D401
    """Describe web images in plain text."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or _VISION_MODEL
        self.llm = GeneralLlm(model=self.model_name, temperature=0)
        self.cache = EmbeddingCache()

    async def describe(self, url: str) -> str:  # noqa: D401
        """Return plain-text description of *url* (<=120 words)."""
        if not _ENABLE_VISION:
            return ""

        key = f"img:{hashlib.sha256(url.encode()).hexdigest()}"

        async def _fetch():
            # Download bytes (HEAD first) just to validate size
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        content_type = resp.headers.get("Content-Type", "")
                        if not content_type.startswith("image"):
                            return ""  # not an image → skip
                        data = await resp.content.read(_MAX_DOWNLOAD_BYTES + 1)
                        if len(data) > _MAX_DOWNLOAD_BYTES:
                            return ""  # too large
            except Exception as err:  # noqa: BLE001
                logger.debug("Image download failed %s", err)
                return ""

            prompt = "Describe this chart or image in plain text, max 120 words."
            try:
                message = VisionMessageData(image_url=url, prompt=prompt)
                desc = await self.llm.invoke(message)
                return desc.strip()
            except Exception as err:  # noqa: BLE001
                logger.warning("Vision model failed: %s", err)
                return ""

        return await self.cache.get_or_fetch(key, _fetch)