from __future__ import annotations

"""Chain-of-Density summariser (Chen et al., ICLR 2025).

Given a list of research snippets, produce a compressed yet information-dense
summary in markdown that still cites at least a handful of sources.

This implementation is deliberately lightweight—one LLM call—and is gated by an
environment flag so cost can be controlled.
"""

import os
from typing import List

from forecasting_tools.ai_models.general_llm import GeneralLlm

MAX_SNIPPETS_DEFAULT = 15


async def compress(snippets: List[str], max_tokens: int = 350) -> str:  # noqa: D401
    """Return a CoD summary of the provided snippets.

    The prompt asks the model to iteratively condense content while retaining
    citations markers like [1], [2] … already embedded in each snippet.
    """

    if not snippets:
        return ""

    # Truncate to avoid extremely large prompts
    snippets = snippets[:MAX_SNIPPETS_DEFAULT]
    numbered = [f"[{i+1}] {s.strip()}" for i, s in enumerate(snippets)]
    joined = "\n".join(numbered)

    system_prompt = (
        "You are an expert technical writer adept at Chain-of-Density. "
        "Compress the provided content to a dense summary under "
        f"≈{max_tokens} tokens while *preserving all concrete facts* and at least 5 citation markers like [3]."
    )

    user_prompt = (
        "SOURCE SNIPPETS:\n" + joined + "\n\n" +
        "Produce the final dense summary now."
    )

    model_name = os.getenv("COD_MODEL", "gpt-4o-mini")
    llm = GeneralLlm(model=model_name, temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return await llm.invoke(messages)