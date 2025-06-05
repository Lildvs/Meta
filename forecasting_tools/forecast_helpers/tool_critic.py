from __future__ import annotations

"""Cost-aware tool critic.

Decides, for a given user question, whether paying for an expensive
Perplexity *deep* search is worthwhile.
"""

from typing import Final
import os
import json
import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)

_CRITIC_MODEL: Final[str] = os.getenv("CRITIC_MODEL", "gpt-3.5-turbo-0125")
_COST_THRESHOLD: Final[float] = float(os.getenv("CRITIC_COST_THRESHOLD", "0.20"))


class ToolCritic:  # noqa: D401
    """LLM-based heuristic deciding if *deep* search should be invoked."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or _CRITIC_MODEL
        self.llm = GeneralLlm(model=self.model_name, temperature=0)

    # ------------------------- public API -------------------------
    async def score(self, question: str) -> float:  # noqa: D401
        """Return usefulness score 0-1 for running deep search."""
        # Mocking shortcuts for tests / offline mode
        mock_flag = os.getenv("CRITIC_MOCK")
        if mock_flag is not None:
            return 1.0 if mock_flag.upper() == "TRUE" else 0.0

        prompt = self._build_prompt(question)
        try:
            raw = await self.llm.invoke(prompt)
            score = json.loads(raw).get("score")  # type: ignore[arg-type]
            return float(score)
        except Exception as err:  # noqa: BLE001
            logger.warning("ToolCritic failed (%s) – defaulting to deep search", err)
            return 1.0  # safe side

    async def should_deep_search(self, question: str, cost_threshold: float | None = None) -> bool:  # noqa: D401
        threshold = cost_threshold or _COST_THRESHOLD
        sc = await self.score(question)
        decision = sc >= threshold
        logger.info("ToolCritic score=%.2f decision=%s (threshold %.2f)", sc, decision, threshold)
        return decision

    # ------------------------- helpers ----------------------------
    @staticmethod
    def _build_prompt(question: str) -> str:  # noqa: D401
        system = (
            "You are a cost-aware research planner."
        )
        user = f"""QUESTION = "{question}"
RULES:
1. Output only JSON: {{"score": float 0-1}}
2. Score high if answer needs newest facts, high uncertainty or legal stakes.
3. Score low if question is trivial, subjective, or already well-known.

EXAMPLES:
Q: "Summarize WW2 major events" → 0.05
Q: "Latest Nvidia quarterly revenue?" → 0.8"""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]