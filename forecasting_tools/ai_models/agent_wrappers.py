import asyncio
from typing import Callable, overload, Any
import os

import nest_asyncio
from agents import Agent, FunctionTool, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.tool import ToolFunction

from forecasting_tools.ai_models.model_tracker import ModelTracker

nest_asyncio.apply()


class AgentSdkLlm(LitellmModel):
    """
    Wrapper around openai-agent-sdk's LiteLlm Model for later extension.
    This class ensures that direct API calls (e.g., to Perplexity) are correctly configured.
    """
    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        **litellm_kwargs: Any,
    ):
        updated_litellm_kwargs = litellm_kwargs.copy()

        if "perplexity" in model:
            # Ensure litellm uses Perplexity API directly, not via OpenRouter
            updated_litellm_kwargs["custom_llm_provider"] = "perplexity"
            # Only set api_key if not already provided in litellm_kwargs
            if "api_key" not in updated_litellm_kwargs or updated_litellm_kwargs["api_key"] is None:
                updated_litellm_kwargs["api_key"] = os.getenv("PERPLEXITY_API_KEY")
            # Only set base_url if not already provided
            if "base_url" not in updated_litellm_kwargs or updated_litellm_kwargs["base_url"] is None:
                 updated_litellm_kwargs["base_url"] = "https://api.perplexity.ai"
            # Standard headers for Perplexity, ensure Content-Type if others are added
            if "extra_headers" not in updated_litellm_kwargs:
                updated_litellm_kwargs["extra_headers"] = {}
            updated_litellm_kwargs["extra_headers"]["Content-Type"] = "application/json"


        # Note: Similar blocks could be added here for other direct providers
        # if AgentSdkLlm needs to support them directly (e.g., exa, direct OpenAI, Anthropic).

        super().__init__(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            **updated_litellm_kwargs, # Pass the potentially modified kwargs
        )

    async def get_response(self, *args, **kwargs):  # NOSONAR
        ModelTracker.give_cost_tracking_warning_if_needed(self.model)
        response = await super().get_response(*args, **kwargs)
        await asyncio.sleep(
            0.0001
        )  # For whatever reason, it seems you need to await a coroutine to get the litellm cost callback to work
        return response


AgentRunner = Runner  # Alias for Runner for later extension
AgentTool = FunctionTool  # Alias for FunctionTool for later extension
AiAgent = Agent  # Alias for Agent for later extension


@overload
def agent_tool(func: ToolFunction[...], **kwargs) -> FunctionTool:
    """Overload for usage as @function_tool (no parentheses)."""
    ...


@overload
def agent_tool(**kwargs) -> Callable[[ToolFunction[...]], FunctionTool]:
    """Overload for usage as @function_tool(...)."""
    ...


def agent_tool(
    func: ToolFunction[...] | None = None, **kwargs
) -> AgentTool | Callable[[ToolFunction[...]], AgentTool]:
    if func is None:

        def decorator(f):
            return function_tool(f, **kwargs)

        return decorator
    return function_tool(func, **kwargs)
