import asyncio
from typing import Callable, overload, Any, List, Dict, AsyncGenerator, Tuple
import os
import litellm  # Add litellm import
import logging

import nest_asyncio
from agents import Agent, FunctionTool, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.tool import ToolFunction

from forecasting_tools.ai_models.model_tracker import ModelTracker

logger = logging.getLogger(__name__)
nest_asyncio.apply()

# Set global litellm configuration
litellm.drop_params = True

class AgentSdkLlm(LitellmModel):
    """
    Wrapper around openai-agent-sdk's LiteLlm Model for later extension.
    This class ensures that direct API calls (e.g., to Perplexity) are correctly configured.
    """
    def __init__(
        self,
        model: str,
        **litellm_kwargs: Any,
    ):
        # First initialize with just the model
        super().__init__(model=model)

        # Then configure for Perplexity if needed
        if "perplexity" in model:
            # Configure litellm to use Perplexity API directly
            self.api_key = os.getenv("PERPLEXITY_API_KEY")
            self.base_url = "https://api.perplexity.ai"
            self.extra_headers = {"Content-Type": "application/json"}

            # Update any additional kwargs that were passed
            for key, value in litellm_kwargs.items():
                setattr(self, key, value)

    def _order_messages_for_perplexity(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensures messages follow Perplexity's requirement of alternating user/assistant roles
        after any system messages.
        """
        # Check if messages are already ordered (to prevent double-ordering)
        if hasattr(messages, '_perplexity_ordered'):
            logger.info(f"PERPLEXITY DEBUG: Messages already ordered, skipping")
            return messages

        logger.info(f"PERPLEXITY DEBUG: Input messages: {[msg.get('role', 'unknown') for msg in messages]}")

        # First, separate system messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        # Ensure alternating user/assistant messages
        ordered_messages = []
        for i, msg in enumerate(other_messages):
            if i > 0 and msg.get("role") == other_messages[i-1].get("role"):
                # If we have consecutive messages of the same role, combine them
                if ordered_messages:
                    ordered_messages[-1]["content"] += "\n" + str(msg.get("content", ""))
                else:
                    ordered_messages.append(msg)
            else:
                ordered_messages.append(msg)

        # Ensure we start with a user message if there are no system messages
        if not system_messages and ordered_messages and ordered_messages[0].get("role") != "user":
            # Add an empty user message at the start
            ordered_messages.insert(0, {"role": "user", "content": ""})

        final_messages = system_messages + ordered_messages

        # Mark as ordered to prevent double-ordering
        try:
            final_messages._perplexity_ordered = True
        except AttributeError:
            # If we can't set the attribute (list doesn't support it), that's okay
            pass

        logger.info(f"PERPLEXITY DEBUG: Output messages: {[msg.get('role', 'unknown') for msg in final_messages]}")
        logger.info(f"PERPLEXITY DEBUG: Full messages: {final_messages}")

        return final_messages

    async def _fetch_response(
        self, *args, **kwargs
    ) -> Tuple[Any, Any]:
        """
        Override _fetch_response to ensure proper message ordering for Perplexity at the lowest level.
        """
        # Only apply ordering if this is a Perplexity model and we have messages to order
        if "perplexity" in self.model and args:
            messages = args[0]
            if isinstance(messages, list) and messages and isinstance(messages[0], dict) and any(msg.get("role") for msg in messages):
                logger.info(f"PERPLEXITY DEBUG: Applying message ordering in _fetch_response")
                ordered_messages = self._order_messages_for_perplexity(messages)
                # Replace the first argument with ordered messages
                args = (ordered_messages,) + args[1:]

        return await super()._fetch_response(*args, **kwargs)

    async def stream_response(
        self, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        Override stream_response to ensure proper message ordering for Perplexity.
        Note: We rely on _fetch_response to do the actual ordering to avoid double-ordering.
        """
        async for event in super().stream_response(*args, **kwargs):
            yield event

    async def get_response(self, *args, **kwargs):  # NOSONAR
        ModelTracker.give_cost_tracking_warning_if_needed(self.model)

        # Handle message ordering for Perplexity models
        if "perplexity" in self.model and "messages" in kwargs:
            logger.info(f"PERPLEXITY DEBUG: Applying message ordering in get_response")
            kwargs["messages"] = self._order_messages_for_perplexity(kwargs["messages"])

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
