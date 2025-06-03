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

        # Debug model initialization
        print(f"🔍 PERPLEXITY DEBUG: Initializing AgentSdkLlm with model: {model}")
        logger.warning(f"PERPLEXITY DEBUG: Initializing AgentSdkLlm with model: {model}")

        # Then configure for Perplexity if needed
        if "perplexity" in model:
            print(f"🔍 PERPLEXITY DEBUG: Configuring for Perplexity model")
            logger.warning(f"PERPLEXITY DEBUG: Configuring for Perplexity model")
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
        print(f"🔍 PERPLEXITY DEBUG: _order_messages_for_perplexity called with {len(messages)} messages")
        logger.warning(f"PERPLEXITY DEBUG: _order_messages_for_perplexity called with {len(messages)} messages")

        # Check if messages are already ordered (to prevent double-ordering)
        if hasattr(messages, '_perplexity_ordered'):
            print(f"🔍 PERPLEXITY DEBUG: Messages already ordered, skipping")
            logger.warning(f"PERPLEXITY DEBUG: Messages already ordered, skipping")
            return messages

        print(f"🔍 PERPLEXITY DEBUG: Input messages: {[msg.get('role', 'unknown') for msg in messages]}")
        logger.warning(f"PERPLEXITY DEBUG: Input messages: {[msg.get('role', 'unknown') for msg in messages]}")

        # First, separate system messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        print(f"🔍 PERPLEXITY DEBUG: Found {len(system_messages)} system messages, {len(other_messages)} other messages")
        logger.warning(f"PERPLEXITY DEBUG: Found {len(system_messages)} system messages, {len(other_messages)} other messages")

        # Ensure alternating user/assistant messages
        ordered_messages = []
        for i, msg in enumerate(other_messages):
            if i > 0 and msg.get("role") == other_messages[i-1].get("role"):
                # If we have consecutive messages of the same role, combine them
                if ordered_messages:
                    ordered_messages[-1]["content"] += "\n" + str(msg.get("content", ""))
                    print(f"🔍 PERPLEXITY DEBUG: Combined consecutive {msg.get('role')} message")
                    logger.warning(f"PERPLEXITY DEBUG: Combined consecutive {msg.get('role')} message")
                else:
                    ordered_messages.append(msg)
            else:
                ordered_messages.append(msg)

        # Ensure we start with a user message if there are no system messages
        if not system_messages and ordered_messages and ordered_messages[0].get("role") != "user":
            # Add an empty user message at the start
            ordered_messages.insert(0, {"role": "user", "content": ""})
            print(f"🔍 PERPLEXITY DEBUG: Added empty user message at start")
            logger.warning(f"PERPLEXITY DEBUG: Added empty user message at start")

        final_messages = system_messages + ordered_messages

        # Mark as ordered to prevent double-ordering
        try:
            final_messages._perplexity_ordered = True
        except AttributeError:
            # If we can't set the attribute (list doesn't support it), that's okay
            pass

        print(f"🔍 PERPLEXITY DEBUG: Output messages: {[msg.get('role', 'unknown') for msg in final_messages]}")
        logger.warning(f"PERPLEXITY DEBUG: Output messages: {[msg.get('role', 'unknown') for msg in final_messages]}")
        print(f"🔍 PERPLEXITY DEBUG: Full messages: {final_messages}")
        logger.warning(f"PERPLEXITY DEBUG: Full messages: {final_messages}")

        return final_messages

    async def _fetch_response(
        self, *args, **kwargs
    ) -> Tuple[Any, Any]:
        """
        Override _fetch_response to ensure proper message ordering for Perplexity at the lowest level.
        """
        print(f"🔍 PERPLEXITY DEBUG: _fetch_response called with model: {self.model}")
        logger.warning(f"PERPLEXITY DEBUG: _fetch_response called with model: {self.model}")
        print(f"🔍 PERPLEXITY DEBUG: _fetch_response args length: {len(args)}")
        logger.warning(f"PERPLEXITY DEBUG: _fetch_response args length: {len(args)}")

        # Only apply ordering if this is a Perplexity model and we have messages to order
        if "perplexity" in self.model:
            print(f"🔍 PERPLEXITY DEBUG: This is a Perplexity model")
            logger.warning(f"PERPLEXITY DEBUG: This is a Perplexity model")

            if args:
                print(f"🔍 PERPLEXITY DEBUG: Args available, checking first arg")
                logger.warning(f"PERPLEXITY DEBUG: Args available, checking first arg")
                messages = args[0]
                print(f"🔍 PERPLEXITY DEBUG: First arg type: {type(messages)}")
                logger.warning(f"PERPLEXITY DEBUG: First arg type: {type(messages)}")

                if isinstance(messages, list):
                    print(f"🔍 PERPLEXITY DEBUG: First arg is list with {len(messages)} items")
                    logger.warning(f"PERPLEXITY DEBUG: First arg is list with {len(messages)} items")

                    if messages and isinstance(messages[0], dict):
                        print(f"🔍 PERPLEXITY DEBUG: First item is dict: {messages[0]}")
                        logger.warning(f"PERPLEXITY DEBUG: First item is dict: {messages[0]}")

                        if any(msg.get("role") for msg in messages):
                            print(f"🔍 PERPLEXITY DEBUG: Applying message ordering in _fetch_response")
                            logger.warning(f"PERPLEXITY DEBUG: Applying message ordering in _fetch_response")
                            ordered_messages = self._order_messages_for_perplexity(messages)
                            # Replace the first argument with ordered messages
                            args = (ordered_messages,) + args[1:]
                        else:
                            print(f"🔍 PERPLEXITY DEBUG: No role found in messages")
                            logger.warning(f"PERPLEXITY DEBUG: No role found in messages")
                    else:
                        print(f"🔍 PERPLEXITY DEBUG: First item not a dict or list empty")
                        logger.warning(f"PERPLEXITY DEBUG: First item not a dict or list empty")
                else:
                    print(f"🔍 PERPLEXITY DEBUG: First arg not a list")
                    logger.warning(f"PERPLEXITY DEBUG: First arg not a list")
            else:
                print(f"🔍 PERPLEXITY DEBUG: No args available")
                logger.warning(f"PERPLEXITY DEBUG: No args available")
        else:
            print(f"🔍 PERPLEXITY DEBUG: Not a Perplexity model")
            logger.warning(f"PERPLEXITY DEBUG: Not a Perplexity model")

        return await super()._fetch_response(*args, **kwargs)

    async def stream_response(
        self, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        Override stream_response to ensure proper message ordering for Perplexity.
        Note: We rely on _fetch_response to do the actual ordering to avoid double-ordering.
        """
        print(f"🔍 PERPLEXITY DEBUG: stream_response called with model: {self.model}")
        logger.warning(f"PERPLEXITY DEBUG: stream_response called with model: {self.model}")

        async for event in super().stream_response(*args, **kwargs):
            yield event

    async def get_response(self, *args, **kwargs):  # NOSONAR
        ModelTracker.give_cost_tracking_warning_if_needed(self.model)

        print(f"🔍 PERPLEXITY DEBUG: get_response called with model: {self.model}")
        logger.warning(f"PERPLEXITY DEBUG: get_response called with model: {self.model}")

        # Handle message ordering for Perplexity models
        if "perplexity" in self.model and "messages" in kwargs:
            print(f"🔍 PERPLEXITY DEBUG: Applying message ordering in get_response")
            logger.warning(f"PERPLEXITY DEBUG: Applying message ordering in get_response")
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
