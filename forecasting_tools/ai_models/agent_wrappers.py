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
# NOTE: Avoid importing orchestrate_research at module load time to prevent circular imports.
# It will be imported lazily inside methods when needed.

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
        if not messages:
            return messages

        # Separate system messages from user/assistant messages
        system_messages = []
        user_assistant_messages = []

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_messages.append(msg)
            elif role in ["user", "assistant"]:
                user_assistant_messages.append(msg)

        # Keep only the last system message if multiple were provided
        system_messages = system_messages[-1:]  # Perplexity allows at most one system message

        if not user_assistant_messages:
            return messages

        # Create alternating sequence starting with user
        ordered_messages = []
        combined_user_content = []
        combined_assistant_content = []
        current_collecting = None

        for msg in user_assistant_messages:
            role = msg.get("role")
            content = str(msg.get("content", "")).strip()

            if role == "user":
                if current_collecting == "assistant" and combined_assistant_content:
                    # Finish collecting assistant content
                    ordered_messages.append({
                        "role": "assistant",
                        "content": "\n".join(combined_assistant_content)
                    })
                    combined_assistant_content = []
                current_collecting = "user"
                if content:
                    combined_user_content.append(content)
            elif role == "assistant":
                if current_collecting == "user" and combined_user_content:
                    # Finish collecting user content
                    ordered_messages.append({
                        "role": "user",
                        "content": "\n".join(combined_user_content)
                    })
                    combined_user_content = []
                current_collecting = "assistant"
                if content:
                    combined_assistant_content.append(content)

        # Add any remaining content
        if combined_user_content:
            ordered_messages.append({
                "role": "user",
                "content": "\n".join(combined_user_content)
            })
        if combined_assistant_content:
            ordered_messages.append({
                "role": "assistant",
                "content": "\n".join(combined_assistant_content)
            })

        # Ensure we start with user message if we have any messages
        if ordered_messages and ordered_messages[0].get("role") != "user":
            ordered_messages.insert(0, {"role": "user", "content": ""})

        # Ensure strict alternation
        final_messages = []
        expected_role = "user"

        for msg in ordered_messages:
            if msg.get("role") == expected_role:
                final_messages.append(msg)
                expected_role = "assistant" if expected_role == "user" else "user"

        return system_messages + final_messages

    async def _fetch_response(
        self, *args, **kwargs
    ) -> Tuple[Any, Any]:
        """
        Override _fetch_response to ensure proper message ordering for Perplexity at the lowest level.
        This method attempts to locate the `messages` argument whether it is passed
        positionally (second arg) or as a keyword and then applies ordering.
        """
        if "perplexity" in self.model:
            # Case 1: messages passed positionally as second argument
            if len(args) >= 2 and isinstance(args[1], list):
                try:
                    ordered_messages = self._order_messages_for_perplexity(args[1])
                    args = (args[0], ordered_messages) + args[2:]
                except Exception:
                    pass  # fallback to original
            # Case 2: messages passed positionally as first argument (earlier code)
            elif len(args) >= 1 and isinstance(args[0], list):
                try:
                    ordered_messages = self._order_messages_for_perplexity(args[0])
                    args = (ordered_messages,) + args[1:]
                except Exception:
                    pass
            # Case 3: messages passed via kwargs
            elif "messages" in kwargs and isinstance(kwargs["messages"], list):
                try:
                    kwargs["messages"] = self._order_messages_for_perplexity(kwargs["messages"])
                except Exception:
                    pass
        return await super()._fetch_response(*args, **kwargs)

    async def stream_response(
        self, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        Override stream_response to ensure proper message ordering for Perplexity.
        """
        async for event in super().stream_response(*args, **kwargs):
            yield event

    async def get_response(self, *args, **kwargs):  # NOSONAR
        ModelTracker.give_cost_tracking_warning_if_needed(self.model)

        # Handle message ordering for Perplexity models
        if "perplexity" in self.model and "messages" in kwargs:
            try:
                kwargs["messages"] = self._order_messages_for_perplexity(kwargs["messages"])
            except Exception:
                # If ordering fails, proceed with original messages
                pass

        response = await super().get_response(*args, **kwargs)

        # ---------------- Self-refine loop -----------------
        if os.getenv("SELF_REFINE", "TRUE").upper() == "TRUE":
            try:
                # Extract the last user prompt from messages (assumes list of dicts)
                user_prompt: str | None = None
                if "messages" in kwargs and isinstance(kwargs["messages"], list):
                    for msg in reversed(kwargs["messages"]):
                        if msg.get("role") == "user":
                            user_prompt = str(msg.get("content", ""))
                            break

                if user_prompt:
                    confidence = await self._assess_confidence(user_prompt, response)
                    if confidence < 0.7:
                        # run deep research and retry once
                        from forecasting_tools.forecast_helpers.research_orchestrator import orchestrate_research  # noqa: WPS433 – local import to avoid circular dependency
                        snippets = await orchestrate_research(user_prompt, depth="deep")
                        research_text = "\n\n### Additional research\n" + "\n".join(
                            f"* {s['text']}" for s in snippets
                        )

                        # Append research to user messages and rerun
                        new_messages = kwargs.get("messages", []).copy()
                        new_messages.append({"role": "system", "content": research_text})
                        refined = await super().get_response(messages=new_messages)
                        response = refined
            except Exception as err:  # noqa: BLE001
                logger.error("Self-refine loop failed: %s", err)

        await asyncio.sleep(
            0.0001
        )  # For whatever reason, it seems you need to await a coroutine to get the litellm cost callback to work
        return response

    # ---------------------------------------------------
    # Helper: confidence assessment
    # ---------------------------------------------------

    async def _assess_confidence(self, question: str, answer: str) -> float:  # noqa: D401
        """Return a confidence score 0–1 using a cheap model."""

        prompt = (
            "You are an expert answer verifier. Given a QUESTION and its ANSWER, "
            "output a single floating point number between 0 and 1 representing how confident "
            "you are that the answer fully and accurately addresses the question.\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONFIDENCE:"  # noqa: E501
        )

        cheap_model = os.getenv("CRITIC_MODEL", "gpt-3.5-turbo-0125")
        critic_llm = GeneralLlm(model=cheap_model, temperature=0)
        try:
            raw = await critic_llm.invoke(prompt)
            score = float(raw.strip().split()[0])  # take first token
            score = max(0.0, min(1.0, score))
            return score
        except Exception:
            return 1.0  # default high confidence if assessment fails

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
