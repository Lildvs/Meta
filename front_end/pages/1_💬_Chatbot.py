from __future__ import annotations

import logging
import os
import time

import streamlit as st
from agents import Agent, RunItem, Runner, Tool, trace
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from forecasting_tools.agents_and_tools.misc_tools import (
    create_tool_for_forecasting_bot,
    get_general_news_with_asknews,
    grab_open_questions_from_tournament,
    grab_question_details_from_metaculus,
    perplexity_pro_search,
    perplexity_quick_search,
    smart_searcher_search,
)
from forecasting_tools.agents_and_tools.question_generators.info_hazard_identifier import (
    InfoHazardIdentifier,
)
from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)
from forecasting_tools.agents_and_tools.question_generators.topic_generator import (
    TopicGenerator,
)
from forecasting_tools.ai_models.agent_wrappers import AgentSdkLlm, AgentTool
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_bots.bot_lists import (
    get_all_important_bot_classes,
)
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


DEFAULT_MODEL: str = (
    "perplexity/sonar-pro"  # Changed from openrouter/google/gemini-2.5-pro-preview
)
DEFAULT_MESSAGE: dict = {
    "role": "assistant",
    "content": "How may I assist you today?",
}

class ChatSession(BaseModel, Jsonable):
    name: str
    messages: list[dict]
    model_choice: str = DEFAULT_MODEL
    trace_id: str | None = None
    last_chat_cost: float | None = None
    last_chat_duration: float | None = None


def display_messages(messages: list[dict]) -> None:
    assistant_message_num = 0
    st.sidebar.write("**Tool Calls and Outputs:**")
    for message in messages:
        output_emoji = "🔍"
        call_emoji = "📞"
        if "type" in message and message["type"] == "function_call":
            call_id = message["name"]
            with st.sidebar.expander(f"{call_emoji} Call: {call_id}"):
                st.write(f"Function: {message['name']}")
                st.write(f"Arguments: {message['arguments']}")
                st.write(f"Call ID: {message['call_id']}")
                st.write(
                    f"Assistant Message Number: {assistant_message_num}"
                )
                continue
        if "type" in message and message["type"] == "function_call_output":
            call_id = message["call_id"]
            with st.sidebar.expander(f"{output_emoji} Output: {call_id}"):
                st.write(f"Call ID: {message['call_id']}")
                st.write(
                    f"Assistant Message Number: {assistant_message_num}"
                )
                st.write(f"Output:\n\n{message['output']}")
                continue

        try:
            role = message["role"]
        except KeyError:
            if "type" in message and message["type"] == "reasoning":
                logger.warning(f"Found message with no role: {message}")
            else:
                st.error(f"Unexpected message role. Message: {message}")
            continue

        with st.chat_message(role):
            if role == "assistant":
                assistant_message_num += 1
            content = message["content"]
            if isinstance(content, list):
                text = content[0]["text"]
            else:
                text = content
            st.write(ReportDisplayer.clean_markdown(text))


def display_debug_mode() -> None:
    local_streamlit_mode = (
        os.getenv("LOCAL_STREAMLIT_MODE", "false").lower() == "true"
    )
    if local_streamlit_mode:
        if st.sidebar.checkbox("Debug Mode", value=True):
            st.session_state["debug_mode"] = True
        else:
            st.session_state["debug_mode"] = False


def display_model_selector() -> None:
    if "model_choice" not in st.session_state.keys():
        st.session_state["model_choice"] = DEFAULT_MODEL
    model_name: str = st.session_state["model_choice"]
    model_choice = st.sidebar.text_input(
        "Litellm compatible model used for chat (not tools)",
        value=model_name,
    )
    if "o1-pro" in model_choice or "gpt-4.5" in model_choice:
        raise ValueError(
            "o1 pro and gpt-4.5 are not available for this application."
        )
    st.session_state["model_choice"] = model_choice


def get_chat_tools() -> list[Tool]:
    return [
        TopicGenerator.find_random_headlines_tool,
        QuestionDecomposer.decompose_into_questions_tool,
        QuestionOperationalizer.question_operationalizer_tool,
        perplexity_pro_search,
        get_general_news_with_asknews,
        smart_searcher_search,
        grab_question_details_from_metaculus,
        grab_open_questions_from_tournament,
        TopicGenerator.get_headlines_on_random_company_tool,
        perplexity_quick_search,
        InfoHazardIdentifier.info_hazard_identifier_tool,
    ]


def display_tools() -> list[Tool]:
    default_tools: list[Tool] = get_chat_tools()
    bot_options = get_all_important_bot_classes()

    active_tools: list[Tool] = []
    with st.sidebar.expander("Select Tools"):
        bot_choice = st.selectbox(
            "Select a bot for forecast_question_tool (Main Bot is best)",
            [bot.__name__ for bot in bot_options],
        )
        bot = next(
            bot for bot in bot_options if bot.__name__ == bot_choice
        )
        default_tools = [
            create_tool_for_forecasting_bot(bot)
        ] + default_tools

        tool_names = [tool.name for tool in default_tools]
        all_checked = all(
            st.session_state.get(f"tool_{name}", True)
            for name in tool_names
        )
        toggle_label = "Toggle all Tools"
        if st.button(toggle_label):
            for name in tool_names:
                st.session_state[f"tool_{name}"] = not all_checked
        for tool in default_tools:
            key = f"tool_{tool.name}"
            if key not in st.session_state:
                st.session_state[key] = True

            tool_active = st.checkbox(tool.name, key=key)

            if tool_active:
                active_tools.append(tool)

    with st.sidebar.expander("Tool Explanations"):
        for tool in active_tools:
            if isinstance(tool, AgentTool):
                property_description = ""
                for property_name, metadata in tool.params_json_schema[
                    "properties"
                ].items():
                    description = metadata.get(
                        "description", "No description provided"
                    )
                    field_type = metadata.get("type", "No type provided")
                    property_description += f"- {property_name}: {description} (type: {field_type})\n"
                st.write(
                    clean_indents(
                        f"""
                        **{tool.name}**

                        {tool.description}

                        **Parameters**
                        {property_description}
                        """
                    )
                )
    return active_tools


def display_chat_metadata() -> None:
    if "last_chat_cost" in st.session_state:
        st.sidebar.write(
            f"Cost of Last Chat: ${st.session_state.last_chat_cost:.3f}"
        )
    if "last_chat_duration" in st.session_state:
        st.sidebar.write(
            f"Duration of Last Chat: {st.session_state.last_chat_duration:.2f}s"
        )
    if "trace_id" in st.session_state and st.session_state["trace_id"]:
        st.sidebar.write(
            f"Trace ID: [{st.session_state['trace_id']}](https://smith.langchain.com/public/{st.session_state['trace_id']}/log)"
        )


def display_premade_examples() -> None:
    st.sidebar.write("---")
    st.sidebar.write("**Example Prompts**")

    example_prompts = [
        "What are some interesting forecasting questions I could look into?",
        "Decompose the question: 'Will the Liberal Party of Canada win the most seats in the next Canadian federal election?'",
        "Operationalize the question: 'Will the Liberal Party of Canada win the most seats in the next Canadian federal election?'",
        "Who is likely to win the next US presidential election?",
        "Forecast the question: 'Before 2027, will any national government, or collection of national governments acting in concert, mandate the installation of a 'kill switch' in all new consumer AI hardware (including GPUs and other specialised AI chips)?'",
    ]

    for prompt in example_prompts:
        if st.sidebar.button(prompt):
            st.session_state.messages.append(
                {"role": "user", "content": prompt}
            )
            st.rerun()

async def generate_response(
    prompt_input: str | None,
    active_tools: list[Tool],
) -> None:

    if not prompt_input:
        return

    _update_last_message_if_gemini_bug(st.session_state["model_choice"])

    agent = Agent(
        AgentSdkLlm(
            model=st.session_state["model_choice"],
            temperature=0,
            timeout=180,
            num_retries=1,
        ),
        tools=active_tools,
    )
    runner = Runner(agent)

    message_placeholder = st.empty()
    full_response = ""

    with st.chat_message("assistant"):
        with trace("chatbot-test") as trace_url:
            st.session_state.trace_id = trace_url.id
            final_answer_message = ""
            async for item in runner.run(st.session_state.messages):
                message_type = item.type
                logger.info(f"Message type: {message_type}")
                st.session_state.messages.append(item.data)
                text = _grab_text_of_item(item)
                if text is not None:
                    final_answer_message = text
                    if isinstance(item.data["content"], list):
                        content_data = item.data["content"][0]
                        if isinstance(content_data, ResponseTextDeltaEvent):
                            full_response += content_data.delta
                        else:
                            full_response += text
                    else:
                        full_response += text
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_answer_message}
            )


def _update_last_message_if_gemini_bug(model_choice: str) -> None:
    if "gemini" in model_choice:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            if isinstance(last_message["content"], str):
                last_message["content"] = [
                    {"type": "text", "text": last_message["content"]}
                ]
            else:
                logger.error(
                    "Last message content is not a string, but a list"
                )


def _grab_text_of_item(item: RunItem) -> str:
    message = item.data
    text = None
    if message["role"] == "assistant":
        if isinstance(message["content"], list):
            content_data = message["content"][0]
            if isinstance(content_data, ResponseTextDeltaEvent):
                text = content_data.delta
            else:
                text = content_data["text"]
        else:
            text = message["content"]
    return text


def clear_chat_history() -> None:
    st.session_state.messages = [DEFAULT_MESSAGE]
    st.session_state.trace_id = None
    st.session_state.last_chat_cost = None
    st.session_state.last_chat_duration = None

async def main():
    st.title("💬 Chatbot")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [DEFAULT_MESSAGE]
    display_debug_mode()
    st.sidebar.button(
        "Clear Chat History", on_click=clear_chat_history
    )
    display_model_selector()
    active_tools = display_tools()
    display_chat_metadata()
    display_premade_examples()
    st.sidebar.write("---")
    display_messages(st.session_state.messages)

    if prompt := st.chat_input():
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with MonetaryCostManager(10) as cost_manager:
            start_time = time.time()
            await generate_response(prompt, active_tools)
            st.session_state.last_chat_cost = cost_manager.current_usage
            end_time = time.time()
            st.session_state.last_chat_duration = end_time - start_time
        st.rerun()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())