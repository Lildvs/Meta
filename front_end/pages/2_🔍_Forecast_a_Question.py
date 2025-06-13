import logging
import re

import dotenv
import streamlit as st
from pydantic import BaseModel

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.main_bot import MainBot
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable
from forecasting_tools.forecast_helpers.research_orchestrator import orchestrate_research

logger = logging.getLogger(__name__)


class ForecastInput(Jsonable, BaseModel):
    question: BinaryQuestion
    jarvis_mode: bool = False


async def _get_input() -> ForecastInput | None:
    # __display_metaculus_url_input()  # Disabled: hide Metaculus URL input
    with st.form("forecast_form"):
        question_text = st.text_input(
            "Yes/No Binary Question",
            key="question_text_box",
            help="💡 Tip: Start your question with 'JARVIS' to force deep research using Perplexity Pro (bypasses the research critic)"
        )
        resolution_criteria = st.text_area(
            "Resolution Criteria (optional)",
            key="resolution_criteria_box",
        )
        fine_print = st.text_area(
            "Fine Print (optional)", key="fine_print_box"
        )
        background_info = st.text_area(
            "Background Info (optional)", key="background_info_box"
        )

        submitted = st.form_submit_button("Submit")

        if submitted:
            if not question_text:
                st.error("Question Text is required.")
                return None

            # Check for JARVIS mode trigger
            jarvis_mode = question_text.strip().lower().startswith("jarvis")
            if jarvis_mode:
                # Remove the JARVIS trigger from the question text
                question_text = question_text.strip()[len("jarvis"):].strip()

            question = BinaryQuestion(
                question_text=question_text,
                background_info=background_info,
                resolution_criteria=resolution_criteria,
                fine_print=fine_print,
                page_url="",
                api_json={},
            )

            return ForecastInput(
                question=question,
                jarvis_mode=jarvis_mode,
            )
    return None


# ------------------------------------------------------------
# Custom bot that always runs *deep* research (Perplexity + others)
# ------------------------------------------------------------


class DeepResearchBot(MainBot):
    def __init__(self, jarvis_mode: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.jarvis_mode = jarvis_mode

    async def run_research(self, question):  # noqa: D401
        if self.jarvis_mode:
            # JARVIS mode: Force deep research without ToolCritic
            try:
                snippets = await self._orchestrate_research_force_deep(question.question_text)
                research_text = "\n".join(f"* {s['text']}" for s in snippets)
                return research_text or "No research found."
            except Exception as e:
                logger.error(f"Error running JARVIS deep research: {e}")
                return "Research unavailable due to technical issues."
        else:
            # Normal mode: Use standard orchestrate_research with ToolCritic
            try:
                snippets = await orchestrate_research(question.question_text, depth="deep")
                research_text = "\n".join(f"* {s['text']}" for s in snippets)
                return research_text or "No research found."
            except Exception as e:
                logger.error(f"Error running deep research: {e}")
                return "Research unavailable due to technical issues."

    async def _orchestrate_research_force_deep(self, query: str) -> list[dict[str, str]]:
        """Modified orchestrate_research that bypasses ToolCritic for JARVIS mode."""
        # Import the orchestrate_research function and modify its behavior
        from forecasting_tools.forecast_helpers.research_orchestrator import _dedupe
        import asyncio
        import os
        from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
        from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
        from forecasting_tools.agents_and_tools.misc_tools import perplexity_pro_search

        snippets: list[dict[str, str]] = []

        # Run SmartSearcher
        smart_searcher = SmartSearcher(num_searches_to_run=1, num_sites_per_search=5)
        smart_future = smart_searcher.invoke(query)

        # Run AskNews if available
        asknews_enabled = os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")
        ask_task = (
            AskNewsSearcher().get_formatted_news_async(query) if asknews_enabled else None
        )

        # JARVIS mode: Always run Perplexity deep search (bypass ToolCritic)
        # Call it as a coroutine since it's an async function
        deep_task = perplexity_pro_search(query)

        # Gather all tasks
        tasks = [smart_future]
        if ask_task:
            tasks.append(ask_task)
        tasks.append(deep_task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning(f"Research task {i} failed: {res}")
                continue
            if isinstance(res, list):
                # This is likely from perplexity_pro_search which returns list[dict]
                snippets.extend(res)
            else:
                # This is a string result, wrap it
                src_name = ["smart_search", "asknews", "perplexity"][i]
                snippets.append({"source": src_name, "text": str(res)})

        # Use the original dedupe function
        return _dedupe(snippets)


async def _run_tool(input: ForecastInput) -> BinaryReport:
    with st.spinner("Forecasting... This may take a minute or two..."):
        report = await DeepResearchBot(
            research_reports_per_question=1,
            predictions_per_research_report=5,
            publish_reports_to_metaculus=False,
            folder_to_save_reports_to=None,
            jarvis_mode=input.jarvis_mode,
        ).forecast_question(input.question)
        assert isinstance(report, BinaryReport)
        return report


async def _save_run_to_coda(
    input_to_tool: ForecastInput,
    output: BinaryReport,
    is_premade: bool,
) -> None:
    if is_premade:
        output.price_estimate = 0
    ForecastDatabaseManager.add_forecast_report_to_database(
        output, run_type=ForecastRunType.WEB_APP_FORECAST
    )


async def _display_outputs(outputs: list[BinaryReport]) -> None:
    ReportDisplayer.display_report_list(outputs)


def __display_metaculus_url_input() -> None:
    with st.expander("Use an existing Metaculus Binary question"):
        st.write(
            "Enter a Metaculus question URL to autofill the form below."
        )

        metaculus_url = st.text_input(
            "Metaculus Question URL", key="metaculus_url_input"
        )
        fetch_button = st.button("Fetch Question", key="fetch_button")

        if fetch_button and metaculus_url:
            with st.spinner("Fetching question details..."):
                try:
                    question_id = __extract_question_id(metaculus_url)
                    metaculus_question = (
                        MetaculusApi.get_question_by_post_id(question_id)
                    )
                    if isinstance(metaculus_question, BinaryQuestion):
                        __autofill_form(metaculus_question)
                    else:
                        st.error(
                            "Only binary questions are supported at this time."
                        )
                except Exception as e:
                    st.error(
                        f"An error occurred while fetching the question: {e.__class__.__name__}: {e}"
                    )


def __extract_question_id(url: str) -> int:
    match = re.search(r"/questions/(\d+)/", url)
    if match:
        return int(match.group(1))
    raise ValueError(
        "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
    )


def __autofill_form(question: BinaryQuestion) -> None:
    st.session_state["question_text_box"] = question.question_text
    st.session_state["background_info_box"] = (
        question.background_info or ""
    )
    st.session_state["resolution_criteria_box"] = (
        question.resolution_criteria or ""
    )
    st.session_state["fine_print_box"] = question.fine_print or ""


async def main():
    st.title("🔍 Forecast a Question")

    page = ToolPage(
        get_input_func=_get_input,
        run_tool_func=_run_tool,
        display_outputs_func=_display_outputs,
        save_run_to_coda_func=_save_run_to_coda,
        input_type=ForecastInput,
        output_type=BinaryReport,
        examples_file_path="forecasting_tools/front_end/example_outputs/forecast_page_examples.json",
    )
    await page._async_main()


if __name__ == "__main__":
    import asyncio
    dotenv.load_dotenv()
    asyncio.run(main())