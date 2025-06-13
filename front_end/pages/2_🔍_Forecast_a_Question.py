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


async def _get_input() -> ForecastInput | None:
    # __display_metaculus_url_input()  # Disabled: hide Metaculus URL input
    with st.form("forecast_form"):
        question_text = st.text_input(
            "Yes/No Binary Question", key="question_text_box"
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
            )
    return None


# ------------------------------------------------------------
# Custom bot that always runs *deep* research (Perplexity + others)
# ------------------------------------------------------------


class DeepResearchBot(MainBot):
    async def run_research(self, question):  # noqa: D401
        snippets = await orchestrate_research(question.question_text, depth="deep")
        research_text = "\n".join(f"* {s['text']}" for s in snippets)
        return research_text or "No research found."


async def _run_tool(input: ForecastInput) -> BinaryReport:
    with st.spinner("Forecasting... This may take a minute or two..."):
        report = await DeepResearchBot(
            research_reports_per_question=1,
            predictions_per_research_report=5,
            publish_reports_to_metaculus=False,
            folder_to_save_reports_to=None,
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