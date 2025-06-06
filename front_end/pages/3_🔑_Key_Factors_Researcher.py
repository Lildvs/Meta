from __future__ import annotations

import logging
import os
import re
import sys

import dotenv
import streamlit as st
from pydantic import BaseModel

dotenv.load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(top_level_dir)

from forecasting_tools.agents_and_tools.key_factors_researcher import (
    KeyFactorsResearcher,
    ScoredKeyFactor,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
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

logger = logging.getLogger(__name__)


class KeyFactorsInput(Jsonable, BaseModel):
    metaculus_url: str


class KeyFactorsOutput(Jsonable, BaseModel):
    question_text: str
    markdown: str
    cost: float
    scored_key_factors: list[ScoredKeyFactor] | None = None


async def _get_input() -> KeyFactorsInput | None:
    with st.form("key_factors_form"):
        metaculus_url = st.text_input("Metaculus Question URL")
        submitted = st.form_submit_button("Find Key Factors")
        if submitted and metaculus_url:
            return KeyFactorsInput(metaculus_url=metaculus_url)
    return None


async def _run_tool(input: KeyFactorsInput) -> KeyFactorsOutput:
    with st.spinner(
        "Finding key factors... This may take a minute or two..."
    ):
        question_id = __extract_question_id(input.metaculus_url)
        metaculus_question = MetaculusApi.get_question_by_post_id(
            question_id
        )

        with MonetaryCostManager() as cost_manager:
            num_questions_to_research = 16
            num_key_factors_to_return = 7
            key_factors = await KeyFactorsResearcher.find_and_sort_key_factors(
                metaculus_question,
                num_questions_to_research_with=num_questions_to_research,
                num_key_factors_to_return=num_key_factors_to_return,
            )
            cost = cost_manager.current_usage
            markdown = ScoredKeyFactor.turn_key_factors_into_markdown_list(
                key_factors
            )
            return KeyFactorsOutput(
                question_text=metaculus_question.question_text,
                markdown=markdown,
                cost=cost,
                scored_key_factors=key_factors,
            )


async def _save_run_to_coda(
    input_to_tool: KeyFactorsInput,
    output: KeyFactorsOutput,
    is_premade: bool,
) -> None:
    if is_premade:
        output.cost = 0
    ForecastDatabaseManager.add_general_report_to_database(
        question_text=output.question_text,
        background_info=None,
        resolution_criteria=None,
        fine_print=None,
        prediction=None,
        explanation=output.markdown,
        page_url=None,
        price_estimate=output.cost,
        run_type=ForecastRunType.WEB_APP_KEY_FACTORS,
    )


async def _display_outputs(outputs: list[KeyFactorsOutput]) -> None:
    for output in outputs:
        with st.expander(f"Key Factors for: {output.question_text}"):
            st.markdown(f"Cost: ${output.cost:.2f}")
            st.markdown(ReportDisplayer.clean_markdown(output.markdown))


def __extract_question_id(url: str) -> int:
    match = re.search(r"/questions/(\d+)/", url)
    if match:
        return int(match.group(1))
    raise ValueError(
        "Invalid Metaculus question URL. Please ensure it's in the format: https://metaculus.com/questions/[ID]/[question-title]/"
    )

async def main():
    st.title("🔑 Key Factors Researcher")

    page = ToolPage(
        get_input_func=_get_input,
        run_tool_func=_run_tool,
        display_outputs_func=_display_outputs,
        save_run_to_coda_func=_save_run_to_coda,
        input_type=KeyFactorsInput,
        output_type=KeyFactorsOutput,
        examples_file_path="forecasting_tools/front_end/example_outputs/key_factors_page_examples.json"
    )
    await page._async_main()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())