import logging

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.agents_and_tools.base_rates.niche_list_researcher import (
    FactCheckedItem,
    NicheListResearcher,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.front_end.helpers.report_displayer import (
    ReportDisplayer,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class NicheListOutput(Jsonable, BaseModel):
    question_text: str
    niche_list_items: list[FactCheckedItem]
    cost: float

    @property
    def markdown_output(self) -> str:
        return FactCheckedItem.make_markdown_with_valid_and_invalid_lists(
            self.niche_list_items
        )


class NicheListInput(Jsonable, BaseModel):
    question_text: str


async def _get_input() -> NicheListInput | None:
    with st.form("niche_list_form"):
        question_text = st.text_input(
            "Enter your niche list research query here"
        )
        submitted = st.form_submit_button("Research and Generate List")
        if submitted and question_text:
            return NicheListInput(question_text=question_text)
    return None

async def _run_tool(input: NicheListInput) -> NicheListOutput:
    with st.spinner(
        "Researching and fact-checking... This may take several minutes..."
    ):
        with MonetaryCostManager() as cost_manager:
            generator = NicheListResearcher(input.question_text)
            fact_checked_items = (
                await generator.research_niche_reference_class(
                    return_invalid_items=True
                )
            )

            cost = cost_manager.current_usage

            return NicheListOutput(
                question_text=input.question_text,
                cost=cost,
                niche_list_items=fact_checked_items,
            )

async def _save_run_to_coda(
    input_to_tool: NicheListInput,
    output: NicheListOutput,
    is_premade: bool,
) -> None:
    if is_premade:
        output.cost = 0
    ForecastDatabaseManager.add_general_report_to_database(
        question_text=input_to_tool.question_text,
        background_info=None,
        resolution_criteria=None,
        fine_print=None,
        prediction=len(output.niche_list_items),
        explanation=output.markdown_output,
        page_url=None,
        price_estimate=output.cost,
        run_type=ForecastRunType.WEB_APP_NICHE_LIST,
    )

async def _display_outputs(outputs: list[NicheListOutput]) -> None:
    for output in outputs:
        with st.expander(f"{output.question_text}"):
            st.markdown(f"**Cost:** ${output.cost:.2f}")
            st.markdown(
                ReportDisplayer.clean_markdown(output.markdown_output)
            )

async def main():
    st.title("📋 Niche List Researcher")

    page = ToolPage(
        get_input_func=_get_input,
        run_tool_func=_run_tool,
        display_outputs_func=_display_outputs,
        save_run_to_coda_func=_save_run_to_coda,
        input_type=NicheListInput,
        output_type=NicheListOutput,
        examples_file_path="forecasting_tools/front_end/example_outputs/niche_list_page_examples.json"
    )
    await page._async_main()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())