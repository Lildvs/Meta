from __future__ import annotations

import logging

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.agents_and_tools.base_rates.estimator import Estimator
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.forecast_helpers.forecast_database_manager import (
    ForecastDatabaseManager,
    ForecastRunType,
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class EstimatorInput(Jsonable, BaseModel):
    estimate_type: str
    previous_research: str | None = None


class EstimatorOutput(Jsonable, BaseModel):
    estimate_type: str
    previous_research: str | None
    number: float
    markdown: str
    cost: float


async def _get_input() -> EstimatorInput | None:
    with st.form("estimator_form"):
        estimate_type = st.text_input("What do you want to estimate?")
        submitted = st.form_submit_button("Generate Estimate")
        if submitted and estimate_type:
            return EstimatorInput(estimate_type=estimate_type)
    return None

async def _run_tool(input: EstimatorInput) -> EstimatorOutput:
    with MonetaryCostManager() as cost_manager:
        estimator = Estimator(input.estimate_type, input.previous_research)
        number, markdown = await estimator.estimate_size()
        cost = cost_manager.current_usage
        return EstimatorOutput(
            estimate_type=input.estimate_type,
            previous_research=input.previous_research,
            number=number,
            markdown=markdown,
            cost=cost,
        )

async def _save_run_to_coda(
    input_to_tool: EstimatorInput,
    output: EstimatorOutput,
    is_premade: bool,
) -> None:
    if is_premade:
        output.cost = 0
    ForecastDatabaseManager.add_general_report_to_database(
        question_text=output.estimate_type,
        background_info=output.previous_research,
        resolution_criteria=None,
        fine_print=None,
        prediction=output.number,
        explanation=output.markdown,
        page_url=None,
        price_estimate=output.cost,
        run_type=ForecastRunType.WEB_APP_ESTIMATOR,
    )

async def _display_outputs(outputs: list[EstimatorOutput]) -> None:
    for output in outputs:
        with st.expander(
            f"Estimate for {output.estimate_type}: {int(output.number):,}"
        ):
            st.markdown(f"Cost: ${output.cost:.2f}")
            st.markdown(output.markdown)

async def main():
    st.title("🧮 Fermi Estimator")

    page = ToolPage(
        get_input_func=_get_input,
        run_tool_func=_run_tool,
        display_outputs_func=_display_outputs,
        save_run_to_coda_func=_save_run_to_coda,
        input_type=EstimatorInput,
        output_type=EstimatorOutput,
        examples_file_path="forecasting_tools/front_end/example_outputs/estimator_page_examples.json"
    )
    await page._async_main()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())