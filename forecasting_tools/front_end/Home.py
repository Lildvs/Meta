import os
import sys

import dotenv
import streamlit as st

from forecasting_tools.front_end.app_pages.benchmark_page import BenchmarkPage
from forecasting_tools.front_end.app_pages.chat_page import ChatPage

current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(top_level_dir)


from forecasting_tools.front_end.app_pages.base_rate_page import BaseRatePage
from forecasting_tools.front_end.app_pages.estimator_page import EstimatorPage
from forecasting_tools.front_end.app_pages.forecaster_page import (
    ForecasterPage,
)
from forecasting_tools.front_end.app_pages.key_factors_page import (
    KeyFactorsPage,
)
from forecasting_tools.front_end.app_pages.niche_list_researcher_page import (
    NicheListResearchPage,
)
from forecasting_tools.front_end.helpers.app_page import AppPage
from forecasting_tools.util.custom_logger import CustomLogger


class HomePage(AppPage):
    PAGE_DISPLAY_NAME: str = "🏠 Home"
    URL_PATH: str = "/"
    IS_DEFAULT_PAGE: bool = True

    CHAT_PAGE: type[AppPage] = ChatPage
    FORECASTER_PAGE: type[AppPage] = ForecasterPage
    BASE_RATE_PAGE: type[AppPage] = BaseRatePage
    NICHE_LIST_RESEARCH_PAGE: type[AppPage] = NicheListResearchPage
    ESTIMATOR_PAGE: type[AppPage] = EstimatorPage
    KEY_FACTORS_PAGE: type[AppPage] = KeyFactorsPage
    BENCHMARK_PAGE: type[AppPage] = BenchmarkPage
    NON_HOME_PAGES: list[type[AppPage]] = [
        CHAT_PAGE,
        FORECASTER_PAGE,
        KEY_FACTORS_PAGE,
        BASE_RATE_PAGE,
        NICHE_LIST_RESEARCH_PAGE,
        ESTIMATOR_PAGE,
    ]

    @classmethod
    async def _async_main(cls) -> None:
        st.title("What do you want to do?")
        for page in cls.NON_HOME_PAGES:
            label = page.PAGE_DISPLAY_NAME
            if st.button(label, key=label):
                st.switch_page(page.convert_to_streamlit_page())


def run_forecasting_streamlit_app() -> None:
    st.set_page_config(
        page_title="Forecasting-Tools",
        page_icon=":material/explore:",
        layout="wide",
    )
    st.title("🏠 Home")
    st.write("Select a tool from the sidebar to get started.")

    st.markdown("---")
    st.write(
        "This is the demo site for the "
        "[forecasting-tools python package](https://github.com/CodexVeritas/forecasting-tools)."
    )
    st.write(
        "Give feedback on the [Forecasting Tools Discord](https://discord.gg/Dtq4JNdXnw) or email "
        "me at ben [at] metaculus [dot com]. "
        "Let me know what I can do to make this a tool you will want to use "
        "every day! Let me know if you want to chat and we can find a time!"
    )
    st.write(
        "Queries made to the website are saved to a database and may be "
        "reviewed to help improve the tool"
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    if "logger_initialized" not in st.session_state:
        CustomLogger.clear_latest_log_files()
        CustomLogger.setup_logging()
        st.session_state["logger_initialized"] = True
    run_forecasting_streamlit_app()
