import os
import sys

import dotenv
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure the forecasting_tools package is importable when running via `streamlit run front_end/Home.py`.
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)

# We prefer to reuse the richer Home page inside the Python package if it is fully
# available. However, in lightweight deployments the sub-module
# `forecasting_tools.front_end.app_pages` (and its dependencies) might be absent.
# Fall back to a minimal Home implementation in that case so the app still boots.

try:
    from forecasting_tools.front_end.Home import (
        run_forecasting_streamlit_app as _pkg_home_fn,
    )

    _USE_PKG_HOME = True
except ModuleNotFoundError:
    _USE_PKG_HOME = False


def _fallback_run_home() -> None:  # noqa: D401 – simple page helper
    """Minimal Home page shown when full package Home cannot be imported."""

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


# ------------------------------
# Entry-point
# ------------------------------


if __name__ == "__main__":
    dotenv.load_dotenv()

    from forecasting_tools.util.custom_logger import CustomLogger

    if "logger_initialized" not in st.session_state:
        CustomLogger.clear_latest_log_files()
        CustomLogger.setup_logging()
        st.session_state["logger_initialized"] = True

    if _USE_PKG_HOME:
        _pkg_home_fn()
    else:
        _fallback_run_home()
