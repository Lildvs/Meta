import os
import sys

import dotenv
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
# Note: This path manipulation is to ensure the forecasting_tools package is found.
# In a real deployment, the package would be installed, and this would not be necessary.
top_level_dir = os.path.abspath(os.path.join(current_dir, "../"))
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)

from forecasting_tools.front_end.Home import run_forecasting_streamlit_app
from forecasting_tools.util.custom_logger import CustomLogger

if __name__ == "__main__":
    dotenv.load_dotenv()
    if "logger_initialized" not in st.session_state:
        CustomLogger.clear_latest_log_files()
        CustomLogger.setup_logging()
        st.session_state["logger_initialized"] = True

    run_forecasting_streamlit_app()
