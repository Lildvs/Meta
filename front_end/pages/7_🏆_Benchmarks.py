import logging

from forecasting_tools.benchmarking.benchmark_displayer import (
    run_benchmark_streamlit_page,
)
import streamlit as st

logger = logging.getLogger(__name__)


class ChatMessage:
    def __init__(self, role: str, content: str, reasoning: str = "") -> None:
        self.role = role
        self.content = content
        self.reasoning = reasoning

    def to_open_ai_message(self) -> dict:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            reasoning=data.get("reasoning", ""),
        )

st.title("🏆 Benchmarks")
run_benchmark_streamlit_page("logs/forecasts/benchmarks/")