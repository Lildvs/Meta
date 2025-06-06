from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Coroutine

import streamlit as st
from pydantic import BaseModel

from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class Example(BaseModel, Jsonable):
    short_name: str | None = None
    notes: str | None = None
    input: dict[str, Any]
    output: dict[str, Any]


class ToolPage:
    def __init__(
        self,
        get_input_func: Callable[[], Coroutine[Any, Any, Jsonable | None]],
        run_tool_func: Callable[[Jsonable], Coroutine[Any, Any, Jsonable]],
        display_outputs_func: Callable[[list[Jsonable]], Coroutine[Any, Any, None]],
        save_run_to_coda_func: Callable[
            [Jsonable, Jsonable, bool], Coroutine[Any, Any, None]
        ],
        input_type: type[Jsonable],
        output_type: type[Jsonable],
        examples_file_path: str,
        intro_text: str | None = None,
    ):
        self.get_input_func = get_input_func
        self.run_tool_func = run_tool_func
        self.display_outputs_func = display_outputs_func
        self.save_run_to_coda_func = save_run_to_coda_func
        self.input_type = input_type
        self.output_type = output_type
        self.examples_file_path = examples_file_path
        self.intro_text = intro_text

    async def _async_main(self) -> None:
        if self.intro_text:
            st.markdown(self.intro_text)

        await self._display_example_buttons_expander()
        input_to_tool = await self.get_input_func()
        if input_to_tool:
            assert isinstance(input_to_tool, self.input_type)
            output = await self.run_tool_func(input_to_tool)
            assert isinstance(output, self.output_type)
            await self._save_run(input_to_tool, output)
        outputs = await self._get_saved_outputs()
        if outputs:
            await self.display_outputs_func(outputs)

    async def _display_example_buttons_expander(self) -> None:
        examples = await self._get_examples()
        if examples:
            with st.expander("📋 Premade Examples", expanded=False):
                cols = st.columns(len(examples))
                for index, example in enumerate(examples):
                    with cols[index]:
                        await self._display_single_example_button(example, index)

    async def _display_single_example_button(
        self, example: Example, example_number: int
    ) -> None:
        button_label = f"Show Example {example_number + 1}"
        if example.short_name:
            button_label += f": {example.short_name}"
        example_clicked = st.button(button_label, use_container_width=True)
        if example.notes:
            st.markdown(
                f"<div style='text-align: center'>{example.notes}</div>",
                unsafe_allow_html=True,
            )
        if example_clicked:
            input_to_tool = self.input_type.from_json(example.input)
            output = self.output_type.from_json(example.output)
            await self._save_run(input_to_tool, output, is_premade_example=True)

    async def _get_examples(self) -> list[Example]:
        if self.examples_file_path is None:
            return []
        examples = Example.load_json_from_file_path(self.examples_file_path)
        return examples

    async def _save_run(
        self,
        input_to_tool: Jsonable,
        output: Jsonable,
        is_premade_example: bool = False,
    ) -> None:
        assert isinstance(output, self.output_type)
        await self._save_output_to_session_state(output)

        if not is_premade_example:
            try:
                await self._save_run_to_file(input_to_tool, output)
            except Exception as e:
                logger.error(f"Error saving output to file: {e}")

        assert isinstance(input_to_tool, BaseModel)
        assert isinstance(output, BaseModel)
        input_to_tool_copy = input_to_tool.model_copy(deep=True)
        output_copy = output.model_copy(deep=True)

        try:
            await self.save_run_to_coda_func(
                input_to_tool_copy, output_copy, is_premade_example
            )
        except Exception as e:
            logger.error(f"Error saving output to Coda: {e}")

    async def _save_output_to_session_state(self, output: Jsonable) -> None:
        assert isinstance(output, self.output_type)
        session_state_key = self.__get_saved_outputs_key()
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = []
        st.session_state[session_state_key].insert(0, output)

    async def _save_run_to_file(
        self, input_to_tool: Jsonable, output: Jsonable
    ) -> None:
        assert isinstance(output, self.output_type)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"logs/forecasts/streamlit/{timestamp}_{self.__class__.__name__}.json"
        Example.save_object_list_to_file_path(
            [Example(input=input_to_tool.to_json(), output=output.to_json())],
            file_path,
        )

    async def _get_saved_outputs(self) -> list[Jsonable]:
        session_state_key = self.__get_saved_outputs_key()
        if session_state_key not in st.session_state:
            st.session_state[session_state_key] = []
        outputs = st.session_state[session_state_key]
        validated_outputs = [
            self.output_type.from_json(output.to_json()) for output in outputs
        ]
        return validated_outputs

    def __get_saved_outputs_key(self) -> str:
        return f"{self.__class__.__name__}_saved_outputs"
