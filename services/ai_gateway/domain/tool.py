from typing import Any, Callable, Dict, Literal, Optional

from langchain.tools import tool as lc_tool  # pyright: ignore
from langchain_core.tools import ArgsSchema, BaseTool


class Tool:
    def __init__(
        self,
        description: str,
        target: Callable,
        return_direct: bool = False,
        args_schema: Optional[ArgsSchema] = None,
        infer_schema: Optional[bool] = True,
        response_format: Optional[Literal["content", "content_and_artifact"]] = "content",
        parse_docstring: Optional[bool] = False,
        error_on_invalid_docstring: Optional[bool] = True,
    ) -> None:
        self.description = description
        self.target = target  # pyright: ignore
        self.return_direct = return_direct
        self.args_schema = args_schema
        self.infer_schema = infer_schema
        self.response_format = response_format
        self.parse_docstring = parse_docstring
        self.error_on_invalid_docstring = error_on_invalid_docstring

    def create_tool(
        self,
    ) -> BaseTool:
        tool_kwargs: Dict[str, Any] = {
            "name_or_callable": self.target,  # pyright: ignore
            "description": self.description,
            "return_direct": self.return_direct,
            "args_schema": self.args_schema,
            "infer_schema": self.infer_schema,
            "response_format": self.response_format,
            "parse_docstring": self.parse_docstring,
            "error_on_invalid_docstring": self.error_on_invalid_docstring,
        }

        return lc_tool(**tool_kwargs)
