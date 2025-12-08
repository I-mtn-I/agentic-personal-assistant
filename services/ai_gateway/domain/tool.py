from typing import Any, Callable, Dict, Literal, Optional

from langchain.tools import tool as lc_tool  # pyright: ignore
from langchain_core.tools import ArgsSchema, BaseTool


class Tool:
    """
    Represents a tool that can be used by an AI agent. Initializes with various configurations
    and provides methods to create the tool instance.

    :param description: A description of what the tool does.
    :type description: str
    :param target: The callable function or method that the tool executes.
    :type target: Callable
    :param return_direct: Whether the response should be returned directly.
    :type return_direct: bool
    :param args_schema: Schema for the arguments of the tool.
    :type args_schema: Optional[ArgsSchema]
    :param infer_schema: Whether to automatically infer the schema from the target function.
    :type infer_schema: Optional[bool]
    :param response_format: The format of the response, either "content" or "content_and_artifact".
    :type response_format: Optional[Literal["content", "content_and_artifact"]]
    :param parse_docstring: Whether to parse the docstring for schema information.
    :type parse_docstring: Optional[bool]
    :param error_on_invalid_docstring: Whether to raise an error if the docstring is invalid
    :type error_on_invalid_docstring: Optional[bool]
    """

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
        """
        Builds a LangChain ``BaseTool`` instance.
        Used to create built-in tools.
        Aimed to be consumed by ```built-in-factory``` only.
        """
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


def create_tool_from_callable(
    target: Callable,
    description: str,
    *,
    return_direct: bool = False,
    args_schema: Optional[ArgsSchema] = None,
    infer_schema: Optional[bool] = True,
    response_format: Optional[Literal["content", "content_and_artifact"]] = "content",
    parse_docstring: Optional[bool] = False,
    error_on_invalid_docstring: Optional[bool] = True,
) -> BaseTool:
    """
    Convenience wrapper for ``Tool.create_tool``. It is used by the factories and can also be called
    directly by library users.
    Use this to create lanchain tool instance by passing a callable method

    :param target: The callable function or method that the tool executes.
    :type target: Callable
    :param description: A description of what the tool does.
    :type description: str
    :param return_direct: Whether the response should be returned directly.
    :type return_direct: bool
    :param args_schema: Schema for the arguments of the tool.
    :type args_schema: Optional[ArgsSchema]
    :param infer_schema: Whether to automatically infer the schema from the target function.
    :type infer_schema: Optional[bool]
    :param response_format: The format of the response, either "content" or "content_and_artifact".
    :type response_format: Optional[Literal["content", "content_and_artifact"]]
    :param parse_docstring: Whether to parse the docstring for schema information.
    :type parse_docstring: Optional[bool]
    :param error_on_invalid_docstring: Whether to raise an error if the docstring is invalid.
    :type error_on_invalid_docstring: Optional[bool]

    :return: A ``BaseTool`` instance.
    :rtype: BaseTool
    """
    return Tool(
        description=description,
        target=target,
        return_direct=return_direct,
        args_schema=args_schema,
        infer_schema=infer_schema,
        response_format=response_format,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
    ).create_tool()
