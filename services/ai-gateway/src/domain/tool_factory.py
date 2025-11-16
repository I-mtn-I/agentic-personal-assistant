from typing import Any, Callable, Dict, Optional, Union
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.runnables.base import Runnable
from langchain.tools import tool as lc_tool


class ToolConfig(BaseModel):
    callable_or_runnable: Union[Callable[..., Any], Runnable] = Field(
        ...,
        description="A callable or a Runnable (agent) to expose as a tool.",
    )
    description: str = Field(
        default="", description="Human-readable description of the tool."
    )
    return_direct: bool = Field(
        default=False, description="Short-circuit return from tool."
    )
    response_format: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional response format descriptor."
    )
    parse_docstring: bool = Field(
        default=False,
        description="Attempt to parse Google-style docstrings for metadata.",
    )
    error_on_invalid_docstring: bool = Field(
        default=True, description="Raise/log when docstring parsing fails."
    )
    is_agent_as_tool: bool = Field(
        default=False,
        description="Force treating the passed value as an agent-backed tool (use string name and runnable).",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


def create_lc_tool(
    callable: Union[Callable[..., Any], Runnable],
    description: str = "",
    overrides: Optional[Dict[str, Any]] = None,
    is_agent_as_tool: bool = False,
) -> BaseTool:
    """
    Create a langchain BaseTool from either a plain callable or a Runnable (agent).
    By default it expects a callable method, preferably using llamaindex (for rag tools etc).
    If you are passing an agent to be used as tool, then is_agent_as_tool should be set to True
    #AgentRights #TheyAreOneOfUs #AgentsAreNotTools #CarsiHerseyeKarsi
    """
    base_tool = ToolConfig(callable_or_runnable=callable, description=description)
    tool_cfg = base_tool.model_copy(update=dict(overrides)) if overrides else base_tool

    is_agent = is_agent_as_tool or bool(tool_cfg.is_agent_as_tool)

    value = tool_cfg.callable_or_runnable

    # Determine values of name_or_callable and runnable according to the value.
    is_runnable_instance = isinstance(value, Runnable)

    if is_agent:
        name_val = getattr(value, "__name__", None) or (
            value.__class__.__name__ if is_runnable_instance else str(value)
        )
        runnable_val = value if is_runnable_instance else None
    else:
        if is_runnable_instance:
            name_val = value.__class__.__name__
            runnable_val = value
        else:
            name_val = value
            runnable_val = None

    tool = lc_tool(
        name_or_callable=name_val,
        runnable=runnable_val,
        description=tool_cfg.description,
        return_direct=tool_cfg.return_direct,
        parse_docstring=tool_cfg.parse_docstring,
        error_on_invalid_docstring=tool_cfg.error_on_invalid_docstring,
    )

    return tool
