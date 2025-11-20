from typing import Any, Dict, List, Mapping, Optional

from langchain.agents import create_agent as lc_create_agent
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ai_gateway.config import AGENTS_CONFIG, APP_CONFIG


class LCAgentModel(BaseModel):
    model: str = Field(
        default_factory=lambda: getattr(APP_CONFIG, "LLM_MODEL", ""),
        description="LLM model name/identifier",
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: getattr(APP_CONFIG, "LLM_HOST", None),
        description="Base URL for the LLM endpoint (optional)",
    )
    tools: List[Any] = Field(
        default_factory=list,
        description="List of tool specs/instances available to the agent",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to seed the agent's behavior",
    )
    state_schema: Optional[Any] = Field(
        default=None,
        description="Optional schema/type for agent state (CompiledStateGraph state)",
    )
    context_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context schema used by the agent (dict form)",
    )
    middleware: List[Any] = Field(
        default_factory=list,
        description="List of middleware or hooks applied to the agent",
    )
    response_format: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected response format descriptor for the agent",
    )
    checkpointer: Optional[Any] = Field(
        default=None,
        description="Optional checkpointer object (for state persistence)",
    )


def create_lc_agent(
    agent_name: str, overrides: Optional[Mapping[str, Any]] = None
) -> CompiledStateGraph[Any]:
    """
    Generate a langchain agent by name.
    - agent_name: key in AGENTS_CONFIG
    - overrides: dict-like of LCAgentModel fields to override.

    Returns the object returned by langchain.agents.create_agent.
    """
    if agent_name not in AGENTS_CONFIG:
        raise KeyError(f"Unknown agent: {agent_name}")

    base_agent_cfg = AGENTS_CONFIG[agent_name]

    cfg_dict = (
        dict(base_agent_cfg) if not isinstance(base_agent_cfg, dict) else base_agent_cfg
    )
    base_agent = LCAgentModel(**cfg_dict)

    lc_cfg = base_agent.model_copy(update=dict(overrides)) if overrides else base_agent

    model_arg: str = lc_cfg.model or ""
    if not model_arg:
        raise ValueError(f"Agent {agent_name} has no model configured")

    _model = ChatOllama(model=model_arg, base_url=lc_cfg.base_url)

    agent_kwargs: Dict[str, Any] = {"model": _model, "name": agent_name}
    if lc_cfg.tools:
        agent_kwargs["tools"] = lc_cfg.tools
    if lc_cfg.system_prompt is not None:
        agent_kwargs["system_prompt"] = lc_cfg.system_prompt
    if lc_cfg.middleware:
        agent_kwargs["middleware"] = lc_cfg.middleware
    if lc_cfg.response_format:
        agent_kwargs["response_format"] = lc_cfg.response_format
    if lc_cfg.state_schema is not None:
        agent_kwargs["state_schema"] = lc_cfg.state_schema
    if lc_cfg.context_schema:
        agent_kwargs["context_schema"] = lc_cfg.context_schema
    if lc_cfg.checkpointer is not None:
        agent_kwargs["checkpointer"] = lc_cfg.checkpointer

    agent = lc_create_agent(**agent_kwargs)

    return agent


def generate_all_agents() -> Dict[str, CompiledStateGraph[Any]]:
    return {name: create_lc_agent(name) for name in AGENTS_CONFIG.keys()}
