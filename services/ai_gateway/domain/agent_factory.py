# factory.py
import importlib
from typing import Any, Callable, Dict, List, Optional

from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool

from ai_gateway.config.settings import (
    AGENTS_CONFIG,
    APP_CONFIG,
    TOOLS_CONFIG,
    AgentConfigNamespace,
    BaseAgentConfig,
    ToolConfigNamespace,
)
from ai_gateway.domain.agent import Agent
from ai_gateway.domain.scaffolding_models import (
    AgentGeneratorOutput,
    PlannerOutput,
    PlanQAOutput,
    TeamQAOutput,
)
from ai_gateway.domain.tool import create_tool_from_callable
from ai_gateway.utils.streaming import build_streaming_session


class ToolFactory:
    """
    Factory class to build concrete LangChain tools.
    """

    @staticmethod
    def _resolve_tool_callable(name: str) -> Callable:
        tools_pkg = importlib.import_module("ai_gateway.toolbox")
        try:
            return getattr(tools_pkg, name)
        except AttributeError as exc:
            raise AttributeError(f"Tool callable '{name}' not found in the tools package") from exc

    @staticmethod
    def build_tool(target: Callable, description: str) -> BaseTool:
        """Instantiate a concrete LangChain ``BaseTool``."""
        return create_tool_from_callable(target=target, description=description)

    @staticmethod
    def get_default_tools() -> ToolConfigNamespace:
        """Build every tool defined in ``TOOLS_CONFIG``."""
        tools_dict: Dict[str, BaseTool] = {}
        for name, cfg in TOOLS_CONFIG._raw.items():
            # ``cfg`` is a BaseToolConfig (target is a string)
            callable_obj = ToolFactory._resolve_tool_callable(cfg.target)
            tools_dict[name] = ToolFactory.build_tool(target=callable_obj, description=cfg.description)
        return ToolConfigNamespace(tools_dict)

    @staticmethod
    def get_tool_by_name(name: str) -> BaseTool:
        """Fetch a single already‑built tool by its config key."""
        return ToolFactory.get_default_tools()._raw[name]


class AgentFactory:
    """
    Factory class to build concrete LangChain Agents
    """

    @staticmethod
    def build_agent(
        name: str,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        *,
        streaming: bool = False,
        callbacks: Optional[List] = None,
    ):
        """
        Build basic agent with name, prompt and tools
        """
        return Agent(name=name, prompt=prompt, tools=tools).create_agent(streaming=streaming, callbacks=callbacks)

    @staticmethod
    def build_deep_agent(
        name: str,
        prompt: str,
        tools: Optional[List[BaseTool]] = None,
        *,
        subagents: Optional[List[dict]] = None,
        streaming: bool = False,
        callbacks: Optional[List] = None,
        response_format: Optional[Any] = None,
        model_name: Optional[str] = None,
    ):
        """
        Build deep agent with name, prompt, tools, and optional subagents.
        """
        return Agent(
            name=name,
            prompt=prompt,
            tools=tools,
            response_format=response_format,
            model_name=model_name,
        ).create_deep_agent(subagents=subagents, streaming=streaming, callbacks=callbacks)

    @staticmethod
    def _build_agent_from_config(name: str, cfg: BaseAgentConfig) -> Agent:
        """
        Create an ``Agent`` instance and attach the tools listed in the
        agent’s config (if any) by resolving them from toolbox package.
        """
        tool_objs: List[BaseTool] = []
        if cfg.tools:
            for tool_name in cfg.tools:
                # ``tool_name`` refers to the key in tools.yaml
                tool_objs.append(ToolFactory.get_tool_by_name(tool_name))

        response_format = None
        if name == "planner":
            response_format = ToolStrategy(PlannerOutput)
        elif name == "plan_qa":
            response_format = ToolStrategy(PlanQAOutput)
        elif name == "agent_generator":
            response_format = ToolStrategy(AgentGeneratorOutput)
        elif name == "qa":
            response_format = ToolStrategy(TeamQAOutput)

        model_name = None
        if cfg.model_size == "small":
            model_name = APP_CONFIG.LLM_MODEL_SMALL or APP_CONFIG.LLM_MODEL
        elif cfg.model_size == "medium":
            model_name = APP_CONFIG.LLM_MODEL_MEDIUM or APP_CONFIG.LLM_MODEL
        elif cfg.model_size == "large":
            model_name = APP_CONFIG.LLM_MODEL_LARGE or APP_CONFIG.LLM_MODEL

        stream_session = build_streaming_session(name, is_subagent=False) if cfg.streaming else None
        callbacks = stream_session.callbacks if stream_session else None

        return Agent(
            name=name,
            prompt=cfg.prompt,
            tools=tool_objs,
            middleware=[],
            state_schema=None,
            context_schema={},
            checkpointer=None,
            response_format=response_format,
            model_name=model_name,
        ).create_agent(streaming=cfg.streaming, callbacks=callbacks)

    @staticmethod
    def generate_default_agents() -> AgentConfigNamespace:
        """Build all agents defined in ``AGENTS_CONFIG``."""
        agents_dict: Dict[str, Agent] = {}
        for agent_name, agent_cfg in AGENTS_CONFIG._raw.items():
            agents_dict[agent_name] = AgentFactory._build_agent_from_config(agent_name, agent_cfg)
        return AgentConfigNamespace(agents_dict)
