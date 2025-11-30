from typing import Any, Dict

from ai_gateway.config.settings import AGENTS_CONFIG, AgentConfigNamespace
from ai_gateway.domain.agent import Agent

# import all the tools from ai_gateway.tools module
# update yaml to have tools property
# values should match the exported tool names from tools module


class AgentFactory:
    @staticmethod
    def _build_agent(name: str, cfg: Any) -> Agent:
        """Create the concrete Agent instance from a BaseAgentConfig."""
        return Agent(
            name=name,
            prompt=cfg.prompt,
            tools=[],
            middleware=[],
            state_schema=None,
            context_schema={},
            checkpointer=None,
        ).create_agent()

    @staticmethod
    def get_default_agents() -> AgentConfigNamespace:
        """
        Build all agents defined in ``AGENTS_CONFIG`` and return them wrapped
        in an attribute‑only ``AgentConfigNamespace``.
        """
        agents_dict: Dict[str, Any] = {}

        for agent_name, agent_cfg in AGENTS_CONFIG._raw.items():
            print("agent_name: ", agent_name)
            print("agent_config:", agent_cfg)
            agents_dict[agent_name] = AgentFactory._build_agent(agent_name, agent_cfg)

        return AgentConfigNamespace(agents_dict)
