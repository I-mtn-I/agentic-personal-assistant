"""Spawn a crew of agents from persisted team configuration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from ai_gateway.domain.agent import Agent
from ai_gateway.domain.agent_factory import AgentFactory, ToolFactory
from ai_gateway.domain.agent_persistence import AgentConfigRepository, TeamConfigRecord
from ai_gateway.utils.streaming import StreamSession, build_streaming_session


@dataclass(frozen=True)
class SpawnedCrewResult:
    team_record: TeamConfigRecord
    manager_agent: Agent
    manager_stream: StreamSession | None


class CrewSpawner:
    """Load team configuration and build a manager agent with sub-agents attached."""

    def __init__(self, repo: AgentConfigRepository) -> None:
        self._repo = repo

    def spawn_team(
        self,
        *,
        team_id: uuid.UUID | None = None,
        stream_response: bool = True,
    ) -> SpawnedCrewResult:
        if team_id is not None:
            team_record = self._repo.get_team_config(team_id=team_id)
        else:
            team_record = self._repo.get_latest_team_config()

        if not team_record:
            raise ValueError("No team configuration found in database.")

        manager_record = self._repo.get_agent_config_by_id(agent_config_id=team_record.manager_agent_id)
        if not manager_record:
            raise ValueError("Manager agent config not found for the selected team.")

        manager_stream = build_streaming_session(manager_record.agent_name, is_subagent=False) if stream_response else None
        streaming_callbacks = manager_stream.callbacks if manager_stream else None

        manager_tools = [ToolFactory.get_tool_by_name(tool.name) for tool in manager_record.tools]
        subagents: list[dict] = []
        for agent_config_id in team_record.agent_config_ids:
            if agent_config_id == team_record.manager_agent_id:
                continue
            sub_record = self._repo.get_agent_config_by_id(agent_config_id=agent_config_id)
            if not sub_record:
                raise ValueError(f"Sub-agent config not found: {agent_config_id}")

            sub_tools = [ToolFactory.get_tool_by_name(tool.name) for tool in sub_record.tools]
            subagents.append(
                {
                    "name": sub_record.agent_id,
                    "description": f"{sub_record.agent_name}: {sub_record.purpose}",
                    "system_prompt": sub_record.system_prompt,
                    "tools": sub_tools,
                }
            )

        manager_agent = AgentFactory.build_deep_agent(
            manager_record.agent_name,
            manager_record.system_prompt,
            manager_tools,
            subagents=subagents,
            streaming=stream_response,
            callbacks=streaming_callbacks,
        )

        return SpawnedCrewResult(
            team_record=team_record,
            manager_agent=manager_agent,
            manager_stream=manager_stream,
        )
