"""Persistence layer for approved agent configurations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, cast

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extensions import cursor as PgCursor

from ai_gateway.config.settings import AppConfig


@dataclass(frozen=True)
class ToolSpec:
    name: str
    target: str
    description: str


@dataclass(frozen=True)
class AgentConfigRecord:
    id: uuid.UUID
    agent_id: str
    agent_name: str
    purpose: str
    system_prompt: str
    version: int
    team_id: uuid.UUID
    is_manager: bool
    tools: list[ToolSpec]


@dataclass(frozen=True)
class TeamConfigRecord:
    id: uuid.UUID
    team_id: uuid.UUID
    agent_config_ids: list[uuid.UUID]
    manager_agent_id: uuid.UUID
    user_request: str


class AgentConfigRepository:
    """Store approved agent configurations and tool metadata in Postgres."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "AgentConfigRepository":
        dsn = f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
        return cls(dsn=dsn)

    def _connect(self) -> PgConnection:
        return psycopg2.connect(self._dsn)

    def save_agent_config(
        self,
        *,
        agent_id: str,
        agent_name: str,
        purpose: str,
        system_prompt: str,
        tools: list[ToolSpec],
        team_id: uuid.UUID,
        is_manager: bool,
    ) -> uuid.UUID:
        agent_config_id = uuid.uuid4()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur = cast(PgCursor, cur)
                cur.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM agent_configs WHERE agent_id = %s",
                    (agent_id,),
                )
                row = cur.fetchone()
                version = int(row[0]) if row else 1

                cur.execute(
                    """
                    INSERT INTO agent_configs (
                        id, agent_id,
                        agent_name,
                        purpose,
                        system_prompt,
                        version,
                        team_id,
                        is_manager
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(agent_config_id),
                        agent_id,
                        agent_name,
                        purpose,
                        system_prompt,
                        version,
                        str(team_id),
                        is_manager,
                    ),
                )

                for tool in tools:
                    tool_id = self._upsert_tool(
                        cur,
                        name=tool.name,
                        target=tool.target,
                        description=tool.description,
                    )
                    cur.execute(
                        """
                        INSERT INTO agent_tools (agent_config_id, tool_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (str(agent_config_id), str(tool_id)),
                    )

        return agent_config_id

    def save_team_config(
        self,
        *,
        team_id: uuid.UUID,
        agent_config_ids: list[uuid.UUID],
        manager_agent_id: uuid.UUID,
        user_request: str,
    ) -> uuid.UUID:
        team_config_id = uuid.uuid4()

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur = cast(PgCursor, cur)
                cur.execute(
                    """
                    INSERT INTO agentic_team_configs (
                        id, team_id, agent_config_ids, manager_agent_id, user_request
                    )
                    VALUES (%s, %s, %s::uuid[], %s, %s)
                    """,
                    (
                        str(team_config_id),
                        str(team_id),
                        [str(agent_id) for agent_id in agent_config_ids],
                        str(manager_agent_id),
                        user_request,
                    ),
                )

        return team_config_id

    @staticmethod
    def _normalize_uuid_list(raw_value: Any) -> list[uuid.UUID]:
        if raw_value is None:
            return []
        if isinstance(raw_value, (list, tuple)):
            return [uuid.UUID(str(item)) for item in raw_value]
        if isinstance(raw_value, str):
            trimmed = raw_value.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                trimmed = trimmed[1:-1]
            if not trimmed:
                return []
            return [uuid.UUID(item.strip()) for item in trimmed.split(",") if item.strip()]
        return [uuid.UUID(str(raw_value))]

    def get_team_config(self, *, team_id: uuid.UUID) -> TeamConfigRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur = cast(PgCursor, cur)
                cur.execute(
                    """
                    SELECT id, team_id, agent_config_ids, manager_agent_id, user_request
                    FROM agentic_team_configs
                    WHERE team_id = %s
                    """,
                    (str(team_id),),
                )
                row = cur.fetchone()
                if not row:
                    return None

        return TeamConfigRecord(
            id=uuid.UUID(str(row[0])),
            team_id=uuid.UUID(str(row[1])),
            agent_config_ids=self._normalize_uuid_list(row[2]),
            manager_agent_id=uuid.UUID(str(row[3])),
            user_request=row[4],
        )

    def get_latest_team_config(self) -> TeamConfigRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur = cast(PgCursor, cur)
                cur.execute(
                    """
                    SELECT id, team_id, agent_config_ids, manager_agent_id, user_request
                    FROM agentic_team_configs
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                )
                row = cur.fetchone()
                if not row:
                    return None

        return TeamConfigRecord(
            id=uuid.UUID(str(row[0])),
            team_id=uuid.UUID(str(row[1])),
            agent_config_ids=self._normalize_uuid_list(row[2]),
            manager_agent_id=uuid.UUID(str(row[3])),
            user_request=row[4],
        )

    def get_latest_agent_config(self, *, agent_id: str) -> AgentConfigRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur = cast(PgCursor, cur)
                cur.execute(
                    """
                    SELECT
                        id,
                        agent_id,
                        agent_name,
                        purpose,
                        system_prompt,
                        version,
                        team_id,
                        is_manager
                    FROM agent_configs
                    WHERE agent_id = %s
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    (agent_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                agent_config_id = row[0]
                cur.execute(
                    """
                    SELECT t.name, t.target, t.description
                    FROM tools_config t
                    JOIN agent_tools at ON at.tool_id = t.id
                    WHERE at.agent_config_id = %s
                    ORDER BY t.name
                    """,
                    (agent_config_id,),
                )
                tools = [ToolSpec(name=name, target=target, description=description) for name, target, description in cur.fetchall()]

        return AgentConfigRecord(
            id=uuid.UUID(str(row[0])),
            agent_id=row[1],
            agent_name=row[2],
            purpose=row[3],
            system_prompt=row[4],
            version=row[5],
            team_id=uuid.UUID(str(row[6])),
            is_manager=row[7],
            tools=tools,
        )

    def get_agent_config_by_id(self, *, agent_config_id: uuid.UUID) -> AgentConfigRecord | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur = cast(PgCursor, cur)
                cur.execute(
                    """
                    SELECT
                        id,
                        agent_id,
                        agent_name,
                        purpose,
                        system_prompt,
                        version,
                        team_id,
                        is_manager
                    FROM agent_configs
                    WHERE id = %s
                    """,
                    (str(agent_config_id),),
                )
                row = cur.fetchone()
                if not row:
                    return None

                cur.execute(
                    """
                    SELECT t.name, t.target, t.description
                    FROM tools_config t
                    JOIN agent_tools at ON at.tool_id = t.id
                    WHERE at.agent_config_id = %s
                    ORDER BY t.name
                    """,
                    (str(agent_config_id),),
                )
                tools = [ToolSpec(name=name, target=target, description=description) for name, target, description in cur.fetchall()]

        return AgentConfigRecord(
            id=uuid.UUID(str(row[0])),
            agent_id=row[1],
            agent_name=row[2],
            purpose=row[3],
            system_prompt=row[4],
            version=row[5],
            team_id=uuid.UUID(str(row[6])),
            is_manager=row[7],
            tools=tools,
        )

    @staticmethod
    def _upsert_tool(cur: PgCursor, *, name: str, target: str, description: str) -> uuid.UUID:
        tool_id = uuid.uuid4()
        cur.execute(
            """
            INSERT INTO tools_config (id, name, target, description)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name)
            DO UPDATE SET
                target = EXCLUDED.target,
                description = EXCLUDED.description,
                updated_at = now()
            RETURNING id
            """,
            (str(tool_id), name, target, description),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Failed to upsert tool; no id returned.")
        return uuid.UUID(str(row[0]))
