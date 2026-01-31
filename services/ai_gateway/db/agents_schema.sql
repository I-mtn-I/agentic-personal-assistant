CREATE TABLE IF NOT EXISTS tools_config (
    id uuid PRIMARY KEY,
    name text UNIQUE NOT NULL,
    target text NOT NULL,
    description text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_configs (
    id uuid PRIMARY KEY,
    agent_id text NOT NULL,
    agent_name text NOT NULL,
    purpose text NOT NULL,
    system_prompt text NOT NULL,
    version integer NOT NULL,
    team_id uuid NOT NULL,
    is_manager boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (agent_id, version)
);

CREATE TABLE IF NOT EXISTS agent_tools (
    agent_config_id uuid NOT NULL REFERENCES agent_configs(id) ON DELETE CASCADE,
    tool_id uuid NOT NULL REFERENCES tools_config(id) ON DELETE RESTRICT,
    PRIMARY KEY (agent_config_id, tool_id)
);

CREATE TABLE IF NOT EXISTS agentic_team_configs (
    id uuid PRIMARY KEY,
    team_id uuid NOT NULL UNIQUE,
    agent_config_ids uuid[] NOT NULL,
    manager_agent_id uuid NOT NULL REFERENCES agent_configs(id) ON DELETE RESTRICT,
    user_request text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
