# AI Gateway Service

Local agent orchestration service. It builds agents from config, wires tools, and
supports a dynamic scaffolding graph for multi-agent execution.

For repo-wide setup, see `README.md`.

## Entry Points

- `python -m ai_gateway.main` runs the demo agent patterns in `services/ai_gateway/main.py`.
- `python -m ai_gateway.main_scaffold` runs the scaffolding graph flow in
  `services/ai_gateway/main_scaffold.py`.

## Agent Scaffolding

You can create your custom agents and persist them in the system to use later.
This is where Agent Scaffolding comes into picture.

Use scaffolding when you have rather a complex or broader task that a simple AI agent interraction is not enough.
Scaffolding feature will plan and generate sub-agents while auditing the results according to your request.
It is designed for multi-step, multi-skill tasks where you want a planner to break down the work and delegate it to sub-agents.

Typical use cases:

- Research-and-summarize requests that need multiple angles or sources
- Planning + execution tasks where each step needs a focused agent
- Workflows that benefit from QA or retries before returning a result

## Configuration

Config lives under `services/ai_gateway/config/`:

- `agents.yaml` defines agent prompts, tools, and streaming options.
- `tools.yaml` maps tool names to tool targets and descriptions.
- `settings.py` loads environment variables and wiring for embeddings.

The service expects these environment variables (via `services/.env`):

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `QDRANT_HOST`, `QDRANT_PORT`
- `LLM_PROVIDER`, `LLM_HOST`, `LLM_MODEL`, `LLM_EMBED_MODEL`, `LLM_API_KEY`

Notes:

- `LLM_HOST` is required when `LLM_PROVIDER=local` (Ollama).
- `LLM_API_KEY` is required when `LLM_PROVIDER` is `mistral` or `openai`.

Optional overrides:

- `AGENTS_CONFIG_PATH` and `TOOLS_CONFIG_PATH` to point to alternate YAML configs.

## Extending

- Add a new tool implementation under `services/ai_gateway/toolbox/`, then reference it in
  `services/ai_gateway/config/tools.yaml`.
- Add or update agents in `services/ai_gateway/config/agents.yaml` and wire them through
  `AgentFactory` as needed.
