# Agentic Personal Assistant Team

> [!WARNING]
> The personal assistance features are not ready
> Project is under heavy development. This means:
> - docs are high likely to be outdated.
> - There's no guarantee of stability.


A fully local, privacy focused, team of ai agents for personal assistance.
This monorepo project aims to handle below tasks at a high level:

- Schedule events, tasks and appointments upon user request (just like calendar) while handling conflicts
- Allows grouping tasks in multiple categories: work, chores, appointments, specialdays...
- Use Eisenhower Matrix technique to prioritize tasks
- Learn from previous interactions to improve over time
- TTS and STT with client-server architecture
- Integration with other calendars: ical, outlook, etc...

## Tech Stack

Project is leveraging following apps and frameworks:

- Ollama: local model setup
- LangChain: agent orchestration
- LlamaIndex: tools, RAG and embeddings
- AI SDK: front-end
- Keycloak: for user management
- FastAPI: middleware
- Qdrant: vector store
- [Nettu](https://github.com/fmeringdal/nettu-scheduler): self-hosted calendar and scheduler server with rest api

## Project Structure

### Services Workspace (`services/`)

Backend services that make up the Agentic Personal Assistant. Currently only `ai_gateway/` is implemented.

- `ai_gateway/`
  - `main.py`: Entry point (`python -m ai_gateway.main`) that wires the demo agent
  - `config/`: Application/agent configuration (`agents.yaml`, `settings.py`)
  - `domain/`: Agent and tool factories
  - `tools/`: Concrete LangChain/LlamaIndex tool implementations
- `example.env`: Template of environment variables consumed by `ai_gateway`
- `pyproject.toml`: Shared dependency + tooling definitions for all Python services
- `ty.toml`: Type-checker configuration
- `uv.lock`: Resolved dependency lock
- `.python-version`: Python version pin (3.13.0)

### Frontend (`frontend/`)

Front-end assets (TODO: AI-SDK implementation)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Setup and Run

1. Copy environment files:
```bash
cp services/example.env services/.env
```

2. Configure `services/.env` with your desired LLM model and settings

3. Start infrastructure services (Ollama, PostgreSQL, Qdrant):
```bash
./start.sh
```

4. Run the AI gateway service:
```bash
cd services
uv run python -m ai_gateway.main
```

## Development Environment Setup

### Python Environment

```bash
cd services

# Ensure Python matches the project pin
pyenv install 3.13.0 --skip-existing
pyenv local 3.13.0

# Create and sync the uv environment
uv venv
source .venv/bin/activate
uv sync --all-groups

# Install git hooks (ruff + ty)
uv run pre-commit install
```

### VS Code Setup

1. Install recommended extensions when prompted, or manually install:
   - Ruff (charliermarsh.ruff)
   - Python (ms-python.python)
   - Ty (optional; CLI-driven by default)
   - Docker (ms-azuretools.vscode-docker)
   - YAML (redhat.vscode-yaml)

2. Add the following to your VS Code user settings (Cmd/Ctrl + Shift + P → "Preferences: Open User Settings (JSON)"):

```json
"[python]": {
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff"
}
```

### Useful Commands

Run from `services/` directory:

- `uv run ruff check ai_gateway` – lint
- `uv run ruff format ai_gateway` – format
- `uv run ty check ai_gateway data_portal` – type-check
- `uv run python -m ai_gateway.main` – run the sample agent

See `.pre-commit-config.yaml` for enforced checks on commits.
