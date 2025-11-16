# Agentic Gateway


## Current Code Status:

- `src/main.py`: Entry point for the AI Gateway service, orchestrating agent workflows using LangChain's agent framework.
- `src/config/agents.yaml`: Configuration file defining agent behaviors and tool integrations, leveraging LangChain's agent configuration schema.
- `src/config/settings.py`: Centralized configuration for the service, including LlamaIndex data indexing settings and LangChain tool parameters.
- `src/domain/agent_factory.py`: Factory class for creating LangChain agents with LlamaIndex-powered memory and tool integration.
- `src/domain/tool_factory.py`: Factory for LangChain tools that interface with LlamaIndex data indexes for contextual querying.
- `src/tools/data_indexer.py`: Implements LlamaIndex's data indexing capabilities for document retrieval in agent workflows.
- `src/tools/duckduck_search.py`: LangChain tool wrapper for DuckDuckGo search, integrated with LlamaIndex for contextual search results.
- `src/tools/postgres_tool.py`: PostgreSQL database interface using LangChain's tool framework, with LlamaIndex for query optimization.

## Setting up Dev Environment

TODO: properly explain below with bash commands
- install ollama
- ollama pull qwen3:14b
- set up python 3.13
- set up astral uv
- uv venv && uv sync
- uv pip install -e .



#### ADD:
https://hub.docker.com/r/tomsquest/docker-radicale