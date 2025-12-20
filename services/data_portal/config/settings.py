import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from data_portal.helpers import ColoredLogger

# --------------------------------------------------------------------------- #
# Paths (allow override via environment variables)
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_AGENTS_PATH = PACKAGE_DIR / "agents.yaml"
AGENTS_CONFIG_PATH = Path(os.getenv("AGENTS_CONFIG_PATH", DEFAULT_AGENTS_PATH)).resolve()

DEFAULT_TOOLS_PATH = PACKAGE_DIR / "tools.yaml"
TOOLS_CONFIG_PATH = Path(os.getenv("TOOLS_CONFIG_PATH", DEFAULT_TOOLS_PATH)).resolve()

ENV_PATH = PACKAGE_DIR.parents[1] / ".env"
log = ColoredLogger(level="DEBUG")


# --------------------------------------------------------------------------- #
# Runtime configuration instances
# --------------------------------------------------------------------------- #
class AppConfig(BaseSettings):
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    LLM_HOST: str
    LLM_MODEL: str
    LLM_EMBED_MODEL: str
    QDRANT_HOST: str
    QDRANT_PORT: str

    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8")


def _load_env(load_dotenv_file: bool = True) -> AppConfig:
    if load_dotenv_file and ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    config = AppConfig()  # pyright: ignore[reportCallIssue]
    return config


def _get_model_embedding_dim(model: str, llm_host: str) -> int:
    """Get the embedding dimension of the model."""

    payload = {
        "model": model,
        "input": "Dimension",
        "options": {"embedding": True},
    }

    response = requests.post(url=f"{llm_host}/api/embed", json=payload)
    if response.status_code == 200:
        data = response.json()
        vector = data["embeddings"][0]
        return len(vector)
    else:
        log.error(f"Failed to get model embedding dimension: {response.text}")
        return 0


APP_CONFIG = _load_env()
EMBEDDING_DIM = _get_model_embedding_dim(APP_CONFIG.LLM_EMBED_MODEL, APP_CONFIG.LLM_HOST)
