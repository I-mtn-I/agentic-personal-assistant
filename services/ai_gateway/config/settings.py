"""
Central configuration module.

*   Loads YAML files for agents & tools.
*   Instantiates the application settings from a `.env` file.
*   Exposes singletons: AGENTS_CONFIG, TOOLS_CONFIG, APP_CONFIG.

All functions raise descriptive errors if files are missing or malformed.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

import yaml
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths (allow override via environment variables)
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_AGENTS_PATH = PACKAGE_DIR / "agents.yaml"
AGENTS_CONFIG_PATH = Path(os.getenv("AGENTS_CONFIG_PATH", DEFAULT_AGENTS_PATH)).resolve()

DEFAULT_TOOLS_PATH = PACKAGE_DIR / "tools.yaml"
TOOLS_CONFIG_PATH = Path(os.getenv("TOOLS_CONFIG_PATH", DEFAULT_TOOLS_PATH)).resolve()

ENV_PATH = PACKAGE_DIR.parents[1] / ".env"


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

    model_config = SettingsConfigDict(env_file=str(ENV_PATH), env_file_encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pydantic models for the config files
# --------------------------------------------------------------------------- #


class BaseToolConfig(BaseModel):
    # The name of the function that lives in the ``tools`` package.
    target: str
    # short description to help the agent when to use the tool.
    description: str


class ToolConfigNamespace:
    """
    Holds a dict for tools.
    Attribute access, dict‑style access, and is iterable.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        builder: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self._raw: Dict[str, Any] = dict(data)
        self._builder = builder
        self._cache: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name not in self._raw:
            raise AttributeError(f"No such tool: {name}")

        if name not in self._cache:
            self._cache[name] = (
                self._builder(name, self._raw[name]) if self._builder else self._raw[name]
            )
        return self._cache[name]

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def list_names(self) -> list[str]:
        return list(self._raw.keys())

    def raw(self, name: str) -> Any:
        return self._raw[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


class BaseAgentConfig(BaseModel):
    streaming: bool
    prompt: str
    tools: list[str] | None = None


class AgentConfigNamespace:
    """
    Holds a dict of objects (raw configs or built Agent instances) and
    exposes them via attribute access:
        ns.helpdesk
        ns.it

    Supports nested agents through sub_agents.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        builder: Optional[Callable[[str, Any], Any]] = None,
    ) -> None:
        """
        :param data:    mapping name → raw object (e.g. BaseAgentConfig)
        :param builder: optional callable that receives (name, raw_obj) and
                        returns the final object to expose.  If ``None`` the
                        raw object is used unchanged.
        """
        self._raw: Dict[str, Any] = dict(data)
        self._builder = builder
        self._cache: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name not in self._raw:
            raise AttributeError(f"No such agent: {name}")

        # Build once and cache
        if name not in self._cache:
            if self._builder:
                self._cache[name] = self._builder(name, self._raw[name])
            else:
                self._cache[name] = self._raw[name]
        return self._cache[name]

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def list_names(self) -> list[str]:
        return list(self._raw.keys())

    def raw(self, name: str) -> Any:
        return self._raw[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


# ----------------------------------------------------------------------
# Loading functions
# ----------------------------------------------------------------------
def _safe_load_agents() -> Dict[str, BaseAgentConfig]:
    """
    Load agents.yaml, validate each entry with ``BaseAgentConfig``,
    and return a plain dict mapping name → BaseAgentConfig.
    """
    raw = yaml.safe_load(AGENTS_CONFIG_PATH.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Agents YAML must be a mapping of name → config")

    configs: Dict[str, BaseAgentConfig] = {}
    errors: Dict[str, ValidationError] = {}

    for name, cfg in raw.items():
        try:
            configs[name] = BaseAgentConfig.model_validate(cfg)
        except ValidationError as e:
            errors[name] = e

    if errors:
        err_msgs = "\n".join(f"{n}: {e.errors()}" for n, e in errors.items())
        raise ValueError(f"Validation errors in agents config:\n{err_msgs}")

    return configs


def _safe_load_tools() -> Dict[str, BaseToolConfig]:
    """
    Load tools.yaml, validate each entry with ``BaseToolConfig``,
    and return a plain dict mapping name → BaseToolConfig.
    """
    if not TOOLS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Tools config not found: {TOOLS_CONFIG_PATH}")

    raw = yaml.safe_load(TOOLS_CONFIG_PATH.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Tools YAML must be a mapping of tool-name -> config")

    configs: Dict[str, BaseToolConfig] = {}
    errors: Dict[str, ValidationError] = {}

    for name, cfg in raw.items():
        try:
            configs[name] = BaseToolConfig.model_validate(cfg)
        except ValidationError as e:
            errors[name] = e

    if errors:
        err_msgs = "\n".join(f"{n}: {e}" for n, e in errors.items())
        raise ValueError(f"Validation errors in tools config:\n{err_msgs}")

    return configs


def _init_llama_embeddings(config: AppConfig) -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name=config.LLM_EMBED_MODEL,
        base_url=config.LLM_HOST,
    )


def _load_env(load_dotenv_file: bool = True) -> AppConfig:
    if load_dotenv_file and ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    config = AppConfig()  # pyright: ignore[reportCallIssue]
    return config


# ----------------------------------------------------------------------
# Public objects
# ----------------------------------------------------------------------
APP_CONFIG = _load_env()
_init_llama_embeddings(APP_CONFIG)

# Agents – iteratable attribute‑style namespace
_AGENTS_DICT = _safe_load_agents()
AGENTS_CONFIG = AgentConfigNamespace(_AGENTS_DICT)

# Tools – iteratable attribute‑style namespace
_TOOLS_DICT = _safe_load_tools()
TOOLS_CONFIG = ToolConfigNamespace(_TOOLS_DICT)
