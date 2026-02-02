"""
Central configuration module.

*   Loads YAML files for agents & tools.
*   Instantiates the application settings from a `.env` file.
*   Exposes singletons: AGENTS_CONFIG, TOOLS_CONFIG, APP_CONFIG.

All functions raise descriptive errors if files are missing or malformed.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Literal, Optional

import yaml
from dotenv import load_dotenv
from llama_index.core import Settings
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_gateway.utils.llm_provider import build_llama_index_embed_model, normalize_provider

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
    QDRANT_HOST: str
    QDRANT_PORT: str
    LLM_PROVIDER: str = "local"
    LLM_HOST: str | None = None
    LLM_MODEL: str
    LLM_MODEL_SMALL: str | None = None
    LLM_MODEL_MEDIUM: str | None = None
    LLM_MODEL_LARGE: str | None = None
    LLM_EMBED_MODEL: str
    LLM_API_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("LLM_PROVIDER")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return normalize_provider(value)

    @model_validator(mode="after")
    def _validate_llm_settings(self) -> "AppConfig":
        provider = self.LLM_PROVIDER
        if provider == "local" and not self.LLM_HOST:
            raise ValueError("LLM_HOST is required when LLM_PROVIDER is 'local'")
        if provider != "local" and not self.LLM_API_KEY:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER is not 'local'")
        return self


# --------------------------------------------------------------------------- #
# Tags
# --------------------------------------------------------------------------- #
AVAILABLE_TAGS = [
    "admin",
    "analysis",
    "data",
    "db",
    "manager",
    "meta",
    "orchestration",
    "planning",
    "report",
    "research",
    "search",
    "support",
    "summary",
    "time",
    "tools",
    "web",
]

# --------------------------------------------------------------------------- #
# Pydantic models for the config files
# --------------------------------------------------------------------------- #


class BaseToolConfig(BaseModel):
    # The name of the function that lives in the ``tools`` package.
    target: str
    # short description to help the agent when to use the tool.
    description: str
    tags: list[str] = Field(default_factory=list)
    disallowed_tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Tool tags must not be empty")
        return value


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
            self._cache[name] = self._builder(name, self._raw[name]) if self._builder else self._raw[name]
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
    model_size: Literal["small", "medium", "large"] | None = Field(
        default=None,
        description="Model size override: small, medium, or large",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Optional tags for static agents",
    )


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
    Settings.embed_model = build_llama_index_embed_model(config)


def _load_env(load_dotenv_file: bool = True) -> AppConfig:
    if load_dotenv_file and ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    config = AppConfig.model_validate(os.environ)
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
