import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, cast

import yaml
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_AGENTS_PATH = PACKAGE_DIR / "agents.yaml"
AGENTS_CONFIG_PATH = Path(os.getenv("AGENTS_CONFIG_PATH", DEFAULT_AGENTS_PATH)).resolve()
ENV_PATH = PACKAGE_DIR.parents[1] / ".env"


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


class BaseAgentConfig(BaseModel):
    streaming: bool
    prompt: str


class AgentConfigNamespace:
    """
    Holds a dict of objects (either raw configs or built Agent instances) and
    exposes them **only** via attribute access:

        ns.helpdesk
        ns.it

    The underlying dict can still be inspected with the helper methods below.
    """

    def __init__(
        self,
        data: Dict[str, Any],
        builder: Callable[[str, Any], Any] | None = None,
    ) -> None:
        """
        :param data:    mapping name → raw object (e.g. BaseAgentConfig)
        :param builder: optional callable that receives (name, raw_obj) and
                        returns the final object to expose.  If ``None`` the
                        raw object is used unchanged.
        """
        self._raw: Dict[str, Any] = dict(data)
        self._builder = builder
        self._cache: Dict[str, Any] = {}  # stores built objects

    # ------------------------------------------------------------------
    # Attribute‑only access
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Small utility helpers (still attribute‑only)
    # ------------------------------------------------------------------
    def list_names(self) -> list[str]:
        """Return all defined names."""
        return list(self._raw.keys())

    def raw(self, name: str) -> Any:
        """Access the underlying raw object without building."""
        return self._raw[name]

    # ------------------------------------------------------------------
    # Optional: make the object iterable if you ever need it
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[str]:
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self._raw)


def load_agents() -> Dict[str, BaseAgentConfig]:
    """
    Load and validate agent configs from AGENTS_CONFIG_PATH.
    Returns a dict mapping agent name -> BaseAgentConfig.
    """
    if not AGENTS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Agents config not found: {AGENTS_CONFIG_PATH}")

    raw = yaml.safe_load(AGENTS_CONFIG_PATH.read_text())

    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping of agent-name -> config")

    raw_data = cast(Dict[str, Any], raw)
    configs: Dict[str, BaseAgentConfig] = {}
    errors: Dict[str, ValidationError] = {}
    for name, cfg in raw_data.items():
        try:
            configs[name] = BaseAgentConfig.model_validate(cfg)
        except ValidationError as e:
            errors[name] = e

    if errors:
        err_msgs = "\n".join(f"{n}: {e}" for n, e in errors.items())
        raise ValueError(f"Validation errors in agents config:\n{err_msgs}")

    return configs


def load_env(load_dotenv_file: bool = True) -> AppConfig:
    if load_dotenv_file and ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
    config = AppConfig()  # pyright: ignore[reportCallIssue]
    _init_llama_embeddings(config)
    return config


def _init_llama_embeddings(config: AppConfig) -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name=config.LLM_EMBED_MODEL,
        base_url=config.LLM_HOST,
    )


APP_CONFIG = load_env()
# produce both mapping and attribute-accessible object
_AGENTS_DICT = load_agents()
AGENTS_CONFIG = AgentConfigNamespace(_AGENTS_DICT)
