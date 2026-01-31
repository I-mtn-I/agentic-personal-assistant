# ──────────────────────────────────────────────────────────────────────────────
# data_portal/index_builder.py
# ──────────────────────────────────────────────────────────────────────────────
# This module loads **agents** and **tools** from CSV files, validates them with
# Pydantic models, embeds their textual description using an Ollama model and
# stores the vectors in Qdrant.  The code is deliberately kept simple (KISS) and
# type‑safe by using generics and small helper utilities.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable, List, Optional, TypeVar

import pandas as pd
import qdrant_client
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, ValidationError

# --------------------------------------------------------------------------- #
# Local imports
# --------------------------------------------------------------------------- #
from data_portal.config import APP_CONFIG
from data_portal.helpers import ColoredLogger
from data_portal.models import Agent, Tool

log = ColoredLogger(level="DEBUG")

# --------------------------------------------------------------------------- #
# Generic type for Pydantic models
# --------------------------------------------------------------------------- #
T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def parse_list_field(value: Optional[object]) -> List[str]:
    """
    Turn a CSV cell like ``"a; b; c"`` into a list of stripped strings.
    ``None``, ``np.nan`` and ``""`` are treated as “no data”.
    """
    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _apply_list_parsers(df: pd.DataFrame, columns: Iterable[str], parser: Callable[[object], List[str]]) -> pd.DataFrame:
    """Apply *parser* to each column in *columns* (in‑place)."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(parser)
    return df


def load_csv_as_models(
    csv_path: Path,
    model_cls: type[T],
    list_columns: Optional[Iterable[str]] = None,
) -> List[T]:
    """
    Generic CSV loader:

    * reads ``csv_path`` with pandas,
    * optionally parses list‑type columns,
    * validates each row against ``model_cls``,
    * returns a list of successfully created model instances.
    """
    df = pd.read_csv(csv_path).fillna("")  # turn NaN into empty string

    if list_columns:
        _apply_list_parsers(df, list_columns, parse_list_field)

    instances: List[T] = []
    for _, row in df.iterrows():
        try:
            instances.append(model_cls(**row.to_dict()))
        except ValidationError as exc:
            log.error(
                "Row %s failed validation for %s: %s",
                row.get("id", "<unknown>"),
                model_cls.__name__,
                exc,
            )
    return instances


def _get_embedding_model() -> OllamaEmbedding:
    """Create (or reuse) the Ollama embedding model defined in APP_CONFIG."""
    return OllamaEmbedding(
        model_name=APP_CONFIG.LLM_EMBED_MODEL,
        base_url=APP_CONFIG.LLM_HOST,
    )


def _get_qdrant_client() -> qdrant_client.QdrantClient:
    """Factory for a Qdrant client - keeps host/port in one place."""
    return qdrant_client.QdrantClient(
        host=APP_CONFIG.QDRANT_HOST,
        port=int(APP_CONFIG.QDRANT_PORT),
    )


def build_vector_index(
    docs: List[Document],
    collection_name: str,
) -> VectorStoreIndex:
    """
    Create a Qdrant‑backed ``VectorStoreIndex`` from a list of ``Document`` objects.
    The function is reusable for any collection (agents, tools, …).
    """
    Settings.embed_model = _get_embedding_model()
    client = _get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(docs, storage_context=storage_context)


# --------------------------------------------------------------------------- #
# Domain‑specific loaders & index builders
# --------------------------------------------------------------------------- #
def load_agents(csv_path: Path) -> List[Agent]:
    """Load ``agents.csv`` - parses ``tools_list`` and ``tags`` as list fields."""
    return load_csv_as_models(
        csv_path,
        Agent,
        list_columns=("tools_list", "tags"),
    )


def load_tools(csv_path: Path) -> List[Tool]:
    """Load ``tools.csv`` - parses ``tags`` as a list field."""
    return load_csv_as_models(
        csv_path,
        Tool,
        list_columns=("tags",),
    )


def build_agent_index(agents: List[Agent]) -> VectorStoreIndex:
    """Index agents; each document contains the full prompt and all metadata."""
    docs = [
        Document(
            text=agent.full_prompt,
            metadata=agent.model_dump(),
        )
        for agent in agents
    ]
    return build_vector_index(docs, collection_name="agents")


def build_tool_index(tools: List[Tool]) -> VectorStoreIndex:
    """Index tools; document text is a concise description + optional signature."""
    docs = [
        Document(
            text=(f"{tool.name}: {tool.description}" + (f"\nSignature: {tool.signature}" if tool.signature else "")),
            metadata=tool.model_dump(),
        )
        for tool in tools
    ]
    return build_vector_index(docs, collection_name="tools")


def main() -> None:
    """Load CSVs, build their indexes and log the outcomes."""
    data_dir = Path(__file__).resolve().parents[1] / "data_csv"

    agents_path = data_dir / "agents.csv"
    tools_path = data_dir / "tools.csv"

    # ---- Agents -------------------------------------------------
    if not agents_path.is_file():
        log.error(f"Agents CSV not found: {agents_path}")
        return
    agents = load_agents(agents_path)
    if not agents:
        log.warn("No valid agents loaded; exiting.")
        return
    build_agent_index(agents)  # grab agent_index here
    log.info(f"Successfully indexed {len(agents)} agents.")

    # ---- Tools --------------------------------------------------
    if not tools_path.is_file():
        log.error(f"Tools CSV not found: {tools_path}")
        return
    tools = load_tools(tools_path)
    if not tools:
        log.warn("No valid tools loaded; exiting.")
        return
    build_tool_index(tools)  # grab tool_index here
    log.info(f"Successfully indexed {len(tools)} tools.")

    # return agent_index, tool_index tuple to use as RAG for agents in the future


if __name__ == "__main__":
    main()
