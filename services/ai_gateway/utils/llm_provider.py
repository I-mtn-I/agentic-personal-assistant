from __future__ import annotations

from typing import Any, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from pydantic import Field, PrivateAttr


class MistralEmbedding(BaseEmbedding):
    model_name: str = Field(default="mistral-embed")
    api_key: Optional[str] = Field(default=None, exclude=True)
    endpoint: Optional[str] = Field(default=None)

    _client: Optional[Any] = PrivateAttr(default=None)

    def _get_client(self) -> Any:
        if self._client is None:
            from langchain_mistralai import MistralAIEmbeddings

            kwargs: dict[str, Any] = {"model": self.model_name}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.endpoint:
                kwargs["endpoint"] = self.endpoint
            self._client = MistralAIEmbeddings(**kwargs)
        return self._client

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_client().embed_query(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await self._get_client().aembed_query(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_client().embed_query(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._get_client().embed_documents(texts)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return await self._get_client().aembed_documents(texts)


def normalize_provider(provider: Optional[str]) -> str:
    if not provider:
        return "local"
    value = provider.strip().lower()
    if value in {"local", "ollama"}:
        return "local"
    if value in {"openai", "mistral"}:
        return value
    raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Use local, mistral, or openai.")


def build_langchain_chat_model(
    config: Any,
    *,
    model_name: Optional[str] = None,
    callbacks: Optional[list[Any]] = None,
    reasoning: bool = False,
) -> Any:
    provider = normalize_provider(getattr(config, "LLM_PROVIDER", None))
    model_id = model_name or getattr(config, "LLM_MODEL", "")

    if provider == "local":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_id,
            base_url=getattr(config, "LLM_HOST", None),
            callbacks=callbacks,
            reasoning=reasoning,
        )

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        kwargs: dict[str, Any] = {"model": model_id, "callbacks": callbacks}
        api_key = getattr(config, "LLM_API_KEY", None)
        if api_key:
            kwargs["api_key"] = api_key
        base_url = getattr(config, "LLM_HOST", None)
        if base_url:
            kwargs["endpoint"] = base_url
        return ChatMistralAI(**kwargs)

    from langchain_openai import ChatOpenAI

    base_url = getattr(config, "LLM_HOST", None)
    api_key = getattr(config, "LLM_API_KEY", None)
    kwargs: dict[str, Any] = {"model": model_id, "api_key": api_key, "callbacks": callbacks}
    if base_url:
        kwargs["base_url"] = base_url

    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        if "base_url" not in kwargs:
            raise
        kwargs.pop("base_url")
        kwargs["openai_api_base"] = base_url
        return ChatOpenAI(**kwargs)


def build_llama_index_llm(config: Any, *, model_name: Optional[str] = None) -> Any:
    provider = normalize_provider(getattr(config, "LLM_PROVIDER", None))
    model_id = model_name or getattr(config, "LLM_MODEL", "")

    if provider == "local":
        from llama_index.llms.ollama import Ollama

        base_url = getattr(config, "LLM_HOST", "") or "http://localhost:11434"
        return Ollama(model=model_id, base_url=base_url)

    from llama_index.llms.openai import OpenAI

    base_url = getattr(config, "LLM_HOST", None)
    api_key = getattr(config, "LLM_API_KEY", None)
    kwargs: dict[str, Any] = {"model": model_id, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def build_llama_index_embed_model(config: Any) -> Any:
    provider = normalize_provider(getattr(config, "LLM_PROVIDER", None))

    if provider == "local":
        from llama_index.embeddings.ollama import OllamaEmbedding

        base_url = getattr(config, "LLM_HOST", "") or "http://localhost:11434"
        return OllamaEmbedding(
            model_name=getattr(config, "LLM_EMBED_MODEL", ""),
            base_url=base_url,
        )

    if provider == "mistral":
        return MistralEmbedding(
            model_name=getattr(config, "LLM_EMBED_MODEL", ""),
            api_key=getattr(config, "LLM_API_KEY", None),
            endpoint=getattr(config, "LLM_HOST", None),
        )

    from llama_index.embeddings.openai import OpenAIEmbedding

    base_url = getattr(config, "LLM_HOST", None)
    api_key = getattr(config, "LLM_API_KEY", None)
    try:
        return OpenAIEmbedding(
            model=getattr(config, "LLM_EMBED_MODEL", ""),
            api_key=api_key,
            base_url=base_url,
        )
    except TypeError:
        return OpenAIEmbedding(
            model_name=getattr(config, "LLM_EMBED_MODEL", ""),
            api_key=api_key,
            base_url=base_url,
        )
