import requests

from data_portal.helpers import ColoredLogger

log = ColoredLogger(level="DEBUG")


def embed_text(texts: list[str], embed_model: str, llm_host: str) -> list[list[float]]:
    if not texts:
        return []
    payload = {
        "model": embed_model,
        "input": texts,
        "options": {"embedding": True},
    }
    resp = requests.post(f"{llm_host}/api/embed", json=payload)
    if resp.status_code != 200:
        log.error(f"Ollama embedding failed: {resp.status_code} {resp.text}")
        raise RuntimeError("Embedding failed")
    return resp.json()["embeddings"]
