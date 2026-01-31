import qdrant_client
import requests
from qdrant_client.http import models as qmodels

from data_portal.config import APP_CONFIG
from data_portal.helpers import ColoredLogger

log = ColoredLogger(level="DEBUG")
client = qdrant_client.QdrantClient(host=APP_CONFIG.QDRANT_HOST, port=int(APP_CONFIG.QDRANT_PORT))


def get_model_embedding_dim() -> int:
    """Get the embedding dimension of the model."""

    payload = {
        "model": APP_CONFIG.LLM_EMBED_MODEL,
        "input": "Dimension",
        "options": {"embedding": True},
    }

    response = requests.post(url=f"{APP_CONFIG.LLM_HOST}/api/embed", json=payload)
    if response.status_code == 200:
        data = response.json()
        vector = data["embeddings"][0]
        return len(vector)
    else:
        log.error(f"Failed to get model embedding dimension: {response.text}")
        return 0


def init_qdrant_collection(name: str, dim: int) -> None:
    if client.collection_exists(name):
        log.info(f"Collection '{name}' exists, skipping creation...")
        return
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
                # with SSD or NVME, keep this true - prod use
                on_disk=False,
            ),
            # with SSD or NVME, keep this true to save memory - prod use
            on_disk_payload=False,
            # low treshold to activate vector indexing despite small data
            optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=1),
        )
        log.info(f"Created collection '{name}' with dim={dim}.")
    except Exception as e:
        log.error(f"Failed to create collection: \n{e}")


if __name__ == "__main__":
    dim = get_model_embedding_dim()
    init_qdrant_collection("agents", dim)
    init_qdrant_collection("tools", dim)
