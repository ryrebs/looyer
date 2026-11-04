"""Qdrant client: local Docker or Qdrant Cloud, chosen in .env.

From .env:
    QDRANT_URL      e.g. http://127.0.0.1:6333 (Docker) or https://<cluster>.cloud.qdrant.io:6333
    QDRANT_API_KEY  required by Qdrant Cloud; leave empty for the local Docker service
"""

import os

from qdrant_client import QdrantClient


def build_qdrant_client() -> QdrantClient:
    """Build the Qdrant client from .env. No request is made until it is used."""
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError("QDRANT_URL is not set. See env.sample.")
    return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None)


def check_qdrant(client: QdrantClient) -> None:
    """Fail early with a clear message if Qdrant is unreachable or rejects the API key."""
    try:
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant is not reachable at {os.getenv('QDRANT_URL')}, or QDRANT_API_KEY is wrong. "
            "Local: docker compose up -d qdrant."
        ) from exc
