import os
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./.cache/hf")

# Single-file GGUF models
gguf_models = [
    {
        "repo_id": "MaziyarPanahi/Saul-Instruct-v1-GGUF",
        "filename": "Saul-Instruct-v1.Q4_K_M.gguf",
    },
    {
        "repo_id": "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF",
        "filename": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    },
]

# Full repo models (embeddings, rerankers)
repo_models = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-base",
]


def get_models():
    for m in gguf_models:
        path = hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            cache_dir=CACHE_DIR,
        )
        print(f"GGUF: {path}")

    for repo_id in repo_models:
        path = snapshot_download(repo_id=repo_id, cache_dir=CACHE_DIR)
        print(f"Repo: {path}")


if __name__ == "__main__":
    get_models()
