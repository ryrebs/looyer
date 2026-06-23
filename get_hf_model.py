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
    {
        "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    },
]


# Full repo models (embeddings, rerankers)
repo_models = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-base",
]


def get_models(model_source):
    if model_source == "gguf":
        for m in gguf_models:
            path = hf_hub_download(
                repo_id=m["repo_id"],
                filename=m["filename"],
                cache_dir=CACHE_DIR,
            )
            print(f"GGUF: {path}")
        return

    if model_source == "repo":
        for repo_id in repo_models:
            path = snapshot_download(repo_id=repo_id, cache_dir=CACHE_DIR)
            print(f"Repo: {path}")
        return

    print("\n\nUsage: python3 get_hf_model.py <model_source: gguf | repo>")


if __name__ == "__main__":
    import sys

    model_source = sys.argv[1:]
    if model_source:
        get_models(model_source[0])
    else:
        print("\n\nUsage: python3 get_hf_model.py <model_source: gguf | repo>")

