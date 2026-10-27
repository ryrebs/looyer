# Looyer

Simple RAG pipeline for answering legal queries.

### Pipeline

```
INDEXING
                   ┌─────────────────────────────────────────────┐
                   │           AUDIT  components/audit/          │
                   │  audit_current_chunks()  inspect_section()  │
                   │  key = sorted(meta.items())                 │
                   │  count > 1  →  atom fragmented              │
                   └────────────────▲───────────────▲───────────┘
                                    │               │
┌────────────┐     ┌─────────────────┐     ┌───────────────────┐
│   loader/  │────▶│  transformer/   │────▶│    indexer/       │
│load_document│    │ chunk + extract │     │ bge-m3 + ChromaDB │
└────────────┘     └─────────────────┘     └────────┬──────────┘
                                                     │ ChromaDB
QUERYING                                             │
┌────────────────┐     ┌──────────────────┐     ┌───▼──────────┐
│  synthesizer/  │◀────│    reranker/     │◀────│  retrieval/  │
│  IRAC + Saul   │     │ bge-reranker ×5  │     │BM25+Dense+RRF│
└───────┬────────┘     └──────────────────┘     └──────────────┘
        │
        ▼
  ISSUE · RULE · APPLICATION · CONCLUSION
```

### Source data:

- 1987 Constitution
- Civil Code

(More to come..)

### Getting started

0. Populate .env based on env.sample

1. Install dependencies: `pipenv install`

2. Download models using: **get_hf_model.py**

### Future optimizations

- HierarchicalNodeParser for more hierarchy control when parsing

- Multi domain index

- Meta data filters during retrieval

- Evaluation

- Query caching

- Guardrails

- Security considerations

## Qdrant

```
docker run -p 127.0.0.1:6333:6333 \
  --rm \
  --name looyer \
  -v "$(pwd)/data/qdrant:/qdrant/storage:z" \
  qdrant/qdrant
```

## Graphify (optional)

Local knowledge graph of the codebase for AI coding assistants. Output goes to `graphify-out/` (gitignored).

```
# Install the skill and hooks for Claude Code
pipenv run python -m graphify install --project

# Generate the graph (local, no LLM)
pipenv run python -m graphify update .

# Update the graph after code changes (only changed files are re-parsed)
pipenv run python -m graphify update .

# Remove the skill and hooks, then delete the generated graph
pipenv run python -m graphify uninstall --project
rm -rf graphify-out/
```
