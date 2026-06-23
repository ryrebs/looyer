# Looyer

Simple RAG pipeline for answering legal queries.

### Pipeline

```
INDEXING
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