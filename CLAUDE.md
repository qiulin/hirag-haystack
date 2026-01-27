# HiRAG-Haystack

This is a Haystack-based implementation of HiRAG (Hierarchical Retrieval-Augmented Generation).

## Project Overview

HiRAG is a hierarchical knowledge retrieval system that:

- Builds knowledge graphs from documents using entity/relationship extraction
- Detects communities within the graph using Louvain clustering
- Generates community reports for global context
- Supports multiple retrieval modes (naive, local, global, bridge, nobridge)
- Enables cross-community reasoning through path finding

## Project Structure

```
hirag_haystack/
├── __init__.py           # High-level HiRAG API
├── prompts.py            # LLM prompt templates
├── core/
│   ├── graph.py          # Entity, Relation data classes
│   ├── community.py      # Community, CommunitySchema data classes
│   └── query_param.py    # QueryParam, RetrievalMode
├── stores/
│   ├── base.py           # GraphDocumentStore abstract base
│   ├── networkx_store.py # NetworkX implementation (in-memory)
│   ├── neo4j_store.py    # Neo4j implementation (production)
│   └── vector_store.py   # EntityVectorStore, ChunkVectorStore, KVStore
├── components/
│   ├── entity_extractor.py        # Entity/relation extraction
│   ├── hierarchical_entity_extractor.py  # Two-stage hierarchical extraction
│   ├── community_detector.py      # Louvain community detection
│   ├── report_generator.py        # Community report generation
│   ├── hierarchical_retriever.py  # Multi-mode retrieval
│   └── path_finder.py             # Cross-community path finding
├── pipelines/
│   ├── indexing.py       # Document indexing pipeline
│   └── query.py          # Query pipeline
└── utils/
    └── token_utils.py    # tiktoken utilities

examples/
├── basic_usage.py        # Simple indexing and query
└── advanced_queries.py   # All retrieval modes

docs/
├── design.md             # Detailed design document
└── implementation.md     # Implementation plan
```

## Key Concepts

### Retrieval Modes

| Mode          | Description                                 |
| ------------- | ------------------------------------------- |
| `naive`       | Simple chunk-based retrieval                |
| `hi_local`    | Entity-level retrieval only                 |
| `hi_global`   | Community report retrieval only             |
| `hi_bridge`   | Cross-community path finding                |
| `hi_nobridge` | Local + global without paths                |
| `hi`          | Full hierarchical (local + global + bridge) |

### Storage Backends

- **NetworkXGraphStore**: In-memory, uses `python-louvain` for clustering
- **Neo4jGraphStore**: Production, uses Cypher queries and GDS library

### Vector Stores

- **EntityVectorStore**: Stores entity embeddings for semantic search
- **ChunkVectorStore**: Stores document chunks for retrieval
- **KVStore**: Key-value store for metadata

## High-Level API Usage

```python
from hirag_haystack import HiRAG

# Initialize
hirag = HiRAG(working_dir="./hirag_cache")

# Index documents
hirag.index("path/to/document.txt")

# Query (default: hi mode)
result = hirag.query("What are the main themes?")
print(result["answer"])

# Incremental indexing
hirag.index(new_docs, incremental=True)

# Different retrieval modes
hirag.query_local(query)    # Entity-level only
hirag.query_global(query)   # Community reports only
hirag.query_nobridge(query) # Combined without paths
```

## Dependencies

- `haystack-ai>=2.6`: Core framework
- `networkx`: Graph data structure
- `python-louvain`: Community detection
- `tiktoken`: Token counting

Optional:

- `neo4j>=5.0`: Neo4j backend
- `openai>=1.0`: OpenAI LLMs
- `scikit-learn`: Hierarchical clustering

## Implementation Notes

1. **Gleaning**: Entity extraction uses a multi-pass approach to catch missed entities
2. **Incremental updates**: Documents are deduplicated by MD5 hash
3. **Path finding**: Uses shortest path algorithms for cross-community reasoning
4. **Hierarchical clustering**: AgglomerativeClustering when scikit-learn is available

## Testing

```bash
pytest tests/
```
