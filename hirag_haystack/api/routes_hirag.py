"""Native HiRAG API routes.

Endpoints:
- GET /api/health - Health check
- POST /api/query - Query the knowledge graph
- POST /api/index - Index documents
- GET /api/graph/stats - Get graph statistics
"""

from fastapi import APIRouter, Depends, HTTPException
from haystack.dataclasses import Document

from hirag_haystack import HiRAG, QueryParam
from hirag_haystack.api.dependencies import (
    get_hirag,
    run_in_executor,
    run_index_with_lock,
)
from hirag_haystack.api.models import (
    GraphStatsResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(prefix="/api", tags=["hirag"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns {"status": "ok"} if the server is running.
    """
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    hirag: HiRAG = Depends(get_hirag),
) -> QueryResponse:
    """Query the HiRAG knowledge graph.

    Supports multiple retrieval modes:
    - naive: Simple chunk-based retrieval
    - hi_local: Entity-level retrieval only
    - hi_global: Community report retrieval only
    - hi_bridge: Cross-community path finding
    - hi_nobridge: Local + global without paths
    - hi: Full hierarchical (local + global + bridge)
    """
    # Build query parameters
    param = QueryParam(
        mode=request.mode,
        top_k=request.top_k,
        top_m=request.top_m,
        response_type=request.response_type,
        only_need_context=request.only_need_context,
    )

    try:
        # Run query in thread executor to avoid blocking
        result = await run_in_executor(
            lambda: hirag.query(query=request.query, mode=request.mode, param=param)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        query=request.query,
        mode=request.mode,
        answer=result.get("answer", ""),
        context=result.get("context", ""),
    )


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    request: IndexRequest,
    hirag: HiRAG = Depends(get_hirag),
) -> IndexResponse:
    """Index documents into the HiRAG system.

    Documents are processed to extract entities and relations,
    detect communities, and generate community reports.
    """
    # Convert request documents to Haystack Documents
    documents = []
    for doc in request.documents:
        meta = doc.meta or {}
        documents.append(Document(content=doc.content, meta=meta))

    try:
        # Run indexing with lock to prevent concurrent writes
        result = await run_index_with_lock(
            lambda: hirag.index(
                documents=documents,
                incremental=request.incremental,
                force_reindex=request.force_reindex,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return IndexResponse(
        status=result.get("status", "unknown"),
        documents_count=result.get("documents_count"),
        new_documents=result.get("new_documents"),
        chunks_count=result.get("chunks_count"),
        new_chunks=result.get("new_chunks"),
        entities_count=result.get("entities_count"),
        relations_count=result.get("relations_count"),
        communities_count=result.get("communities_count"),
    )


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    hirag: HiRAG = Depends(get_hirag),
) -> GraphStatsResponse:
    """Get statistics about the knowledge graph.

    Returns counts of entities, relations, communities, and chunks.
    """

    def _get_stats() -> tuple[int, int, int, int]:
        """Get graph statistics (runs in executor)."""
        graph_store = hirag.graph_store

        # Entity count
        entities = graph_store.get_all_entities() if hasattr(graph_store, "get_all_entities") else []
        entities_count = len(entities)

        # Relation count
        relations = graph_store.get_all_relations() if hasattr(graph_store, "get_all_relations") else []
        relations_count = len(relations)

        # Community count
        communities = hirag.communities or {}
        communities_count = len(communities)

        # Chunk count
        chunks_count = 0
        if hirag.chunk_store and hasattr(hirag.chunk_store, "count_documents"):
            chunks_count = hirag.chunk_store.count_documents()

        return entities_count, relations_count, communities_count, chunks_count

    try:
        # Run in executor to avoid blocking on potentially slow operations
        entities_count, relations_count, communities_count, chunks_count = await run_in_executor(
            _get_stats
        )
        return GraphStatsResponse(
            entities_count=entities_count,
            relations_count=relations_count,
            communities_count=communities_count,
            chunks_count=chunks_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
