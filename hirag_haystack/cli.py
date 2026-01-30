"""CLI entry point for HiRAG-Haystack.

This module provides a command-line interface for indexing documents
and querying the HiRAG knowledge graph.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from dotenv import load_dotenv

from hirag_haystack import HiRAG, QueryParam, RetrievalMode, __version__
from hirag_haystack.document_loader import DocumentLoader
from hirag_haystack.stores import EntityVectorStore, ChunkVectorStore


# ===== CONFIG FILE HANDLING =====

def _find_config_file() -> Path | None:
    """Find config file in default locations.

    Search order:
    1. ./hirag.yaml
    2. ~/.hirag.yaml

    Returns:
        Path to config file if found, None otherwise.
    """
    candidates = [
        Path("./hirag.yaml"),
        Path.home() / ".hirag.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_config(config_path: Path | None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file, or None to auto-detect.

    Returns:
        Configuration dictionary. Empty dict if no config found.
    """
    if config_path is None:
        config_path = _find_config_file()

    if config_path is None or not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        click.echo(f"Warning: Failed to load config from {config_path}: {e}", err=True)
        return {}


def _resolve_value(
    cli_value: Any,
    config_value: Any,
    env_var: str | None = None,
    default: Any = None,
) -> Any:
    """Resolve a configuration value with precedence.

    Precedence order:
    1. CLI flag (if explicitly provided, i.e., not None)
    2. Config file value
    3. Environment variable
    4. Default value

    Args:
        cli_value: Value from CLI flag.
        config_value: Value from config file.
        env_var: Environment variable name to check.
        default: Default value if all else fails.

    Returns:
        Resolved value.
    """
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    if env_var:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value
    return default


# ===== HIRAG INSTANCE BUILDER =====

def _build_hirag(
    working_dir: str,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    graph_backend: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int = 20,
    top_m: int = 10,
    verbose: bool = False,
) -> HiRAG:
    """Build a HiRAG instance with the given configuration.

    Args:
        working_dir: Working directory for cache and data.
        model: LLM model name.
        api_key: OpenAI API key.
        base_url: API base URL.
        graph_backend: Graph backend ("networkx" or "neo4j").
        chunk_size: Token chunk size.
        chunk_overlap: Chunk overlap.
        top_k: Number of entities to retrieve.
        top_m: Key entities per community.
        verbose: Print progress information.

    Returns:
        Configured HiRAG instance.
    """
    # Resolve API key from environment if not provided
    api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise click.ClickException(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or use --api-key flag."
        )

    # Create generator
    try:
        from haystack.components.generators.chat import OpenAIChatGenerator
    except ImportError:
        raise click.ClickException(
            "OpenAI integration not installed. Run: pip install haystack-ai[openai]"
        )

    generator_kwargs: dict[str, Any] = {"api_key": api_key}
    if model:
        generator_kwargs["model"] = model
    if base_url:
        generator_kwargs["api_base_url"] = base_url

    generator = OpenAIChatGenerator(**generator_kwargs)

    # Create vector stores
    entity_store = EntityVectorStore(
        namespace="entities",
        working_dir=working_dir,
    )
    chunk_store = ChunkVectorStore(
        namespace="chunks",
        working_dir=working_dir,
    )

    # Create HiRAG instance
    hirag = HiRAG(
        working_dir=working_dir,
        graph_backend=graph_backend,
        generator=generator,
        entity_store=entity_store,
        chunk_store=chunk_store,
        top_k=top_k,
        top_m=top_m,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return hirag


# ===== CLI COMMANDS =====

@click.group()
@click.option(
    "-d", "--working-dir",
    type=click.Path(),
    default=None,
    help="Working directory for cache and data (default: ./hirag_cache).",
)
@click.option(
    "-c", "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Config file path (default: auto-detect hirag.yaml).",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    help="Enable verbose output.",
)
@click.version_option(version=__version__, prog_name="hirag")
@click.pass_context
def cli(ctx: click.Context, working_dir: str | None, config: Path | None, verbose: bool) -> None:
    """HiRAG: Hierarchical Retrieval-Augmented Generation CLI.

    Index documents and query knowledge graphs using the HiRAG system.
    """
    # Load .env file
    load_dotenv()

    # Load config
    config_data = _load_config(config)

    # Store in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["config"] = config_data
    ctx.obj["working_dir"] = _resolve_value(
        working_dir,
        config_data.get("working_dir"),
        default="./hirag_cache",
    )
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("sources", nargs=-1, required=True)
@click.option(
    "--incremental/--no-incremental",
    default=None,
    help="Skip already-indexed documents (default: on).",
)
@click.option(
    "--force-reindex",
    is_flag=True,
    default=False,
    help="Force reindex all documents.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Token chunk size (default: 1200).",
)
@click.option(
    "--chunk-overlap",
    type=int,
    default=None,
    help="Chunk overlap (default: 100).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name (default: gpt-4o-mini).",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key (default: OPENAI_API_KEY env var).",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="API base URL for custom endpoints.",
)
@click.option(
    "--graph-backend",
    type=click.Choice(["networkx", "neo4j"]),
    default=None,
    help="Graph backend (default: networkx).",
)
@click.pass_context
def index(
    ctx: click.Context,
    sources: tuple[str, ...],
    incremental: bool | None,
    force_reindex: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    graph_backend: str | None,
) -> None:
    """Index documents from files or URLs.

    SOURCES can be file paths, glob patterns (e.g., "docs/*.pdf"),
    or URLs (e.g., "https://example.com/page.html").

    Examples:

        hirag index document.pdf

        hirag index "docs/**/*.md" --model gpt-4o

        hirag index https://example.com/page.html

        hirag index file1.txt file2.txt --force-reindex
    """
    config = ctx.obj["config"]
    index_config = config.get("index", {})
    verbose = ctx.obj["verbose"]

    # Resolve configuration values
    working_dir = ctx.obj["working_dir"]
    chunk_size = _resolve_value(chunk_size, index_config.get("chunk_size"), default=1200)
    chunk_overlap = _resolve_value(chunk_overlap, index_config.get("chunk_overlap"), default=100)
    model = _resolve_value(model, config.get("model"), default="gpt-4o-mini")
    api_key = _resolve_value(api_key, config.get("api_key"), env_var="OPENAI_API_KEY")
    base_url = _resolve_value(base_url, config.get("base_url"), env_var="OPENAI_BASE_URL")
    graph_backend = _resolve_value(graph_backend, config.get("graph_backend"), default="networkx")
    incremental = _resolve_value(incremental, index_config.get("incremental"), default=True)

    # Load documents
    loader = DocumentLoader(verbose=verbose)

    if verbose:
        click.echo(f"Loading documents from {len(sources)} source(s)...")

    try:
        documents = loader.load(list(sources))
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
    except ImportError as e:
        raise click.ClickException(str(e))

    if not documents:
        raise click.ClickException("No documents found to index.")

    click.echo(f"Loaded {len(documents)} document(s)")

    # Build HiRAG instance
    if verbose:
        click.echo(f"Initializing HiRAG (working_dir={working_dir}, model={model})...")

    try:
        hirag = _build_hirag(
            working_dir=working_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            graph_backend=graph_backend,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=verbose,
        )
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to initialize HiRAG: {e}")

    # Index documents
    if verbose:
        click.echo("Indexing documents...")

    try:
        result = hirag.index(
            documents=documents,
            incremental=incremental,
            force_reindex=force_reindex,
        )
    except Exception as e:
        raise click.ClickException(f"Indexing failed: {e}")

    # Print results
    click.echo("\nIndexing complete!")
    click.echo(f"  Status: {result.get('status', 'unknown')}")

    if "documents_count" in result:
        click.echo(f"  Documents: {result['documents_count']}")
    if "new_documents" in result:
        click.echo(f"  New documents: {result['new_documents']}")
    if "chunks_count" in result:
        click.echo(f"  Chunks: {result['chunks_count']}")
    if "new_chunks" in result:
        click.echo(f"  New chunks: {result['new_chunks']}")
    if "entities_count" in result:
        click.echo(f"  Entities: {result['entities_count']}")
    if "relations_count" in result:
        click.echo(f"  Relations: {result['relations_count']}")
    if "communities_count" in result:
        click.echo(f"  Communities: {result['communities_count']}")


@cli.command()
@click.argument("query", required=False)
@click.option(
    "-m", "--mode",
    type=click.Choice(["naive", "hi_local", "hi_global", "hi_bridge", "hi_nobridge", "hi"]),
    default=None,
    help="Retrieval mode (default: hi).",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Number of entities to retrieve (default: 20).",
)
@click.option(
    "--top-m",
    type=int,
    default=None,
    help="Key entities per community (default: 10).",
)
@click.option(
    "--response-type",
    type=str,
    default=None,
    help='Response format (default: "Multiple Paragraphs").',
)
@click.option(
    "--context-only",
    is_flag=True,
    default=False,
    help="Return only retrieved context, no LLM generation.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name.",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="OpenAI API key.",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="API base URL.",
)
@click.option(
    "--graph-backend",
    type=click.Choice(["networkx", "neo4j"]),
    default=None,
    help="Graph backend.",
)
@click.option(
    "--stdin",
    is_flag=True,
    default=False,
    help="Read query from stdin.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output result as JSON.",
)
@click.pass_context
def query(
    ctx: click.Context,
    query: str | None,
    mode: str | None,
    top_k: int | None,
    top_m: int | None,
    response_type: str | None,
    context_only: bool,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    graph_backend: str | None,
    stdin: bool,
    output_json: bool,
) -> None:
    """Query the knowledge graph.

    QUERY is the question to ask. Can also be provided via --stdin.

    Examples:

        hirag query "What are the main themes?"

        hirag query "Summarize the key concepts" --mode hi_global

        echo "What is AI?" | hirag query --stdin

        hirag query "List entities" --context-only --json
    """
    config = ctx.obj["config"]
    query_config = config.get("query", {})
    verbose = ctx.obj["verbose"]

    # Get query text
    if stdin:
        query_text = sys.stdin.read().strip()
    elif query:
        query_text = query
    else:
        raise click.ClickException("Query is required. Provide as argument or use --stdin.")

    if not query_text:
        raise click.ClickException("Empty query provided.")

    # Resolve configuration values
    working_dir = ctx.obj["working_dir"]
    mode = _resolve_value(mode, query_config.get("mode"), default="hi")
    top_k = _resolve_value(top_k, query_config.get("top_k"), default=20)
    top_m = _resolve_value(top_m, query_config.get("top_m"), default=10)
    response_type = _resolve_value(
        response_type,
        query_config.get("response_type"),
        default="Multiple Paragraphs",
    )
    model = _resolve_value(model, config.get("model"), default="gpt-4o-mini")
    api_key = _resolve_value(api_key, config.get("api_key"), env_var="OPENAI_API_KEY")
    base_url = _resolve_value(base_url, config.get("base_url"), env_var="OPENAI_BASE_URL")
    graph_backend = _resolve_value(graph_backend, config.get("graph_backend"), default="networkx")

    # Build HiRAG instance
    if verbose:
        click.echo(f"Initializing HiRAG (working_dir={working_dir})...")

    try:
        hirag = _build_hirag(
            working_dir=working_dir,
            model=model,
            api_key=api_key,
            base_url=base_url,
            graph_backend=graph_backend,
            chunk_size=1200,  # Not used for query
            chunk_overlap=100,
            top_k=top_k,
            top_m=top_m,
            verbose=verbose,
        )
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to initialize HiRAG: {e}")

    # Build query parameters
    param = QueryParam(
        mode=RetrievalMode(mode),
        top_k=top_k,
        top_m=top_m,
        response_type=response_type,
        only_need_context=context_only,
    )

    if verbose:
        click.echo(f"Querying with mode={mode}, top_k={top_k}, top_m={top_m}...")

    # Execute query
    try:
        result = hirag.query(query=query_text, mode=mode, param=param)
    except Exception as e:
        raise click.ClickException(f"Query failed: {e}")

    # Output result
    if output_json:
        # Convert to JSON-serializable format
        output = {
            "query": query_text,
            "mode": mode,
            "answer": result.get("answer", ""),
            "context": result.get("context", ""),
        }
        click.echo(json.dumps(output, indent=2))
    else:
        if context_only:
            click.echo("\n--- Context ---\n")
            click.echo(result.get("context", "No context retrieved."))
        else:
            click.echo("\n--- Answer ---\n")
            click.echo(result.get("answer", "No answer generated."))

            if verbose and result.get("context"):
                click.echo("\n--- Context ---\n")
                click.echo(result.get("context", ""))


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
