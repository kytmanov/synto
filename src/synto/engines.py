"""Engine interfaces (V6 §7.4-7.5).

Engines compose a Reader with an LLM client to produce queries, searches,
and answers. Phase 1A wires QueryEngine into pipeline/query.py while the
Phase 0 skeletons remain for compatibility tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .config import Config
from .readers import Reader
from .state import StateDB

if TYPE_CHECKING:
    from .client_factory import RoleEndpoint

# ── Result types ──────────────────────────────────────────────────────────


@dataclass
class Citation:
    article_id: str
    segment_ids: tuple[str, ...] = ()


@dataclass
class Answer:
    text: str
    citations: tuple[Citation, ...] = ()
    title: str | None = None


@dataclass
class Hit:
    article_id: str
    name: str
    snippet: str
    score: float


# ── Configuration objects (placeholder; populated in Phase 1A) ─────────────


@dataclass
class QueryConfig:
    max_pages: int = 5
    # 1-hop expansion only; agent-driven multi-hop traversal is Feature 38's MCP graph walk.
    graph_expand: bool = True


@dataclass
class SearchConfig:
    max_hits: int = 20
    expand_query_with_fast_model: bool = False


# ── Engine interfaces (Phase 0 skeletons) ─────────────────────────────────


class QueryEngine:
    """Minimal Phase 1A adapter over the existing query pipeline."""

    def __init__(
        self,
        reader: Reader,
        fast_ep: RoleEndpoint,
        heavy_ep: RoleEndpoint,
        config: Config,
        db: StateDB | None = None,
        query_config: QueryConfig | None = None,
    ) -> None:
        self.reader = reader
        self.fast_ep = fast_ep
        self.heavy_ep = heavy_ep
        self.config = config
        self.db = db
        self.query_config = query_config or QueryConfig()
        self.last_selected_pages: tuple[str, ...] = ()
        self.last_index_found = True

    def query(self, question: str) -> Answer:
        from .pipeline.query import _query_core

        result = _query_core(
            self.config,
            self.fast_ep,
            self.heavy_ep,
            self.db,
            question,
            max_pages=self.query_config.max_pages,
            graph_expand=self.query_config.graph_expand,
        )
        self.last_selected_pages = tuple(result.selected_pages)
        self.last_index_found = result.index_found
        return Answer(text=result.answer, citations=(), title=result.title)


class SearchEngine(Protocol):
    """Lexical/structural search across a Reader. May call fast model
    for query expansion.

    Implementation lands in Phase 1C.
    """

    reader: Reader

    def search(self, query: str) -> list[Hit]: ...


class MultiPackQueryEngine(Protocol):
    """Aggregates QueryEngines over multiple Readers with priority
    resolution.

    Implementation lands in Phase 1C.
    """

    engines: list[tuple[QueryEngine, int]]

    def query(self, question: str) -> Answer: ...


# ── Skeleton concrete classes (raise NotImplementedError) ─────────────────


class _BaseQueryEngine:
    """Phase 0 skeleton retained for compatibility tests."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader

    def query(self, question: str) -> Answer:
        raise NotImplementedError("QueryEngine.query lands in Phase 1A")


class _BaseSearchEngine:
    """Phase 0 skeleton."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader

    def search(self, query: str) -> list[Hit]:
        raise NotImplementedError("SearchEngine.search lands in Phase 1C")


class _BaseMultiPackQueryEngine:
    """Phase 0 skeleton."""

    def __init__(self, engines: list[tuple[QueryEngine, int]] | None = None) -> None:
        self.engines = engines or []

    def query(self, question: str) -> Answer:
        raise NotImplementedError("MultiPackQueryEngine.query lands in Phase 1C")
