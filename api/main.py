"""FastAPI service for the static Infineon product search index."""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Annotated

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI, Query  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from datasheetindex.product_search import (  # noqa: E402
    default_data_dir,
    search_static_index,
)

app = FastAPI(title="datasheetindex product search")


class SearchResult(BaseModel):
    name: str
    title: str
    url: str
    status: str = ""
    product_group: str = ""


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_candidates: int


@functools.lru_cache(maxsize=256)
def _cached_search(
    data_dir: str,
    query: str,
    top_k: int,
    groups: tuple[str, ...],
) -> tuple[tuple[tuple[str, str], ...], int]:
    results, total_candidates = search_static_index(
        Path(data_dir),
        query,
        top_k=top_k,
        groups=groups or None,
        enable_vectors=os.getenv("DATASHEETINDEX_ENABLE_VECTORS") == "1",
        vector_model_name=os.getenv(
            "DATASHEETINDEX_VECTOR_MODEL",
            "all-MiniLM-L6-v2",
        ),
    )
    immutable = tuple(tuple(sorted(item.items())) for item in results)
    return immutable, total_candidates


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def api_search(
    q: Annotated[
        str,
        Query(min_length=1, description="Keyword or part number"),
    ],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    group: Annotated[list[str] | None, Query()] = None,
) -> SearchResponse:
    data_dir = str(default_data_dir())
    immutable, total_candidates = _cached_search(
        data_dir,
        q.strip(),
        top_k,
        tuple(group or ()),
    )
    results = [SearchResult(**dict(item)) for item in immutable]
    return SearchResponse(results=results, total_candidates=total_candidates)
