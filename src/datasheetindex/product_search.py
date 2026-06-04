"""Static product catalog sharding and search helpers."""

from __future__ import annotations

import functools
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def slugify(value: object, default: str = "uncategorized") -> str:
    """Normalize a product-group label into a stable file-safe id."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or default


def tokenize(text: str) -> list[str]:
    """Tokenize part numbers and prose into lowercase alphanumeric terms."""
    return TOKEN_RE.findall(text.lower())


def lookup_key(value: object) -> str:
    """Normalize exact part-number lookup keys."""
    return "".join(TOKEN_RE.findall(str(value or "").lower()))


def product_group(product: dict[str, Any]) -> str:
    """Return the best available product-group value for a product record."""
    for key in ("productGroup", "product_group", "group", "category", "family"):
        value = product.get(key)
        if value:
            return slugify(value)
    return "uncategorized"


def product_key(product: dict[str, Any]) -> str:
    """Return a stable deduplication key for a product record."""
    for key in ("name", "part_number", "partNumber", "mpn", "id", "url"):
        value = product.get(key)
        if value:
            return str(value)
    return json.dumps(product, sort_keys=True, ensure_ascii=False)


def product_search_text(product: dict[str, Any]) -> str:
    """Build the text used for ranking a product."""
    fields = (
        "name",
        "part_number",
        "partNumber",
        "mpn",
        "title",
        "description",
        "status",
        "productGroup",
        "product_group",
        "family",
        "category",
    )
    values: list[str] = []
    for field in fields:
        value = product.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return " ".join(values)


def load_product_source(source: Path) -> list[dict[str, Any]]:
    """Load products from a JSON file.

    The source may be either a list of product dictionaries or a dictionary with
    one of the common collection keys used by scrapers.
    """
    with source.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        products = payload
    elif isinstance(payload, dict):
        products = None
        for key in ("products", "datasheets", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                products = value
                break
        if products is None:
            raise ValueError(
                f"{source} must contain a product list or one of: "
                "products, datasheets, items, results, data"
            )
    else:
        raise ValueError(f"{source} must be a JSON object or array")

    invalid = [item for item in products if not isinstance(item, dict)]
    if invalid:
        raise ValueError(f"{source} contains non-object product entries")
    return list(products)


def shard_products(
    products: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group products by normalized product group."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        grouped[product_group(product)].append(dict(product))
    return dict(sorted(grouped.items()))


def write_product_shards(
    products: Sequence[dict[str, Any]],
    data_dir: Path,
) -> dict[str, Any]:
    """Write product shards and return the generated catalog index."""
    products_dir = data_dir / "products"
    products_dir.mkdir(parents=True, exist_ok=True)

    grouped = shard_products(products)
    groups: list[dict[str, Any]] = []
    for group_id, group_products in grouped.items():
        file_path = products_dir / f"{group_id}.json"
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(group_products, file, indent=2, ensure_ascii=False)
            file.write("\n")
        groups.append(
            {
                "id": group_id,
                "label": group_id.replace("_", " ").title(),
                "file": f"products/{group_id}.json",
                "count": len(group_products),
            }
        )

    index = {
        "last_update": datetime.now(UTC).isoformat(),
        "total_count": len(products),
        "exact_lookup_file": "exact_lookup.json",
        "groups": groups,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "index.json").open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, ensure_ascii=False)
        file.write("\n")
    write_exact_lookup(products, data_dir / "exact_lookup.json")
    return index


def write_exact_lookup(products: Sequence[dict[str, Any]], output_file: Path) -> None:
    """Write a compact exact lookup table for part numbers and OPN aliases."""
    lookup: dict[str, dict[str, str]] = {}
    for product in products:
        aliases = [
            product.get("name"),
            product.get("part_number"),
            product.get("partNumber"),
            product.get("mpn"),
        ]
        opns = product.get("opns")
        if isinstance(opns, list):
            aliases.extend(opns)
        compact = normalize_product_result(product)
        for alias in aliases:
            key = lookup_key(alias)
            if key and key not in lookup:
                lookup[key] = compact

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(lookup, file, ensure_ascii=False)
        file.write("\n")


@dataclass
class SimpleBM25:
    """Small BM25 implementation used when rank_bm25 is unavailable."""

    corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_count = len(self.corpus)
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avgdl = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        document_frequency: Counter[str] = Counter()
        self.term_frequencies: list[Counter[str]] = []
        for doc in self.corpus:
            frequencies = Counter(doc)
            self.term_frequencies.append(frequencies)
            document_frequency.update(frequencies.keys())
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: Sequence[str]) -> list[float]:
        """Return BM25 scores for each document in the corpus."""
        if not self.doc_count:
            return []
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            doc_len = self.doc_lengths[index]
            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_len / (self.avgdl or 1)
                )
                score += self.idf.get(token, 0.0) * (
                    term_frequency * (self.k1 + 1) / denominator
                )
            scores.append(score)
        return scores


def create_bm25(tokenized_corpus: list[list[str]]) -> Any:
    """Create a BM25 scorer, preferring rank_bm25 when installed."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return SimpleBM25(tokenized_corpus)
    return BM25Okapi(tokenized_corpus)


def build_bm25_index(group_file: Path, output_file: Path) -> None:
    """Build and pickle a BM25 index for one product shard."""
    products = load_product_source(group_file)
    corpus = [product_search_text(product) for product in products]
    tokenized_corpus = [tokenize(document) for document in corpus]
    bm25 = create_bm25(tokenized_corpus)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as file:
        pickle.dump((bm25, corpus, products), file)


def load_catalog(data_dir: Path) -> dict[str, Any]:
    """Load the static catalog index."""
    index_path = data_dir / "index.json"
    if not index_path.exists():
        return {
            "last_update": None,
            "total_count": 0,
            "groups": [],
        }
    with index_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def selected_groups(
    catalog: dict[str, Any],
    query: str,
    explicit_groups: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Choose groups to search, using explicit groups or catalog keywords."""
    groups = list(catalog.get("groups", []))
    if explicit_groups:
        requested = {slugify(group) for group in explicit_groups}
        return [group for group in groups if group.get("id") in requested]

    query_tokens = set(tokenize(query))
    keyword_matches = []
    for group in groups:
        keywords = {slugify(value) for value in group.get("keywords", [])}
        keywords.add(slugify(group.get("id", "")))
        if keywords & query_tokens:
            keyword_matches.append(group)
    return keyword_matches or groups


def _load_bm25_file(path: Path) -> tuple[Any, list[str], list[dict[str, Any]]]:
    with path.open("rb") as file:
        return pickle.load(file)


def normalize_product_result(product: dict[str, Any]) -> dict[str, str]:
    """Return compact API-safe fields for an indexed product."""
    name = (
        product.get("name")
        or product.get("part_number")
        or product.get("partNumber")
        or product.get("mpn")
        or ""
    )
    url = (
        product.get("url")
        or product.get("datasheet_url")
        or product.get("datasheetUrl")
        or product.get("downloadUrl")
        or ""
    )
    return {
        "name": str(name),
        "title": str(product.get("title") or product.get("description") or name),
        "url": str(url),
        "status": str(product.get("status") or ""),
        "product_group": product_group(product),
    }


def search_static_index(
    data_dir: Path,
    query: str,
    top_k: int = 5,
    groups: Sequence[str] | None = None,
    enable_vectors: bool = False,
    vector_model_name: str = "all-MiniLM-L6-v2",
    bm25_weight: float = 0.7,
    vector_weight: float = 0.3,
) -> tuple[list[dict[str, str]], int]:
    """Search BM25 indexes and return compact results plus candidate count."""
    catalog = load_catalog(data_dir)
    exact_results = search_exact_lookup(data_dir, catalog, query, top_k)
    if exact_results:
        return exact_results, int(catalog.get("total_count") or len(exact_results))

    candidate_groups = selected_groups(catalog, query, groups)
    query_tokens = tokenize(query)
    if not query_tokens:
        return [], 0

    query_vector = None
    if enable_vectors and any(group.get("vectors_file") for group in candidate_groups):
        query_vector = encode_query_vector(query, vector_model_name)

    scored: list[tuple[float, dict[str, Any]]] = []
    total_candidates = 0
    for group in candidate_groups:
        bm25_file = group.get("bm25_file") or f"bm25/{group.get('id')}.pkl"
        bm25_path = data_dir / bm25_file
        if not bm25_path.exists():
            continue
        bm25, _corpus, products = _load_bm25_file(bm25_path)
        total_candidates += len(products)
        scores = bm25.get_scores(query_tokens)
        max_bm25 = max(scores) if len(scores) else 0.0
        vector_scores = vector_similarity_scores(data_dir, group, query_vector)
        for index, score in enumerate(scores):
            normalized_bm25 = float(score) / max_bm25 if max_bm25 else 0.0
            vector_score = vector_scores[index] if index < len(vector_scores) else 0.0
            combined_score = (
                bm25_weight * normalized_bm25 + vector_weight * vector_score
            )
            if combined_score > 0:
                scored.append((combined_score, products[index]))

    scored.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    results: list[dict[str, str]] = []
    for _score, product in scored:
        key = product_key(product)
        if key in seen:
            continue
        seen.add(key)
        results.append(normalize_product_result(product))
        if len(results) >= top_k:
            break
    return results, total_candidates


def search_exact_lookup(
    data_dir: Path,
    catalog: dict[str, Any],
    query: str,
    top_k: int,
) -> list[dict[str, str]]:
    """Return exact part-number matches from the sidecar lookup table."""
    key = lookup_key(query)
    if not key:
        return []

    lookup_file = catalog.get("exact_lookup_file") or "exact_lookup.json"
    lookup_path = data_dir / lookup_file
    if not lookup_path.exists():
        return []

    lookup = _load_exact_lookup(str(lookup_path), lookup_path.stat().st_mtime_ns)
    result = lookup.get(key)
    if not result:
        return []
    return [result][:top_k]


@functools.lru_cache(maxsize=4)
def _load_exact_lookup(
    lookup_path: str,
    mtime_ns: int,
) -> dict[str, dict[str, str]]:
    _ = mtime_ns
    with Path(lookup_path).open("r", encoding="utf-8") as file:
        return json.load(file)


@functools.lru_cache(maxsize=4)
def _load_vector_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_query_vector(query: str, model_name: str) -> Any | None:
    """Return a normalized query embedding when vector dependencies exist."""
    try:
        model = _load_vector_model(model_name)
        return model.encode([query], normalize_embeddings=True)[0]
    except Exception:
        return None


def vector_similarity_scores(
    data_dir: Path,
    group: dict[str, Any],
    query_vector: Any | None,
) -> list[float]:
    """Return non-negative vector similarity scores for a group."""
    vectors_file = group.get("vectors_file")
    if query_vector is None or not vectors_file:
        return []

    vector_path = data_dir / vectors_file
    if not vector_path.exists():
        return []

    try:
        import numpy as np

        embeddings = np.load(vector_path)
        scores = np.dot(embeddings, query_vector)
    except Exception:
        return []
    return [max(0.0, float(score)) for score in scores]


def default_data_dir() -> Path:
    """Return the configured data directory for scripts and API consumers."""
    value = os.getenv("DATASHEETINDEX_DATA_DIR")
    if value:
        return Path(value)
    return Path(__file__).resolve().parents[2] / "data"
