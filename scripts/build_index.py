"""Build static BM25 and optional vector indexes for product shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasheetindex.product_search import (  # noqa: E402
    build_bm25_index,
    product_search_text,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/build_index.py")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing index.json and products/*.json.",
    )
    parser.add_argument(
        "--vectors",
        action="store_true",
        help="Also build sentence-transformers embeddings.",
    )
    parser.add_argument(
        "--vector-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name for --vectors.",
    )
    return parser


def _build_vectors(
    group_file: Path,
    vector_file: Path,
    idmap_file: Path,
    model_name: str,
) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    with group_file.open("r", encoding="utf-8") as file:
        products = json.load(file)
    model = SentenceTransformer(model_name)
    texts = [product_search_text(product) for product in products]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    vector_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_file, embeddings)
    with idmap_file.open("w", encoding="utf-8") as file:
        json.dump(list(range(len(products))), file)
        file.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    index_path = data_dir / "index.json"
    if not index_path.exists():
        print(f"Error: missing catalog index: {index_path}", file=sys.stderr)
        return 1

    with index_path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    (data_dir / "bm25").mkdir(parents=True, exist_ok=True)
    if args.vectors:
        (data_dir / "vectors").mkdir(parents=True, exist_ok=True)

    for group in catalog.get("groups", []):
        group_id = group["id"]
        group_file = data_dir / group["file"]
        bm25_file = data_dir / "bm25" / f"{group_id}.pkl"
        build_bm25_index(group_file, bm25_file)
        group["bm25_file"] = f"bm25/{group_id}.pkl"

        if args.vectors:
            vector_file = data_dir / "vectors" / f"{group_id}_emb.npy"
            idmap_file = data_dir / "vectors" / f"{group_id}_idmap.json"
            _build_vectors(group_file, vector_file, idmap_file, args.vector_model)
            group["vectors_file"] = f"vectors/{group_id}_emb.npy"
            group["idmap_file"] = f"vectors/{group_id}_idmap.json"

    with index_path.open("w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Built indexes for {len(catalog.get('groups', []))} groups in {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
