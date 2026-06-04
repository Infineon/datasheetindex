"""Create static product shards from a JSON product source.

This script is intentionally input-driven. The repository does not include a
public Infineon crawler endpoint, so automation can pass a locally generated or
downloaded product JSON file through ``--source`` or ``DATASHEETINDEX_SOURCE``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datasheetindex.product_search import (  # noqa: E402
    load_product_source,
    write_product_shards,
)

INFINEON_PRODUCTS_SITEMAP = (
    "https://www.infineon.com/en.sitemap.products-row-sitemap.xml"
)
USER_AGENT = "datasheetindex product-source builder"
TABLE_API_RE = re.compile(
    r"https?://www\.infineon\.com/dataApi/en/product-table/"
    r"[^\"'<> ]+?\.product-table\.en\.json"
    r"|/dataApi/en/product-table/[^\"'<> ]+?\.product-table\.en\.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/crawler.py")
    parser.add_argument(
        "--infineon-live",
        action="store_true",
        help=(
            "Fetch public Infineon product-table APIs discovered from the "
            "English products sitemap and write a real source catalog."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "JSON product source. Defaults to DATASHEETINDEX_SOURCE, then "
            "data/datasheets.json."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory that will receive index.json and products/*.json.",
    )
    parser.add_argument(
        "--write-source",
        default="data/datasheets.json",
        help="Output path for normalized products when --infineon-live is used.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limit product sitemap pages while testing --infineon-live.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent fetches for --infineon-live.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Create an empty catalog when the source file is missing.",
    )
    return parser


def _fetch_text(url: str, timeout: int = 60) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _extract_sitemap_locs(xml_text: str) -> list[str]:
    return re.findall(r"<loc>(.*?)</loc>", xml_text)


def _absolute_infineon_url(url: str) -> str:
    decoded = html.unescape(url)
    if decoded.startswith("http"):
        return decoded
    return f"https://www.infineon.com{decoded}"


def _discover_table_apis(
    sitemap_url: str = INFINEON_PRODUCTS_SITEMAP,
    max_pages: int = 0,
    workers: int = 8,
) -> list[str]:
    sitemap = _fetch_text(sitemap_url)
    page_urls = _extract_sitemap_locs(sitemap)
    if max_pages:
        page_urls = page_urls[:max_pages]

    table_apis: set[str] = set()

    def fetch_page_tables(url: str) -> set[str]:
        try:
            body = _fetch_text(url)
        except (HTTPError, URLError, TimeoutError):
            return set()
        return {
            _absolute_infineon_url(match.group(0))
            for match in TABLE_API_RE.finditer(body)
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_page_tables, url): url for url in page_urls}
        for count, future in enumerate(as_completed(futures), start=1):
            table_apis.update(future.result())
            if count % 100 == 0:
                print(
                    f"Scanned {count}/{len(page_urls)} product pages; "
                    f"found {len(table_apis)} product-table APIs"
                )
    return sorted(table_apis)


def _parameter_text(parameters: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for parameter in parameters[:24]:
        name = parameter.get("parameterFormatted") or parameter.get("parameterName")
        if not name:
            continue
        value = (
            parameter.get("valueChar")
            or parameter.get("valueNumber")
            or parameter.get("valueMin")
            or parameter.get("valueMax")
        )
        unit = parameter.get("unitFormatted") or parameter.get("unitName") or ""
        if value is None:
            parts.append(str(name))
        else:
            label = html.unescape(str(name))
            unit_text = html.unescape(str(unit))
            parts.append(f"{label}: {value}{unit_text}")
    return "; ".join(parts)


def _top_level_group(page_url: str, fallback: str) -> str:
    match = re.search(r"/products/([^/?#]+)", page_url or "")
    if match:
        return match.group(1).replace("-", " ")
    return fallback


def _normalize_table_item(item: dict[str, Any]) -> dict[str, Any]:
    datasheet = item.get("dataSheet") or {}
    opns = item.get("opns") or []
    statuses = sorted(
        {
            opn.get("productStatusInfo")
            for opn in opns
            if isinstance(opn, dict) and opn.get("productStatusInfo")
        }
    )
    business_segments = sorted(
        {
            opn.get("businessSegment", {}).get("divShortDescr")
            for opn in opns
            if isinstance(opn, dict)
            and isinstance(opn.get("businessSegment"), dict)
            and opn.get("businessSegment", {}).get("divShortDescr")
        }
    )
    family = item.get("familyName") or ""
    product_url = item.get("pageUrl") or ""
    parameter_summary = _parameter_text(item.get("parameterValues") or [])
    name = item.get("ispnName") or item.get("ispnNameURL") or ""
    return {
        "name": name,
        "title": f"{name} - {family}" if family and name else name or family,
        "description": parameter_summary,
        "url": datasheet.get("assetDmPath") or product_url,
        "datasheet_url": datasheet.get("assetDmPath") or "",
        "datasheet_title": datasheet.get("documentDisplayName") or "",
        "product_url": product_url,
        "status": ", ".join(statuses),
        "productGroup": _top_level_group(product_url, family),
        "family": family,
        "family_id": item.get("familyId"),
        "family_url": item.get("familyPageURL") or "",
        "ispn_id": item.get("ispnId"),
        "opns": [
            opn.get("opnName")
            for opn in opns
            if isinstance(opn, dict) and opn.get("opnName")
        ][:20],
        "business_segments": business_segments,
        "source": "infineon_product_table",
    }


def _load_table_products(url: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_fetch_text(url))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [_normalize_table_item(item) for item in payload if isinstance(item, dict)]


def fetch_infineon_live_products(
    max_pages: int = 0,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Fetch public Infineon product-table APIs and normalize products."""
    started = time.monotonic()
    table_apis = _discover_table_apis(max_pages=max_pages, workers=workers)
    print(f"Discovered {len(table_apis)} product-table APIs")

    products_by_name: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_load_table_products, url): url for url in table_apis
        }
        for count, future in enumerate(as_completed(futures), start=1):
            for product in future.result():
                key = product.get("name")
                if key and key not in products_by_name:
                    products_by_name[key] = product
            if count % 100 == 0:
                print(
                    f"Loaded {count}/{len(table_apis)} product tables; "
                    f"{len(products_by_name)} unique products"
                )

    products = sorted(products_by_name.values(), key=lambda item: item.get("name", ""))
    elapsed = time.monotonic() - started
    print(f"Fetched {len(products)} unique Infineon products in {elapsed:.1f}s")
    return products


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)

    if args.infineon_live:
        products = fetch_infineon_live_products(
            max_pages=args.max_pages,
            workers=args.workers,
        )
        source_output = Path(args.write_source)
        source_output.parent.mkdir(parents=True, exist_ok=True)
        with source_output.open("w", encoding="utf-8") as file:
            json.dump(products, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(f"Wrote normalized source catalog to {source_output}")
    else:
        source = Path(
            args.source
            or __import__("os").getenv(
                "DATASHEETINDEX_SOURCE",
                "data/datasheets.json",
            )
        )

        if not source.exists():
            if not args.allow_empty:
                print(
                    f"Error: product source not found: {source}. "
                    "Pass --source, set DATASHEETINDEX_SOURCE, or use "
                    "--infineon-live.",
                    file=sys.stderr,
                )
                return 1
            products = []
        else:
            products = load_product_source(source)

    index = write_product_shards(products, data_dir)
    print(
        f"Wrote {index['total_count']} products across "
        f"{len(index['groups'])} groups to {data_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
