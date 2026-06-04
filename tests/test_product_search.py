from __future__ import annotations

import json

from datasheetindex.product_search import (
    build_bm25_index,
    load_catalog,
    search_static_index,
    write_exact_lookup,
    write_product_shards,
)


def test_write_product_shards_groups_by_product_group(tmp_path):
    products = [
        {
            "name": "BSC027N10NS5",
            "title": "OptiMOS 100 V power MOSFET",
            "productGroup": "MOSFET",
        },
        {
            "name": "TLE9879",
            "title": "Embedded power IC",
            "productGroup": "Embedded Power",
        },
    ]

    index = write_product_shards(products, tmp_path)

    assert index["total_count"] == 2
    assert {group["id"] for group in index["groups"]} == {
        "embedded_power",
        "mosfet",
    }
    assert load_catalog(tmp_path)["total_count"] == 2
    assert (tmp_path / "products" / "mosfet.json").exists()


def test_bm25_search_returns_compact_deduplicated_results(tmp_path):
    products = [
        {
            "name": "BSC027N10NS5",
            "title": "OptiMOS 100 V low gate charge MOSFET",
            "url": "https://example.test/bsc.pdf",
            "status": "active",
            "productGroup": "MOSFET",
        },
        {
            "name": "TLE9879",
            "title": "Embedded power bridge driver",
            "url": "https://example.test/tle.pdf",
            "productGroup": "Embedded Power",
        },
    ]
    catalog = write_product_shards(products, tmp_path)
    for group in catalog["groups"]:
        group_file = tmp_path / group["file"]
        bm25_file = tmp_path / "bm25" / f"{group['id']}.pkl"
        build_bm25_index(group_file, bm25_file)
        group["bm25_file"] = f"bm25/{group['id']}.pkl"
    with (tmp_path / "index.json").open("w", encoding="utf-8") as file:
        json.dump(catalog, file)

    results, total_candidates = search_static_index(tmp_path, "low gate MOSFET")

    assert total_candidates == 1
    assert results == [
        {
            "name": "BSC027N10NS5",
            "title": "OptiMOS 100 V low gate charge MOSFET",
            "url": "https://example.test/bsc.pdf",
            "status": "active",
            "product_group": "mosfet",
        }
    ]


def test_exact_lookup_matches_opn_alias_before_bm25(tmp_path):
    products = [
        {
            "name": "BTM9011EP",
            "title": "Integrated full-bridge IC",
            "url": "https://example.test/btm.pdf",
            "productGroup": "Integrated Full-Bridge ICs",
            "opns": ["BTM9011EPXUMA1"],
        }
    ]
    catalog = write_product_shards(products, tmp_path)
    write_exact_lookup(products, tmp_path / "exact_lookup.json")
    with (tmp_path / "index.json").open("w", encoding="utf-8") as file:
        json.dump(catalog, file)

    results, total_candidates = search_static_index(tmp_path, "BTM9011EPXUMA1")

    assert total_candidates == 1
    assert results[0]["name"] == "BTM9011EP"
