# Generated Product Search Data

This directory is intentionally kept free of generated catalog and index files.

Generate the local product source, shards, exact lookup, and BM25 indexes with:

```bash
python scripts/crawler.py --infineon-live
python scripts/build_index.py
```

The generated files include `datasheets.json`, `exact_lookup.json`,
`index.json`, `products/*.json`, and `bm25/*.pkl`.
