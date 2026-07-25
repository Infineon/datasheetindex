# On-disk artifact reuse

Design, 2026-07-25. Status: approved, not implemented.

## Problem

`DatasheetTools.build_datasheet` caches artifacts on the instance
(`self._artifacts`) and gates the cache on that field being set. The check is
therefore satisfiable only within the lifetime of one instance. A fresh
instance re-parses the PDF, re-runs the table scan, and rewrites the JSON and
text file even when a byte-valid copy of both is already on disk.

That is the normal path for the library's main consumer. `datasheet-agent`
holds a `LargePdfToolsCache` that is per-comparison-request, owned by
`compare.run_comparison` and closed in its `finally`, so every request builds
every document from scratch. Its tool wrapper calls `t.build_datasheet()` with
no arguments, which resolves `output_dir` to `<tempdir>/datasheetindex-<uid>` --
stable across processes. Request N's artifacts are consequently on disk when
request N+1 starts, and are ignored.

Measured on a 134-page datasheet (2.6 MB), driving the real code paths:

| operation | cost |
|---|---|
| build, cold | 27.5 s (63-90 s when the table scan falls back to sequential) |
| build, fresh instance with valid artifacts on disk | 27.1 s |
| build, same instance (in-memory hit) | 0.00 s |
| **reload from disk: hash + parse JSON + read text** | **4.9 ms** |

The 4.9 ms is 4.0 ms of PDF hashing plus 0.9 ms of file reading and parsing. It
excludes reopening the document with PyMuPDF, which the page-level tools
(`inspect_page`, `locate_text`, `extract_table_markdown`) need and which
`DatasheetIndex.doc` does lazily; that is logged at roughly 0.0 s elsewhere, so
it is small but not zero and not counted here.

Confirmed deterministically rather than by timing alone: the JSON's mtime moves
on every request, so the artifact is being rewritten, not read.

The gap is 7-27 s per document per request, against a 4.9 ms alternative.

## Scope

One change: `DatasheetTools.build_datasheet` reuses valid on-disk artifacts.

Explicitly **not** in scope -- each considered and rejected during design:

- **A document library or manifest.** `build_batch` already writes the
  artifacts, and consumers can enumerate the output directory. A manifest is
  convenience, not capability.
- **Comparison logic, parameter alignment, or axis discovery.** These live in
  `datasheet-agent`, which already implements them: `comparison_axis.plan_axis`
  plans from the union of all documents, and `comparison_structure.py` is a
  deterministic spec-section probe. Adding them here would duplicate that work
  and contradict the architecture's stated position that the library is a
  pre-processor and toolbox, not an extraction engine.
- **Multi-slot document addressing.** `tools/defs.py` holds one document and
  closes the predecessor on switch. This design makes a switch cost a 4.9 ms
  reload instead of a rebuild, which removes the reason to change the protocol.
  Revisit only if per-hop cost at that scale is ever shown to matter.
- **Changes to `generate_preamble`.** Its 2400-char cap truncates a feature
  list mid-item, which matters only for a preamble-based axis strategy that
  does not exist yet, in the other repository.
- **Any change to `datasheet-agent`.** It benefits with no modification.

## Design

### Placement

Reuse is wired into `DatasheetTools.build_datasheet` (`tools/bound.py`), beside
the existing in-memory cache, which that class already owns by its own
docstring ("Build and cache datasheet artifacts for later MCP queries").

It is deliberately **not** placed in `DatasheetIndex.build()`. That method is
documented as building the two deliverables; making it silently decline to
build would change library semantics for direct callers. `build_batch` and the
CLI keep calling `build()` and therefore always build -- both pass an explicit
`output_dir` and exist to produce artifacts.

New module `core/artifact_cache.py` owns the fingerprint: computing it, reading
and writing the sidecar, and deciding validity. No PyMuPDF import, no knowledge
of how a build works, so it is testable without a PDF.

`models.py` gains `TocNode.from_dict()` and `TocQuality.from_dict()`.

### The sidecar

`<output_dir>/<stem>.build.json`, written alongside the two deliverables.

The deliverables stay byte-identical to today. The ToC JSON is one of the two
documented products; cache bookkeeping is infrastructure, and mixing them would
create an output-compatibility surface for consumers that diff or re-serialize
it. A missing or unreadable sidecar degrades to a rebuild, so the failure
direction is safe.

Contents:

| field | purpose |
|---|---|
| `source_sha256` | identity of the PDF's bytes |
| `source_size` | fast pre-check before hashing |
| `build_options` | resolved `output_dir`, `output_stem`, `include_summaries`, `model` |
| `datasheetindex_version` | from `_version.package_version()` |
| `artifacts` | JSON and text filenames, and their byte sizes |

**Content hash, not path plus mtime.** Three reasons. The JSON's existing
`source` field holds a bare basename, so path identity cannot distinguish two
datasheets sharing a filename in different directories. mtime is scrambled by
copies and checkouts, producing false rebuilds (safe) and, if a re-download
preserves it, false hits (not safe). And hashing the 2.6 MB fixture costs 4 ms
against 7-27 s saved. A useful side effect: a URL source downloads to a fresh
temp filename each run, so path identity never hits, while content identity
does.

**Build options must be recorded, not inferred.** `TocNode.to_dict()` omits
empty fields, so an absent `summary` key cannot be distinguished from
`include_summaries=True` that produced nothing.

**Version equality is exact.** A source checkout reports `0+unknown` from
`package_version()` and so never matches, always rebuilding -- the right
default for development. It also prevents serving artifacts built before a
release that changed their content, as 0.17.3 did for `table_count` semantics
and 0.18.0 did by adding continuation notes.

### Validity

Reuse only when *all* hold: the sidecar parses; `datasheetindex_version` equals
the running version exactly; `source_sha256` matches; `build_options` match;
both artifact files exist and their sizes equal the recorded sizes.

Anything else rebuilds. There is no repair path and no partial reuse.

### Flow

`build_datasheet(options)`:

1. In-memory hit -> return it. Unchanged; still 0.00 s.
2. Compute the fingerprint.
3. Sidecar valid -> read the JSON and text file and populate **every**
   `DatasheetArtifacts` field: `json_path`, `text_path`, `json_data`,
   `text_content`, `toc_quality`, and `nodes`. All six are load-bearing --
   `get_artifact_manifest` reads `json_data`, and `search_text` /
   `get_section_text` read `text_content`, so a partially populated instance
   would fail later and at a distance. Return it.
4. Otherwise -> build as today, then write the sidecar.

Reuse is **default-on**; that is the point, since it is what lets
`datasheet-agent` benefit unmodified. `force_rebuild=True` bypasses a valid
sidecar and rewrites it.

### Error handling

Every failure degrades to a rebuild.

- Missing or corrupt sidecar, or an artifact whose size disagrees: routine.
  Log at debug.
- Deserialization failure: rebuild, but log at warning -- it means `to_dict`
  and `from_dict` have diverged, which is a bug.
- **Sidecar write failure must not fail the build.** The artifacts are correct
  and caching is best-effort, mirroring `_safe_close` in `defs.py`, where
  cleanup failure is logged rather than allowed to discard a good result.
- The sidecar is written temp-then-rename, so a concurrent reader never
  observes a partial file. Two processes building the same document is not a
  correctness problem: last writer wins and, for the non-LLM path, both wrote
  the same bytes.

## Testing

- `to_dict` -> `from_dict` round-trips every field, including those omitted
  when empty, and including a node tree with children.
- Reuse hit: two fresh `DatasheetTools` over one PDF; the second leaves the
  JSON's mtime untouched. This is the check that established the problem, so it
  is the check that must invert.
- Invalidation, one test per fingerprint field: changed PDF bytes, changed
  build options, changed version, missing artifact file, truncated artifact
  file. Each must rebuild.
- `force_rebuild=True` rebuilds despite a valid sidecar.
- An unwritable output directory does not fail an otherwise successful build.
- No regression: the two deliverables stay byte-identical to a pre-change
  build, and the in-memory cache still returns without rebuilding.

Tests must not depend on the `[llm]` or `[layout]` extras, so a plain
`uv sync && pytest` exercises all of the above.

## Consequences

- `datasheet-agent` saves 7-27 s per document per comparison request, with no
  change on its side.
- A document switch on the MCP path costs a reload (4.9 ms, plus a lazy
  PyMuPDF open) instead of a rebuild. The single-slot protocol is unchanged, so
  an agent interleaving documents still pays per hop -- at a cost that no
  longer blocks. Note this is not the same as the in-memory hit, which stays at
  0.00 s.
- A third file appears in the output directory. Consumers that enumerate it
  should expect `<stem>.build.json` beside the JSON and text file.
- The output directory becomes a cache rather than a scratch space. Deleting it
  remains safe at all times, and is the recovery action for any suspected
  staleness.
