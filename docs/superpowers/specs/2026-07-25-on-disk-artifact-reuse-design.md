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
| **reload from disk: hash PDF + read/parse/verify artifacts** | **5.2 ms** |

The 5.2 ms is 4.0 ms of PDF hashing, 0.9 ms of file reading and parsing, and
0.3 ms to hash both artifacts for validate-on-read. It excludes reopening the document with PyMuPDF, which the page-level tools
(`inspect_page`, `locate_text`, `extract_table_markdown`) need and which
`DatasheetIndex.doc` does lazily; that is logged at roughly 0.0 s elsewhere, so
it is small but not zero and not counted here.

Confirmed deterministically rather than by timing alone: the JSON's mtime moves
on every request, so the artifact is being rewritten, not read.

The gap is 7-27 s per document per request, against a 5.2 ms alternative.

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
  closes the predecessor on switch. This design makes a switch cost a 5.2 ms
  reload instead of a rebuild, which removes the reason to change the protocol.
  Revisit only if per-hop cost at that scale is ever shown to matter.
- **Changes to `generate_preamble`.** Its 2400-char cap drops half the front
  matter of a dense datasheet, which is real but separate -- it has its own
  spec, `2026-07-25-preamble-front-matter-design.md`.
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
| `artifacts` | JSON and text filenames, and the **sha256** of each |
| `toc_quality` | the complete `TocQuality`, `details` included |

**Content hash, not path plus mtime.** Three reasons. The JSON's existing
`source` field holds a bare basename, so path identity cannot distinguish two
datasheets sharing a filename in different directories. mtime is scrambled by
copies and checkouts, producing false rebuilds (safe) and, if a re-download
preserves it, false hits (not safe). And hashing the 2.6 MB fixture costs 4 ms
against 7-27 s saved.

**Build options must be recorded, not inferred.** `TocNode.to_dict()` omits
empty fields, so an absent `summary` key cannot be distinguished from
`include_summaries=True` that produced nothing.

**`TocQuality` must be stored, not recomputed.** `assess_toc_quality` populates
a `details` string (`core/quality.py:89`) that `index.py` does **not** serialize
-- the emitted `toc_quality` block carries only `score`, `entry_count`,
`max_depth`, `page_coverage`, `recommend_summaries`. So the deliverable alone
cannot reconstruct the object. Recomputing from the loaded nodes was rejected:
it is deterministic and cheap, but it re-derives rather than restores, so if
scoring ever changes the in-memory object would silently disagree with the
`toc_quality` block in the very JSON it was loaded from. The sidecar is ours to
extend, so it carries the complete object and the deliverable stays
byte-identical.

**Artifact hashes, not sizes.** A size check accepts same-size corruption, and
the text file -- 206 KB of page-marked prose -- is exactly where a same-size
edit is plausible. Hashing both artifacts costs nothing beside the 2.6 MB PDF
hash already being paid, and it is also the mechanism that makes concurrent
reuse safe (see below).

### Version equality is necessary but not sufficient

Exact version equality prevents serving artifacts built before a release that
changed their content, as 0.17.3 did for `table_count` semantics and 0.18.0 did
by adding continuation notes.

It does **not** cover development. `package_version()` returns the real version
under an editable install (measured: `0.23.0`, not `0+unknown`), so editing
`core/structure.py` and re-running would silently serve pre-edit artifacts. And
`0+unknown == 0+unknown` would match anyway, so exact equality could never have
forced a rebuild on its own.

**Reuse is therefore disabled entirely for an editable install**, detected via
the distribution's `direct_url.json` -> `dir_info.editable` (verified present
and `true` for this checkout). Chosen over a source-tree mtime or content
fingerprint because development is exactly where code changes without a version
bump, while a wheel install is immutable and version equality genuinely suffices
there. A binary switch needs no calibration and cannot drift.

### Validity

Reuse only when *all* hold:

- the install is not editable;
- the sidecar parses;
- `datasheetindex_version` equals the running version exactly;
- `source_sha256` matches the resolved PDF;
- `build_options` match;
- both artifacts load, and the **sha256 of the bytes actually read** equals the
  recorded hash.

Anything else rebuilds. There is no repair path and no partial reuse.

### Flow

`build_datasheet(options)`:

1. In-memory hit -> return it. Unchanged; still 0.00 s.
2. **Resolve the source to a local path**, then compute the fingerprint over
   that file. This step is not optional and the spec previously omitted it:
   `DatasheetTools.__init__` stores only the path string, and
   `DatasheetIndex._resolve_pdf_source()` is called lazily from the `doc`
   property. So a fresh instance holding a URL has nothing on disk to hash yet,
   and a local path may still need `_resolve_local_path`'s WSL/Windows
   translation. Fingerprinting therefore calls `_resolve_pdf_source()` first --
   downloading a URL source to its temp file -- and hashes the **resolved**
   file.

   Consequence for URL sources: the download is paid on every request, hit or
   miss, so the saving there is 7-27 s down to *download time*, not down to
   5.2 ms. Still worth having, and content identity is what makes a URL source
   cacheable at all, since it downloads to a fresh temp filename each run and
   path identity could never match.
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

- Missing or corrupt sidecar, or an artifact whose hash disagrees: routine.
  Log at debug.
- Deserialization failure: rebuild, but log at warning -- it means `to_dict`
  and `from_dict` have diverged, which is a bug.
- **Sidecar write failure must not fail the build.** The artifacts are correct
  and caching is best-effort, mirroring `_safe_close` in `defs.py`, where
  cleanup failure is logged rather than allowed to discard a good result.

### Concurrency

The deliverables are written as two separate non-atomic calls
(`json_path.write_text(...)` then `text_path.write_text(...)`, `index.py:649-653`;
the only atomic write anywhere in the package is the URL download's
`NamedTemporaryFile`). So a second process can replace one artifact while a
reader is between them, and the reader assembles a **mixed generation**. A crash
between the two writes leaves that mixed pair on disk permanently.

An atomic sidecar write does not fix this -- it protects the sidecar, not the two
files holding the data.

Two measures, together:

1. **Validate the bytes you read, not the bytes on disk.** Load both artifacts
   into memory, then hash *what was loaded* and compare to the sidecar. A
   straddled or crash-mixed pair fails and rebuilds. Hashing after the read
   rather than stat-ing before it is what closes the window entirely instead of
   narrowing it.
2. **Write order: invalidate, write data, publish.** Remove the sidecar first,
   write each deliverable temp-then-`os.replace`, write the sidecar last and
   atomically. A concurrent reader then either finds no sidecar and rebuilds, or
   finds one and must match both hashes.

Rejected: **generation-stamped artifact filenames** with the sidecar as a
pointer. It is sound, but `<stem>.json` and `<stem>.txt` are the documented
deliverable paths that the CLI, `build_batch`, and `datasheet-agent` all depend
on, and changing them to fix a caching concern is far out of proportion.
Also rejected: a **cross-process lock**, which adds stale-lock detection and
recovery to every build.

Residual, stated plainly: this does not stop two processes both *building* the
same document, which wastes work. It stops a reader *consuming* a mixed pair,
which is the correctness property. Duplicate builds converge because, for the
non-LLM path, both write identical bytes -- verified byte-identical across
processes and across the parallel and sequential table-scan paths.

## Testing

- `to_dict` -> `from_dict` round-trips every field, including those omitted
  when empty, and including a node tree with children.
- Reuse hit: two fresh `DatasheetTools` over one PDF; the second leaves the
  JSON's mtime untouched. This is the check that established the problem, so it
  is the check that must invert.
- Invalidation, one test per fingerprint field: changed PDF bytes, changed
  build options, changed version, missing artifact file, truncated artifact
  file. Each must rebuild.
- **Same-size corruption rebuilds.** Overwrite one byte of the text file in
  place, preserving its length, and assert a rebuild. This is the case a size
  check accepts and a hash rejects, so it is the test that justifies the hash.
- **A mixed generation rebuilds.** Build, then replace only the JSON with a
  differently-hashed one while leaving the text file and sidecar alone, and
  assert a rebuild rather than a mismatched pair being served.
- **An editable install never reuses.** Assert against the editability probe
  directly rather than the ambient environment, so the test states the rule
  instead of merely observing that this checkout happens to be editable.
- `TocQuality` round-trips **including `details`**, which the deliverable does
  not carry -- so this test fails if `details` is dropped from the sidecar.
- A URL source is resolved before fingerprinting: serve a PDF from a local HTTP
  server, build twice with fresh instances, and assert the second reuses. This
  pins the resolve-then-hash ordering, which is the step whose omission would
  make URL sources silently uncacheable.
- `force_rebuild=True` rebuilds despite a valid sidecar.
- An unwritable output directory does not fail an otherwise successful build.
- No regression: the two deliverables stay byte-identical to a pre-change
  build, and the in-memory cache still returns without rebuilding.

Tests must not depend on the `[llm]` or `[layout]` extras, so a plain
`uv sync && pytest` exercises all of the above.

## Consequences

- `datasheet-agent` saves 7-27 s per document per comparison request, with no
  change on its side.
- A document switch on the MCP path costs a reload (5.2 ms, plus a lazy
  PyMuPDF open) instead of a rebuild. The single-slot protocol is unchanged, so
  an agent interleaving documents still pays per hop -- at a cost that no
  longer blocks. Note this is not the same as the in-memory hit, which stays at
  0.00 s.
- A third file appears in the output directory. Consumers that enumerate it
  should expect `<stem>.build.json` beside the JSON and text file.
- **The deliverables are now written atomically** (temp then `os.replace`)
  rather than by direct `write_text`. That is an improvement in its own right --
  a crashed build previously left a truncated JSON where it now leaves the
  previous generation intact -- but it means the files are replaced rather than
  updated in place, so a consumer holding an open handle sees the old content
  until it reopens.
- **No reuse under an editable install**, so a source checkout behaves exactly
  as it does today and contributors see their edits take effect.
- The output directory becomes a cache rather than a scratch space. Deleting it
  remains safe at all times, and is the recovery action for any suspected
  staleness.
