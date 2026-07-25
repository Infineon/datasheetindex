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

`DatasheetArtifacts` gains two fields, `llm_enrichment_incomplete: bool = False` and
`llm_enrichment_notes: tuple[str, ...] = ()`. Only
`build()` knows whether an eligible LLM step was skipped, and only
`build_datasheet` writes the sidecar, so the fact has to travel between them. It
defaults `False` and no existing consumer reads it, so the addition is
compatible; see "`model` is not the whole LLM state" below for why it is needed.

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
| `build_options` | resolved `output_dir`, `output_stem`, `include_summaries`, `model` (and `caption_figures` once the figure spec lands -- it is a `_BuildOptions` field, so it arrives here by construction) |
| `datasheetindex_version` | from `_version.package_version()` |
| `artifacts` | JSON and text filenames, and the **sha256** of each |
| `toc_quality` | the complete `TocQuality`, `details` included |
| `llm_enrichment_incomplete` | true when LLM work this build was *eligible* for did not produce its result -- skipped for want of a client, or run and failed (below) |
| `llm_enrichment_notes` | short reason strings; diagnostic only, not part of validity |

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

### `model` is not the whole LLM state

`build_options.model` records what the caller asked for, and that is not the same
as what the build was able to do. With `model=None`, `DatasheetIndex.build()`
still creates a client of its own when ToC quality is below threshold
(`index.py:565-567`, via `_try_create_default_llm_client`), and that call returns
`None` when credentials are absent. So the LLM path depends on ambient
environment state that no recorded option captures:

1. Build with no credentials configured. ToC quality is 0.3, fallback is eligible,
   no client can be created, the native weak ToC is emitted.
2. Credentials appear -- `.env` written, gateway key exported, the process
   restarted inside a configured environment.
3. Every recorded field still matches: same bytes, same options (`model=None`),
   same version. The degraded artifact is reused, and the LLM fallback that a
   fresh build would now run never runs again.

The artifact is not corrupt, but it is worse than what the current environment can
produce, and nothing would ever dislodge it short of deleting the directory.

**The sidecar therefore records what the build actually did, not just what it was
asked to do:** a single boolean, `llm_enrichment_incomplete`, true when LLM work the
build was eligible for did not produce its result. Reuse is refused when it is set:
the artifact is self-described as degraded, so a rebuild is given the chance to do
better.

Two distinct causes, and **both** must set it. Recording only the first was the
earlier draft's mistake, and it left the larger hole of the two:

1. **No callable was available.** ToC quality below `TOC_FALLBACK_THRESHOLD` and
   `_try_create_default_llm_client` returned `None`. On the `build_datasheet`
   path -- the only path that writes a sidecar -- this is the only *skip* case, since
   `bound.py:166` already rejects `include_summaries` without a `model`.
2. **The work ran and failed, and the failure was swallowed.** `index.py:611-615`
   catches every exception from the ToC fallback, logs a warning, and continues with
   the weak native ToC. The figure spec's caption pass does the same by design. So a
   single transient gateway error, rate limit, or timeout produces an artifact
   indistinguishable from a successful build -- and with only cause 1 recorded, every
   fingerprint field matches on the next request and that artifact is served
   *permanently*. A dropped connection would silently cost the document its ToC for
   the life of the output directory. This is the worse failure, because it needs no
   unusual environment: it is one bad network moment.

`add_summaries` is not a third case: it does not catch (`llm/summarizer.py:35`), so a
summary failure propagates out of `build()` and nothing is written at all.

**What must *not* set it: a fallback that ran and was rejected.** `index.py:593-610`
compares the candidate against the original and declines it on the merits
(`_accept_llm_toc_candidate`). That is a completed decision, not a failure -- the LLM
was consulted and its answer was worse. Marking it incomplete would rebuild and
re-pay the LLM cost on every request for exactly the documents the fallback cannot
help, which is the opposite of what reuse is for. The distinction is therefore
`except` versus `else`, and it is the one thing an implementation must not blur.

**The rule governs both caches, not just the sidecar.** The in-memory check
(`bound.py:186-195`) returns `self._artifacts` on any options match and never
consults the sidecar, so a rule applied only on disk is unreachable in the
commonest retry there is: the same `DatasheetTools` instance, called again after a
transient failure. That is the exact shape of the MCP path -- one instance per
document, held across a session -- so the caption or fallback error would be
returned instantly, forever, and the disk rule would never get a chance to run.
`not self._artifacts.llm_enrichment_incomplete` therefore joins the in-memory gate.
One predicate, checked in both places.

Two consequences, both accepted:

- **A retry is possible but not free.** Each call on a degraded instance pays a
  full rebuild, where today it pays 0.00 s. That is the point for a transient
  failure -- it self-heals on the next call and caches normally afterwards.
- **A permanently degraded environment rebuilds every call.** With no credentials
  configured, the flag never clears, so each `build_datasheet` costs 7-27 s instead
  of being served from memory. This is the same cost profile the disk rule already
  accepts for a fresh instance, and having the two caches agree is worth more than
  saving the degenerate case. If it ever proves to matter, the fix is to treat
  `toc_fallback_no_client` as non-retryable while keeping the raised causes
  retryable -- the notes below already carry the distinction. Deliberately deferred:
  it is an optimization, and splitting one predicate into two before there is
  evidence is how the two caches start disagreeing again.

The sidecar also carries `llm_enrichment_notes`, a list of short reason strings
("toc_fallback_no_client", "toc_fallback_raised", "figure_caption_raised"). Purely
diagnostic -- validity keys on the boolean alone -- but it is what makes a sidecar
answer "why is this document rebuilding every time" without a re-run.

Recorded rather than derived from the stored `toc_quality.score`, for the reason
the two rules above already give: `to_dict` omitting empty fields and `details`
not being serialized both showed that inferring build history from the deliverable
does not work. Here it fails in a specific way -- a fallback that *ran* and was
rejected also leaves the final score below threshold, so a derived rule could not
tell "no client" from "client ran, candidate rejected" and would rebuild the
PCN-shaped document on every request, re-paying the LLM cost that reuse exists to
avoid.

Two directions this deliberately does **not** treat as invalidating:

- **A fallback that ran and was rejected, or ran and was accepted.** LLM output is
  not deterministic, so a rebuild would not reproduce it; serving the stored one is
  the stable answer, and stability is what makes an artifact citable across
  requests.
- **Credentials disappearing after a successful LLM build.** Reuse then serves an
  artifact better than a fresh build could produce. That is a gift, not a defect.

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
- `llm_enrichment_incomplete` is false;
- both artifacts load, and the **sha256 of the bytes actually read** equals the
  recorded hash.

Anything else rebuilds. There is no repair path and no partial reuse.

### Flow

`build_datasheet(options)`:

1. In-memory hit -> return it. Still 0.00 s, but the gate gains one condition:
   `not self._artifacts.llm_enrichment_incomplete`. See below -- leaving the
   in-memory check unchanged would have made the sidecar rule unreachable in the
   commonest retry.
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
   `text_content`, `toc_quality`, `nodes`, `llm_enrichment_incomplete`, and
   `llm_enrichment_notes`. The first six are load-bearing --
   `get_artifact_manifest` reads `json_data`, and `search_text` /
   `get_section_text` read `text_content`, so a partially populated instance
   would fail later and at a distance. The last two are `False` and `()` by
   construction, since validity already requires the flag to be false; set them
   from the sidecar anyway rather than hardcoding, so the reloaded object and the
   record it came from cannot disagree, and so the in-memory gate in step 1 reads a
   real value on a reload-then-reuse path. Return it.
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

Two properties of the environment make a naively written hit test impossible to
pass. Both are measured, not anticipated.

**Every reuse-hit fixture must carry bookmarks.** A synthetic PDF built with
`pymupdf.open()` and `new_page()` has no ToC, which scores **0.00** against a
`TOC_FALLBACK_THRESHOLD` of `0.3` -- so the fallback is eligible, CI has no
credentials, no client can be created, and `llm_enrichment_incomplete` is set.
Every such document is therefore permanently uncacheable, and *every* hit test
written on a bare synthetic PDF fails. Two `set_toc` entries on a three-page
document score **0.62**, comfortably clear. So hit fixtures call `set_toc`, and the
fixture helper should assert the resulting score is above the threshold rather than
leaving a future contributor to rediscover this from a baffling failure.

**Every test that expects a reuse *hit* must force the editability probe to
`False`.** The suite runs from an editable checkout, where reuse is disabled by
design, so a hit test written against the ambient environment cannot pass -- it
would report the editable-install rule rather than the behaviour it names. A
single fixture stubbing the probe is the right shape, applied to each hit test.
Invalidation tests need no stub: they expect a rebuild either way, and one that
passed for the wrong reason would be worthless, so they should stub it too and
assert the rebuild came from the field under test.

- `to_dict` -> `from_dict` round-trips every field, including those omitted
  when empty, and including a node tree with children.
- Reuse hit: two fresh `DatasheetTools` over one PDF; the second leaves the
  JSON's mtime untouched. This is the check that established the problem, so it
  is the check that must invert.
- Invalidation, one test per fingerprint field: changed PDF bytes, changed
  build options, changed version, missing artifact file, truncated artifact
  file. Each must rebuild.
- **`llm_enrichment_incomplete=true` rebuilds** even when every other field
  matches, and a build whose ToC quality is above threshold records it `false` so
  the common path is not permanently uncacheable. Drive this through the flag on
  the sidecar rather than by manipulating credentials, so the test states the rule
  and needs no environment.
- **A raising ToC fallback sets the flag, so its artifact is never reused.** Inject
  a callable whose ToC call raises, build, and assert the sidecar records
  `llm_enrichment_incomplete=true`; then build again with a working callable and
  assert it rebuilt rather than serving the ToC-less artifact. This is the
  one-bad-network-moment case, and without it a transient error is permanent.
- **A rejected fallback candidate does *not* set the flag, and its artifact is
  reused.** Inject a callable returning a candidate that `_accept_llm_toc_candidate`
  declines, and assert the second build reuses. This is the `except`-versus-`else`
  distinction; a test on only one side of it would let an implementation collapse
  the two and silently make every hard document uncacheable.
- **Retry on the *same* instance rebuilds.** Call `build_datasheet` twice on one
  `DatasheetTools`, failing the LLM call on the first and succeeding on the second,
  and assert the second returns a complete artifact. Without the in-memory
  condition this test fails while every disk test still passes, which is precisely
  how the gap survived the last review -- so it belongs beside them permanently.
- **A complete artifact still hits in memory at 0.00 s**, unchanged. The new
  condition must not cost the common path its cache.
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
  make URL sources silently uncacheable. Being a hit test, it needs the
  editability stub above.
- `force_rebuild=True` rebuilds despite a valid sidecar.
- **A failing sidecar write does not fail the build.** Inject the failure at the
  sidecar writer -- monkeypatch it to raise -- and assert both deliverables are
  present and correct and the returned artifacts are complete. Not by making the
  output directory unwritable: that fails the deliverable writes too, so the build
  raises for a different reason and the test would pass while proving nothing
  about the sidecar. Cover the inverse as well: the *next* build finds no sidecar
  and rebuilds rather than erroring.
- No regression: the two deliverables stay byte-identical to a pre-change
  build, and the in-memory cache still returns without rebuilding for a complete
  artifact.

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
- **An artifact whose LLM enrichment did not complete is never reused**, so it
  rebuilds on every request until the enrichment succeeds. Two shapes: no
  credentials configured, which is the degenerate environment and stays broken
  until fixed; and a transient failure, which self-heals on the next request at
  the cost of one rebuild. `datasheet-agent` runs with the gateway configured, so
  its weak-ToC documents (the PCN among them) record the flag `false` and cache
  normally. Paying the rebuild is the price of not pinning a degraded artifact in
  place forever.
- The output directory becomes a cache rather than a scratch space. Deleting it
  remains safe at all times, and is the recovery action for any suspected
  staleness.
