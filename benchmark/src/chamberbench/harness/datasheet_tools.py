"""Engine-shared primitives, neutral between the lite and pai engines.

Moved verbatim out of the private repository's ``extract_lite.py`` (Phase 0
of its Pydantic AI migration) so both engines and the chamber benchmark could
share one home; ported here unchanged. Behavior is unchanged -- the move was a
pure refactor guarded by the existing tests.

The docstrings in section (2) ARE the tool schema the model sees, and lite and
pai share this one factory precisely so the two engines cannot drift into
offering different tool surfaces. Define a tool here once; never per engine.

Sections: (1) gateway/client, (2) PDF tools, (3) payload coercions,
(4) text-parse fallback.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from types import MappingProxyType
from typing import Any, Literal

from anthropic import AsyncAnthropic, beta_async_tool

from chamberbench.models import drop_engine_authored_fields

logger = logging.getLogger(__name__)

# datasheetindex types `inspect_page(detail=...)` as Literal["low","medium","high"];
# mirror it so the value is checked where it enters, not swallowed as `str`.
InspectDetail = Literal["low", "medium", "high"]

# ---------------------------------------------------------------------------
# (1) Gateway / client
# ---------------------------------------------------------------------------


def _create_client() -> tuple[AsyncAnthropic, Any]:
    """Create an AsyncAnthropic client using the same env vars as agent.py.

    Returns (client, http_client) where http_client is the custom httpx
    client if TLS verification is disabled, or None otherwise. Callers
    must close both.
    """
    from chamberbench.credentials import setup_credentials, tls_verify_disabled

    setup_credentials()

    kwargs: dict[str, Any] = {
        "api_key": os.environ["ANTHROPIC_API_KEY"],
    }
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    http_client = None
    if tls_verify_disabled():
        # anthropic==1.0.0 requires an httpx2 client -- it rejects a plain
        # httpx.AsyncClient at construction with a TypeError ("Expected an
        # instance of `httpx2.AsyncClient`"). openai==3.3.1 accepts either,
        # so httpx2 is used at every client-construction site in this
        # package for one consistent library rather than two.
        import httpx2

        http_client = httpx2.AsyncClient(verify=False)
        kwargs["http_client"] = http_client

    return AsyncAnthropic(**kwargs), http_client


def _is_claude_model(model: str) -> bool:
    """True for Anthropic Claude models.

    Anthropic's cache_control is wire-format-specific; only attach it when
    the gateway is actually routing to a Claude model. Mirrors
    ``anthropic_path._is_claude_model`` (same name, same logic) --
    established precedent in this codebase, not new logic. Every Claude
    model string in this codebase (DEFAULT_MODEL, ANTHROPIC_SMALL_FAST_MODEL)
    is prefixed "claude" -- a non-Claude model incorrectly matching would
    require an unusual name on this gateway. Skipping the marker for an
    unrecognized prefix costs a missed optimization, never a broken call.
    """
    return model.lower().startswith("claude")


# Reverted from a 1-hour TTL (2026-07-03): a live test against the gateway
# found the extended TTL did not actually survive past ~5 minutes despite
# being accepted without error, and the gateway's own pricing metadata
# (cache_creation_input_token_cost_above_1hr) suggested writes may still be
# billed at the 1-hour 2x rate regardless -- a real cost-regression risk for
# zero observed benefit. Reverted to the plain 5-minute default pending
# confirmation that a given gateway's LiteLLM version actually forwards `ttl`
# through to the underlying provider for this model.
_CACHE_CONTROL: dict[str, Any] = {"type": "ephemeral"}


def _build_pdf_content_block(pdf_path: str, model: str) -> dict[str, Any]:
    """Build a document content block for a PDF file.

    ``model`` gates the block's ``cache_control`` breakpoint (see
    :func:`_is_claude_model`) -- this is the single builder behind every
    small-PDF call, so it covers comparison backfill's repeat-extraction of
    the same PDF and any retry, for free.
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    block: dict[str, Any] = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode("ascii"),
        },
    }
    if _is_claude_model(model):
        block["cache_control"] = _CACHE_CONTROL
    return block


# ---------------------------------------------------------------------------
# (2) PDF tools (datasheetindex)
# ---------------------------------------------------------------------------


class LargePdfToolsCache:
    """Cache of built ``DatasheetTools`` instances, keyed by resolved pdf_path,
    so that several passes over the same document open and index it once.

    **Nothing in this benchmark constructs one.** Every call site here passes
    ``tools_cache=None``, which is the standalone path: the tools are built on
    first use and closed by ``_make_large_pdf_tool_fns``'s own ``_cleanup()``.
    The class is kept because it is part of the tool factory's contract in the
    private repository this harness was extracted from -- there, a caller
    re-extracts each document several times in one request and a cache is what
    keeps that from reopening the same PDFs repeatedly. Building a PDF index is
    expensive in memory that is not returned to the process, so anyone driving
    this factory over the same document more than once wants this rather than
    a second index.

    The one behaviour worth preserving if it is ever used: every close path
    logs a WARNING instead of raising, including single-entry ``evict()``. That
    deliberately diverges from ``_cleanup()``, which still propagates on the
    standalone path -- there the close sits inside the caller's own
    try/finally, so a failure folds into that call's error handling. An
    eviction, by contrast, happens *after* a good result is already in the
    caller's hand, often inside a coroutine gathered concurrently with others,
    where an uncaught exception would abort the whole gather and lose work over
    a cleanup failure. That leniency belongs in ``evict()`` rather than in each
    caller's memory.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def get(self, pdf_path: str) -> Any | None:
        return self._tools.get(pdf_path)

    def remember(self, pdf_path: str, tools: Any) -> None:
        self._tools[pdf_path] = tools

    def get_or_create(self, pdf_path: str):
        """Return the cached DatasheetTools for pdf_path, building the entry if absent.

        Used by the structure-first probe, which runs BEFORE any extraction and so
        cannot rely on a tool call having created the instance. Going through the cache
        is what guarantees the probe's build_datasheet() is the same one the agent's
        later build_datasheet tool call reuses, instead of a second index.
        """
        from datasheetindex import DatasheetTools

        tools = self._tools.get(pdf_path)
        if tools is None:
            tools = DatasheetTools(pdf_path)
            self._tools[pdf_path] = tools
        return tools

    def evict(self, pdf_path: str) -> None:
        """Close and drop one entry, logging (never raising) on failure."""
        tools = self._tools.pop(pdf_path, None)
        if tools is None:
            return
        try:
            tools.close()
        except Exception:
            logger.warning(
                "Failed to close cached DatasheetTools for %s", pdf_path, exc_info=True
            )

    def evict_except(self, keep: set[str]) -> None:
        """Evict every entry not in `keep`, via evict() -- a single bad
        close can't skip or abort the rest of the sweep."""
        for pdf_path in [p for p in self._tools if p not in keep]:
            self.evict(pdf_path)

    def close_all(self) -> None:
        """Defensive final sweep for run_comparison's finally, covering any
        entry that didn't get evicted earlier (e.g. an exception before
        pruning ran)."""
        self.evict_except(keep=set())


def _log_corpus_shape(pdf_path: str, manifest: dict[str, Any]) -> None:
    """Record how much of this document is beyond a text search's reach.

    Open question this exists to answer, not a feature. Figure captioning
    (`caption_figures=True` plus `DATASHEETINDEX_VISION_MODEL`) was evaluated
    and rejected: across the 12 documents reachable locally, every one had a
    healthy text layer (median 1200-1800 chars/page) and every numeric golden
    value was already reachable by `search_text`, so the failure mode captions
    address -- a scanned datasheet, or a parameter table placed as an image --
    did not occur even once. See docs/figure_digest_and_captioning_evidence.md.

    That conclusion is only as good as the corpus behind it, and the corpus is
    ours, not our users'. These two numbers are what would falsify it: a real
    upload with many raster regions AND a near-empty text layer is the scanned
    document the local sample lacks. Both are already computed on every build
    -- the digest by datasheetindex, the density by one stat() on the text
    artifact it already wrote -- so observing them costs a log line, not an
    API call or a page render.

    Deliberately NOT shown to the model. Surfacing the digest is a separate
    change that today's corpus cannot measure; pinned by
    tests/test_datasheet_tools.py.

    Never raises: this runs inside the first tool call of every large-PDF run,
    where an exception would reach the model as a broken document.
    """
    try:
        digest = manifest.get("figures") or {}
        pages = manifest.get("total_pages") or 0
        chars_per_page = -1
        text_path = manifest.get("text_path")
        if text_path and pages:
            try:
                chars_per_page = int(os.path.getsize(str(text_path)) / int(pages))
            except (OSError, ValueError, ZeroDivisionError):
                chars_per_page = -1
        logger.info(
            "corpus shape: pdf=%s pages=%s figures=%s raster=%s captioned=%s chars_per_page=%s",
            os.path.basename(pdf_path),
            pages,
            digest.get("total", "-"),
            digest.get("raster", "-"),
            digest.get("captioned", "-"),
            chars_per_page,
        )
    except Exception:
        logger.debug("corpus shape logging failed", exc_info=True)


# Below this ToC quality score, build_datasheet's OUTPUT tells the agent that
# regenerate_toc exists. Deliberately a local constant rather than a reuse of
# comparison_structure.MIN_TOC_QUALITY, which happens to share the value: that
# one decides whether a spec probe may drive a structure-first axis, this one
# decides whether to mention a repair to a model. Same number today, different
# questions, and coupling them would make tuning either one move the other.
# The floor that matters underneath both is datasheetindex's own 0.300
# auto-fallback gate -- this must stay above it, or the hint fires only for
# documents the library has already regenerated on its own.
_TOC_REPAIR_HINT_BELOW = 0.5


def _is_missing_llm_client(exc: RuntimeError) -> bool:
    """True when `exc` is datasheetindex's "regenerate_toc needs credentials".

    Matched against the library's own exported sentinel rather than a substring
    we keep in step by hand -- the message names the extra and both environment
    variables, so it is the kind of string that gets reworded. The import is
    local and guarded because this runs inside a tool call: if a future release
    moves the constant, the right outcome is that a real error propagates
    normally, not that build_datasheet dies on an ImportError.
    """
    try:
        # Imported from its DEFINITION site, not from `tools.bound`, which merely
        # re-exports it and lists it in no `__all__` -- a refactor dropping that
        # import would silently degrade this helper to `return False`.
        from datasheetindex.index import REGENERATE_TOC_REQUIRES_CLIENT
    except ImportError:  # pragma: no cover - only on an unexpected library layout
        return False
    return str(exc) == REGENERATE_TOC_REQUIRES_CLIENT


def _toc_fallback_threshold() -> float:
    """datasheetindex's own ToC-quality gate, below which it auto-regenerates.

    Read from the library rather than duplicated, so the repair hint's lower
    bound tracks the gate it is defined against. Falls back to the documented
    0.30 if the constant ever moves; being slightly wrong about where the band
    starts only mis-scopes a hint, and is not worth an import-time failure in
    the first tool every large-PDF run calls.
    """
    try:
        from datasheetindex.index import TOC_FALLBACK_THRESHOLD
    except ImportError:  # pragma: no cover - only on an unexpected library layout
        return 0.30
    return float(TOC_FALLBACK_THRESHOLD)


def _artifact_dir_for(pdf_path: str) -> str | None:
    """Where ``build_datasheet`` should write its ToC JSON and page-matched text.

    ``None`` means "the library's own default", which is a SHARED per-UID
    directory under the OS tempdir (``/tmp/datasheetindex-<uid>``) that nothing
    ever cleans. That default is right for a PDF with a real home on disk -- an
    eval corpus file, a chamber fixture -- where writing derived files next to
    the source would dirty the working tree.

    It is wrong for an EPHEMERAL pdf. The API writes each upload into a
    per-request ``TemporaryDirectory`` and deletes it when the request ends, but
    the artifacts landed outside that directory and outlived it: the full
    extracted text of the last uploaded document stayed readable in the pod
    until the next upload overwrote it. Co-locating them with the PDF makes the
    cleanup we already do cover them too -- no new deletion path, no new failure
    mode, and nothing to leak if the process dies mid-request.

    Reuse is not lost where it was ever working. Within one request the
    artifacts live as long as the PDF does, so a comparison's backfill passes
    still reuse them. ACROSS requests the shared default never reused anything
    anyway: every upload is written as ``upload.pdf``, so every build wrote to
    the same ``upload.*`` stem and overwrote its predecessor. The old behaviour
    was residue without reuse; this keeps the reuse and drops the residue.
    """
    parent = os.path.dirname(os.path.abspath(pdf_path))
    tmp_root = os.path.realpath(tempfile.gettempdir())
    real_parent = os.path.realpath(parent)
    if real_parent == tmp_root or real_parent.startswith(tmp_root + os.sep):
        return parent
    return None


def _make_large_pdf_tool_fns(
    pdf_path: str,
    *,
    inspect_page_detail: InspectDetail = "high",
    tools_cache: LargePdfToolsCache | None = None,
    image_return: Literal["wire", "binary"] = "wire",
) -> tuple[list, Callable[[], None], Callable[[], Any]]:
    """Create plain async closures around datasheetindex functions.

    Undecorated on purpose: this is the shared factory both engines build on
    top of. Their docstrings and `Args:` blocks are the prompts the model
    reads, so they must be defined exactly once here; each engine applies
    its own registration decorator via a thin facade (see
    `_make_large_pdf_tools` below for the lite/Anthropic-SDK one).

    Tools are bound to a specific PDF via closure over a DatasheetTools
    instance. The instance is created lazily on first tool call, or reused
    from `tools_cache` if an entry already exists for `pdf_path`.

    The `inspect_page_detail` argument is closure-captured by the
    inspect_page wrapper -- the agent's tool signature stays
    `inspect_page(page: int)` without a detail knob, because the agent
    can't observe its own context budget and the system can. Callers
    that know the model's `max_input_tokens` (e.g. the chamber test
    runner reading `CHAMBER_MODEL_CONFIG`) pick the tier here.
    Available tiers: "low" (~650 vision tokens / page), "medium"
    (~1150), "high" (~2580 -- backward-compatible default).

    The `image_return` argument is likewise closure-captured by inspect_page,
    selecting the shape of the image content it returns: "wire" (default)
    returns Anthropic content-block dicts, the tool_runner wire format the
    lite engine's decorator expects. "binary" returns `list[BinaryContent]`
    with raw (not base64) bytes, which is what pydantic-ai's pai engine
    expects -- pydantic-ai does not understand the Anthropic wire dict.

    Returns (tools_list, cleanup_fn, get_tools_fn). cleanup_fn closes the
    DatasheetTools instance if it was created AND tools_cache is None -- which
    is the only path this benchmark takes. When a `tools_cache` is supplied,
    closing is deferred to whoever owns the cache, since a later pass over the
    same document may still need the instance alive; see
    :class:`LargePdfToolsCache`. get_tools_fn returns the (possibly
    cache-reused) built instance, or None if no tool call has built one yet.
    """
    from datasheetindex import DatasheetTools

    artifact_dir = _artifact_dir_for(pdf_path)
    tools_instance = tools_cache.get(pdf_path) if tools_cache is not None else None

    def _get_tools() -> DatasheetTools:
        nonlocal tools_instance
        if tools_instance is None:
            tools_instance = DatasheetTools(pdf_path)
            if tools_cache is not None:
                tools_cache.remember(pdf_path, tools_instance)
        return tools_instance

    def _cleanup() -> None:
        if tools_cache is None and tools_instance is not None:
            tools_instance.close()

    # Whether this document's ToC has been regenerated. See build_datasheet's
    # latch: it keeps the reuse key steady once the LLM rebuild has been paid
    # for, so an agent that calls build_datasheet again without the argument
    # does not silently buy a second full re-index back to the original outline.
    regenerated = False

    async def build_datasheet(regenerate_toc: bool = False) -> str:
        """Load and index the PDF datasheet. Must be called before other tools.

        Returns the enriched Table of Contents and document metadata for
        orientation. Use this to identify which sections contain the
        parameters you need.

        Check the reported ToC source. "pdf_outline" means the page numbers are
        the document's own bookmarks and can be trusted. "llm_reconstructed"
        means there were no usable bookmarks and the contents were rewritten
        from body text, so every page number is an inference -- confirm a
        section with search_text before reading a range from it.

        Args:
            regenerate_toc: Rebuild the Table of Contents from the body text
                instead of using the PDF's own outline. Read the ToC first and
                set this only if its entries do not identify sections -- an
                outline of "Page 1", "Page 2", ... that names no section, for
                example. It costs an LLM call and a full re-index, so it is a
                one-shot repair and not a retry: calling it a second time
                returns the same rebuilt outline without redoing the work.
        """
        nonlocal regenerated
        t = _get_tools()
        # LATCHED, not just forwarded. regenerate_toc is in the artifact-reuse
        # key, so True-then-plain is a full re-index each way round -- and since
        # the repair hint below is suppressed only while the flag is set, the
        # sequence plain(hint) -> True -> plain(hint again) is reachable. Once a
        # document has been regenerated, every later call keeps it on, which
        # makes "the LLM runs once" a property of this code rather than of the
        # wording in the docstring. It also holds the option set steady for a
        # later comparison backfill pass over the same DatasheetTools instance.
        effective = regenerate_toc or regenerated
        unavailable = False
        # caption_figures=False MUST match comparison_structure.build_spec_probe --
        # DatasheetTools gates artifact reuse on the full option set, so drift here
        # silently rebuilds a document the probe already built. See that function's
        # docstring for why captioning is opted into rather than defaulted on.
        #
        # regenerate_toc is part of that same option set, so a True here rebuilds
        # BY DESIGN (that is the only way to get a different ToC). caption_figures
        # must stay pinned across that rebuild: it is the expensive path, and the
        # agent must not be able to switch captioning on as a side effect of
        # asking for an outline.
        try:
            artifacts = await asyncio.to_thread(
                t.build_datasheet,
                caption_figures=False,
                regenerate_toc=effective,
                output_dir=artifact_dir,
            )
        except RuntimeError as exc:
            # datasheetindex raises here when regeneration was asked for and no
            # LLM client can be built. It raises rather than no-ops so that a
            # tool silently ignoring the parameter cannot invite an endless
            # retry -- but BOTH engines hand a tool exception back to the model
            # as text, so re-raising would hand over a traceback and produce the
            # very loop that guard exists to prevent. `[llm]` is an optional
            # extra, so no-credentials is an ordinary local and eval install,
            # not an outage. Matched on datasheetindex's own sentinel, so a
            # genuine build failure still propagates.
            if not (effective and _is_missing_llm_client(exc)):
                raise
            # The guard fires BEFORE _build_or_reuse, so nothing was built. Just
            # reporting the failure would leave the index absent -- and if this
            # was the model's first call (every system prompt says to call
            # build_datasheet first) every later tool would fail "call
            # build_datasheet first" while we had told it to carry on with an
            # outline it never received. Build normally, then say the repair was
            # unavailable, so one turn produces both.
            logger.info(
                "regenerate_toc requested but no LLM client is configured; building without it"
            )
            unavailable = True
            effective = False
            artifacts = await asyncio.to_thread(
                t.build_datasheet,
                caption_figures=False,
                regenerate_toc=False,
                output_dir=artifact_dir,
            )
        regenerated = regenerated or effective
        manifest: dict[str, Any] = t.get_artifact_manifest()
        toc_entries = []
        for node in manifest.get("toc", []):
            toc_entries.append(f"  p{node.get('page', '?')}: {node.get('title', '?')}")
        toc_text = "\n".join(toc_entries) if toc_entries else "(no ToC found)"
        # datasheetindex declares toc_quality as `TocQuality | None` (models.py),
        # and this is the FIRST tool the model calls on every large PDF -- an
        # unguarded .score here turns a missing quality assessment into an
        # AttributeError inside the tool, which the engines hand back to the
        # model as text and which then reads as a broken document rather than a
        # missing metric. Report it as unknown instead.
        toc_quality = artifacts.toc_quality
        quality_text = (
            f"{toc_quality.score:.2f}" if toc_quality is not None else "unknown"
        )
        # toc_quality scores the tree that came out, not who wrote it: a good
        # outline and a good LLM reconstruction score the same. The provenance is
        # the separate fact that tells the model whether a page number was READ
        # or INFERRED, so it is reported alongside and, when inferred, carries
        # the instruction with it -- a signal the agent never sees does no work.
        # .get() rather than [] for the same reason as the toc_quality guard
        # above: this is the first tool called on every large PDF, and a missing
        # key here would read to the model as a broken document.
        _log_corpus_shape(pdf_path, manifest)
        toc_source = str(manifest.get("toc_source", "unknown"))
        source_line = f"ToC source: {toc_source}"
        if toc_source == "llm_reconstructed":
            # Says the same thing as this tool's own docstring, deliberately and in
            # fewer words. They are not redundant: the docstring is the tool
            # DESCRIPTION, read once when the schema is built, while this is the
            # tool OUTPUT, read at the moment the model is choosing a page range.
            # They cannot be merged -- the docstring has to explain both values,
            # this line only ever fires for one -- so keep the instruction itself
            # consistent if either changes.
            # Until regenerate_toc existed, `llm_reconstructed` had exactly one
            # cause and the message could state it. It now has two, and getting
            # this wrong was observed live: forcing a rebuild on tcan1044a-q1,
            # whose own outline scores 0.82, reported "no usable bookmarks" --
            # false, and an invitation to treat a healthy document as malformed.
            # The caution itself is unchanged either way, because it follows from
            # the pages being inferred rather than from why they are.
            source_line += (
                " -- rebuilt from body text at your request; page numbers are inferred, not read"
                if effective
                else " -- no usable bookmarks; page numbers are inferred from body text"
            )
            source_line += (
                ". Confirm a section with search_text before reading a range from it."
            )
        # Same reasoning as the block above, for the other repairable case: the
        # docstring is read once at schema-build time, this is read at the moment
        # the agent is judging the outline. Deliberately narrow on both sides.
        #
        # Only for `pdf_outline`: `llm_reconstructed` IS the fallback's output, so
        # offering to rebuild it from body text proposes redoing the work that
        # produced it -- and since regenerate_toc joins the artifact-reuse key, the
        # second call returns the identical tree. A loop with a bill attached.
        #
        # Only below the threshold: datasheetindex auto-regenerates under 0.300, so
        # the band that needs a nudge is the one scoring too well to trip its gate
        # while still naming nothing useful. Advertising the repair on every build
        # would spend an LLM call and a full re-index on healthy documents.
        # A BAND, not a half-line. Below datasheetindex's own gate the library has
        # already run the fallback itself, so a still-`pdf_outline` source there
        # means it had no client or its candidate was rejected by the entry-count
        # or page-coverage guards -- hinting would invite the agent to buy that
        # same failure again.
        if (
            not effective
            and not unavailable
            and toc_source == "pdf_outline"
            and toc_quality is not None
            and _toc_fallback_threshold() <= toc_quality.score < _TOC_REPAIR_HINT_BELOW
        ):
            source_line += (
                f"\nThis outline scored {quality_text}. If its entries do not identify"
                " sections, call build_datasheet again with regenerate_toc=true to"
                " rebuild it from the body text."
            )
        if unavailable:
            source_line += (
                "\nToC regeneration is not available: this deployment has no LLM"
                " credentials configured for it. Do not request it again."
            )
        elif effective and toc_source != "llm_reconstructed":
            # `explicit_request` overrides only the score comparison. The
            # entry-count floor and the page-coverage guard still apply, and a
            # degenerate `Page 1..N` outline has perfect page coverage, so it is
            # the likely loser. Without this the agent pays an LLM call and a
            # full re-index, gets a byte-identical ToC, and has nothing telling
            # it the repair did not happen -- which reads as "ask again".
            source_line += (
                "\nRegeneration ran but did not replace the outline: the rebuilt"
                " contents failed datasheetindex's own quality guards, so the PDF's"
                " own outline was kept. Do not request it again; use search_text to"
                " locate sections instead."
            )
        return (
            f"Document loaded: {manifest.get('total_pages', '?')} pages\n"
            f"ToC quality: {quality_text}\n"
            f"{source_line}\n"
            f"Table of Contents:\n{toc_text}"
        )

    async def get_section_text(start_page: int, end_page: int) -> str:
        """Read extracted text for a page range (1-indexed, inclusive).

        Args:
            start_page: First page to read (1-indexed).
            end_page: Last page to read (1-indexed, inclusive).
        """
        return await asyncio.to_thread(
            _get_tools().get_section_text, start_page, end_page
        )

    async def search_text(
        query: str,
        case_sensitive: bool = False,
        max_results: int = 20,
    ) -> str:
        """Search for text across the entire document.

        Args:
            query: Text string to search for.
            case_sensitive: Whether the search is case-sensitive.
            max_results: Maximum number of matches to return.
        """
        matches = await asyncio.to_thread(
            _get_tools().search_text,
            query,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )
        if not matches:
            return "No matches found."
        lines = []
        for m in matches:
            lines.append(f"Page {m['page']}: ...{m['snippet']}...")
        return "\n".join(lines)

    async def extract_table_markdown(page: int) -> str:
        """Extract tables from a page as clean markdown.

        Args:
            page: The 1-indexed page number to extract tables from.
        """
        return await asyncio.to_thread(_get_tools().extract_table_markdown, page)

    async def inspect_page(page: int) -> list[dict[str, Any]] | list[Any] | str:
        """Visually inspect a page as an image (for diagrams or complex layouts).

        Args:
            page: The 1-indexed page number to inspect.

        Returns:
            Image content block for the rendered page. Render fidelity
            (and therefore vision-token cost) is configured per-
            deployment by the system; the agent cannot tune it per call.

        Notes:
            The system picks render fidelity from the model's context
            budget at session start (see `_make_large_pdf_tool_fns`'s
            `inspect_page_detail` arg). Agents that need a higher-
            fidelity look at a specific table or footnote should use
            `extract_table_markdown` (no vision cost) or `get_section_
            text` first; both are cheaper than re-rendering a page.
        """
        result = await asyncio.to_thread(
            _get_tools().inspect_page, page, detail=inspect_page_detail
        )
        images = [item for item in result if item.get("type") == "image"]
        if not images:
            return "No image content available for this page."

        if image_return == "binary":
            # pai (pydantic-ai) path. Imported HERE, not at module scope: this module
            # is on the lite engine's import path and pydantic-ai is an optional extra.
            from pydantic_ai import BinaryContent

            return [
                BinaryContent(
                    data=base64.b64decode(item["data"]),
                    media_type=item.get("mime_type", "image/png"),
                )
                for item in images
            ]

        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.get("mime_type", "image/png"),
                    "data": item["data"],
                },
            }
            for item in images
        ]

    tools_list = [
        build_datasheet,
        get_section_text,
        search_text,
        extract_table_markdown,
        inspect_page,
    ]
    return tools_list, _cleanup, _get_tools


def _make_large_pdf_tools(
    pdf_path: str,
    *,
    inspect_page_detail: InspectDetail = "high",
    tools_cache: LargePdfToolsCache | None = None,
) -> tuple[list, Callable[[], None], Callable[[], Any]]:
    """Anthropic-SDK-flavoured view of the large-PDF tools (the lite engine).

    Thin facade over :func:`_make_large_pdf_tool_fns`: same closures, wrapped in
    ``@beta_async_tool``. The split exists so the pai engine can register the very
    same functions with its own decorator -- the tool docstrings are the prompts the
    model reads, and a second copy of them would drift.
    """
    fns, cleanup, get_tools = _make_large_pdf_tool_fns(
        pdf_path,
        inspect_page_detail=inspect_page_detail,
        tools_cache=tools_cache,
    )
    return [beta_async_tool(f) for f in fns], cleanup, get_tools


# ---------------------------------------------------------------------------
# (3) Payload coercions (shared by lite and pai)
# ---------------------------------------------------------------------------


def _coerce_stringified_results(payload: dict[str, Any]) -> None:
    """Recover a ``results`` array that arrived as a JSON-encoded string.

    Gateways do not enforce the ``submit_extraction`` input schema consistently
    (see the module header). LiteLLM has been observed handing back ``results``
    as a JSON string rather than an array, which Pydantic rejects with
    ``Input should be a valid list [type=list_type]`` -- discarding a whole
    document's agentic extraction over a serialization quirk.

    Narrow by design: only a string that decodes to a *list* is replaced.
    Unparseable text and non-list JSON are left exactly as they arrived, so
    Pydantic raises its precise, field-named error rather than a vaguer one
    from here. Mutates ``payload`` in place.
    """
    raw = payload.get("results")
    if not isinstance(raw, str):
        return
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(decoded, list):
        return
    logger.warning(
        "Gateway returned a stringified `results` array (%d chars); recovered %d entries.",
        len(raw),
        len(decoded),
    )
    payload["results"] = decoded


def _normalize_payload(payload: dict[str, Any], pdf_source: str) -> None:
    """The single normalization applied before validating a submit payload.

    Runs in four places -- the lite engine's submit tool (to decide whether to
    ask the model for a repair) and ``_run_query_lite`` (to build the result),
    plus the pai engine's output validator (same repair decision) and
    ``extract_parameters_pai`` (same result-building step) -- so it must be
    idempotent; all three steps below are. Mutates ``payload`` in place.

    Order matters: drop_engine_authored_fields recurses through dicts and lists only, so
    while `results` is an opaque JSON string it cannot see inside it, and a model-fabricated
    source_location nested in results[] would survive the strip (source_location is a real
    ParameterResult field). Recover the array first.
    """
    # Defensive default: the schema makes pdf_source required, but if the model omits it
    # the input path is more informative than failing the whole extraction over a self-label.
    payload.setdefault("pdf_source", pdf_source)
    _coerce_stringified_results(payload)
    # Ingest guard: a model-supplied process_diagnostics must never win over the
    # engine-computed one (gateways enforce the submit schema inconsistently).
    drop_engine_authored_fields(payload)


# ---------------------------------------------------------------------------
# (4) Text-parse fallback (chamber benchmark's degraded-response channel)
# ---------------------------------------------------------------------------

# Default values for ParameterResult fields that don't accept None.
# Consumed by the chamber benchmark's text-parse normalizer
# (``anthropic_path._sanitize_claim_result_data``).
_PARAMETER_RESULT_DEFAULTS: dict[str, Any] = {
    "bool_value": "not_specified",
    "list_value": [],
    "text_value": "",
    "source_text": "",
    "reason": "",
    "original_terminology": "",
    "page_numbers": [],
    "values": [],
}


def _parse_json_from_text(text: str) -> dict[str, Any]:
    """Extract JSON from a text response, handling markdown fences.

    Falls back to a Python-literal recovery path: occasionally the model
    emits the payload as a pseudo-Python class call (e.g.,
    ``StructuredOutput(pdf_source="...", results=[...])`` with ``True``/
    ``False``/``None`` instead of JSON booleans). We unwrap and rewrite
    those tokens before retrying ``json.loads``. Without this, a single
    instruction-following slip costs three full retry turns.
    """
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown fences
    for marker in ("```json", "```"):
        idx = text.find(marker)
        if idx != -1:
            start = idx + len(marker)
            end = text.find("```", start)
            if end != -1:
                try:
                    return json.loads(text[start:end].strip())
                except (json.JSONDecodeError, ValueError):
                    pass

    # Recover from pseudo-Python class-call wrappers like
    # ``StructuredOutput(pdf_source=..., results=[...])`` -- strip the
    # wrapper, rewrite kwargs (``key=``) to JSON keys (``"key":``), and
    # lowercase Python literals.
    recovered = _recover_python_literal_payload(text)
    if recovered is not None:
        try:
            return json.loads(recovered)
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(
        "Could not parse response as JSON. Full response (%d chars): %s",
        len(text),
        text[:4000],
    )
    raise ValueError(
        f"Could not parse response as JSON. Response starts with: {text[:300]}"
    )


# Compiled patterns and frozen lookup table for the recovery helper. Kept
# at module scope (and immutable) so importers can't mutate the lookup --
# corrupting parsing for anything else that imports this module.
_CLASS_CALL_PREFIX = re.compile(r"\s*([A-Za-z_]\w*)\s*\(")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_PY_LITERAL_TOKENS: MappingProxyType[str, str] = MappingProxyType(
    {"True": "true", "False": "false", "None": "null"}
)


def _recover_python_literal_payload(text: str) -> str | None:
    """Best-effort rewrite of pseudo-Python output to JSON.

    Returns a JSON string if the text looks like a class-call wrapper
    (e.g. ``StructuredOutput(pdf_source="...", results=[...])``), else
    None. Keeps the rewrite narrow: strips a single leading ``Name(``
    and matching trailing ``)``, swaps top-level ``key=`` kwargs to
    ``"key":`` JSON pairs, and rewrites bare ``True``/``False``/``None``
    tokens to JSON literals.

    All rewrites are string- and depth-aware: substitutions never touch
    characters inside JSON string literals (so a ``source_text`` like
    ``"True voltage drop is None at 25C"`` survives intact), and only
    top-level kwargs are rewritten (nested dict keys in JSON form are
    left alone). Nested class-call wrappers (e.g.
    ``ParameterResult(found=True)`` inside ``results=[...]``) are *not*
    unwrapped -- if the model freelances that far, we retry instead.

    Single quotes and trailing commas are out of scope -- handling them
    would require committing to a Python-AST-style parser, and at that
    point the structural fix is to use a tool call (which the production
    path now does). This helper exists for the chamber benchmark's text
    channel where we still occasionally need to scrape a degraded
    response.
    """
    m = _CLASS_CALL_PREFIX.match(text)
    if not m or not text.rstrip().endswith(")"):
        return None

    inner = text[m.end() : text.rstrip().rfind(")")].strip()
    out: list[str] = []
    i = 0
    depth = 0
    in_str: str | None = None
    while i < len(inner):
        ch = inner[i]
        # Inside a string literal: copy verbatim, honor backslash escapes,
        # and never apply any rewrites here. This is the load-bearing
        # invariant -- substituting ``True``/``False``/``None`` over the
        # whole rewritten string would silently corrupt datasheet snippets.
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(inner):
                out.append(inner[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            i += 1
            continue
        if ch in "([{":
            depth += 1
            out.append(ch)
            i += 1
            continue
        if ch in ")]}":
            depth -= 1
            out.append(ch)
            i += 1
            continue
        # Out of string: try identifier-led rewrites. Match the longest
        # bare identifier here, then decide whether it's a Python literal,
        # a top-level kwarg, or a normal token to copy through.
        if ch.isalpha() or ch == "_":
            kw = _IDENT.match(inner, i)
            if kw is not None:
                ident = kw.group()
                end = kw.end()
                # Python literal -> JSON literal (regardless of depth, but
                # only outside string contexts -- which we already are).
                if ident in _PY_LITERAL_TOKENS:
                    out.append(_PY_LITERAL_TOKENS[ident])
                    i = end
                    continue
                # Top-level ``ident =`` (but not ``==``) -> ``"ident":``.
                if depth == 0:
                    j = end
                    while j < len(inner) and inner[j] == " ":
                        j += 1
                    if (
                        j < len(inner)
                        and inner[j] == "="
                        and (j + 1 >= len(inner) or inner[j + 1] != "=")
                    ):
                        out.append(f'"{ident}":')
                        i = j + 1
                        continue
                # Otherwise copy the identifier through verbatim.
                out.append(ident)
                i = end
                continue
        out.append(ch)
        i += 1

    return "{" + "".join(out) + "}"
