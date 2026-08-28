"""Tests for the framework-neutral datasheet tool definitions.

These exercise the tool handlers directly, without ``claude-agent-sdk`` -- the
whole point of ``create_datasheet_tool_defs`` is that a non-SDK host can realize
the tools without importing the SDK.
"""

import asyncio
import dataclasses
import inspect
import json
import sys

import pymupdf
import pytest

from datasheetindex.tools.defs import (
    DatasheetToolDef,
    DatasheetToolSession,
    create_datasheet_tool_defs,
    create_datasheet_tool_session,
)

EXPECTED_TOOL_NAMES = {
    "build_datasheet",
    "get_section_text",
    "search_text",
    "inspect_page",
    "extract_table_markdown",
}


def _make_pdf(path, text="Supply voltage 4.5V to 5.5V"):
    """Write a one-page PDF with a line of text at (72, 72)."""
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), text)
    writer.write_text(page)
    doc.save(str(path))
    doc.close()


def _defs_by_name():
    return {d.name: d for d in create_datasheet_tool_defs()}


def _run(handler, args):
    """Drive a neutral tool handler synchronously.

    The handler contract is ``Callable[[dict], Coroutine[..., dict]]``, so
    ``handler(args)`` is a coroutine that ``asyncio.run`` accepts directly.
    """
    return asyncio.run(handler(args))


def test_create_tool_defs_returns_expected_tools():
    defs = create_datasheet_tool_defs()

    assert {d.name for d in defs} == EXPECTED_TOOL_NAMES
    for d in defs:
        assert isinstance(d, DatasheetToolDef)
        assert isinstance(d.description, str) and d.description
        assert d.input_schema.get("type") == "object"
        assert inspect.iscoroutinefunction(d.handler)


def test_tool_def_is_frozen():
    d = create_datasheet_tool_defs()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.name = "renamed"  # ty: ignore[invalid-assignment]


def test_create_tool_defs_does_not_import_sdk():
    """Realizing the neutral defs must not pull in claude-agent-sdk."""
    sys.modules.pop("claude_agent_sdk", None)
    create_datasheet_tool_defs()
    assert "claude_agent_sdk" not in sys.modules


def test_query_handlers_require_build_first():
    """Query tools return an error envelope until build_datasheet has run."""
    defs = _defs_by_name()

    result = _run(defs["get_section_text"].handler, {"start_page": 1, "end_page": 1})
    assert result["is_error"] is True
    assert "No datasheet loaded" in result["content"][0]["text"]


def test_build_then_query_end_to_end(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path)
    defs = _defs_by_name()

    build_result = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )
    assert build_result["is_error"] is False

    section_result = _run(
        defs["get_section_text"].handler, {"start_page": 1, "end_page": 1}
    )
    assert section_result["is_error"] is False
    section_payload = json.loads(section_result["content"][0]["text"])
    assert "Supply voltage" in section_payload["text"]

    search_result = _run(defs["search_text"].handler, {"query": "5.5v"})
    assert search_result["is_error"] is False
    search_payload = json.loads(search_result["content"][0]["text"])
    assert search_payload["results"][0]["page"] == 1

    inspect_result = _run(defs["inspect_page"].handler, {"page": 1})
    assert inspect_result["is_error"] is False
    assert inspect_result["content"][0]["type"] == "image"

    # Drive extract_table_markdown too, so every tool handler is exercised here.
    # It needs the optional pymupdf4llm (layout extra); without it the handler
    # returns a clean error envelope rather than raising -- either way is_error
    # is a bool, which is what we assert.
    table_result = _run(defs["extract_table_markdown"].handler, {"page": 1})
    assert isinstance(table_result["is_error"], bool)


def test_inspect_page_image_block_carries_both_mime_spellings(tmp_path):
    """The image block must satisfy snake_case and camelCase readers alike.

    The neutral envelope is the Claude Agent SDK's envelope, and that format is
    mixed-case by construction: the SDK reads ``is_error`` (snake) but
    ``item["mimeType"]`` (camel). Emitting only ``mime_type`` made every
    inspect_page call through ``create_datasheet_tools_server`` raise
    ``KeyError('mimeType')`` (#13). Emitting only ``mimeType`` would break the
    other direction -- ``mcp_server._envelope_to_content`` and any host already
    reading the documented snake_case key. Both spellings, same value.
    """
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path)
    defs = _defs_by_name()
    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )

    result = _run(defs["inspect_page"].handler, {"page": 1})

    assert result["is_error"] is False
    block = result["content"][0]
    assert block["type"] == "image"
    assert block["mime_type"] == "image/png"
    assert block["mimeType"] == "image/png"
    assert block["data"]


def test_error_envelopes_name_the_exception_type(tmp_path):
    """A raised exception must be reported as 'TypeName: message', not bare.

    ``str(KeyError('end_page'))`` is just ``"'end_page'"``. When that is the
    entire text of a tool result it reads like truncated output rather than a
    failure, which is what made #13 take two months to trace. Every handler
    wraps its body in a blanket ``except Exception``, so any exception whose
    message is just its argument -- KeyError, IndexError -- degrades this way.
    """
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path)
    defs = _defs_by_name()
    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )

    # A required argument the model forgot to send: raises KeyError inside the
    # handler and is caught by its blanket except.
    missing_arg = _run(defs["get_section_text"].handler, {"start_page": 1})
    assert missing_arg["is_error"] is True
    assert missing_arg["content"][0]["text"] == "KeyError: 'end_page'"

    # The unbound-document guard raises RuntimeError with a real message; the
    # type prefix must not swallow it.
    unbound = _run(_defs_by_name()["search_text"].handler, {"query": "x"})
    assert unbound["is_error"] is True
    assert unbound["content"][0]["text"].startswith("RuntimeError: ")
    assert "No datasheet loaded" in unbound["content"][0]["text"]


def test_validation_errors_are_not_prefixed():
    """Messages the handler writes itself are already legible -- leave them alone.

    Only *exceptions* get a type prefix. ``_err`` is still the plain path, so a
    hand-written validation message does not become 'str: pdf_source is required'.
    """
    defs = _defs_by_name()
    result = _run(defs["build_datasheet"].handler, {})

    assert result["is_error"] is True
    assert result["content"][0]["text"] == "pdf_source is required"


def test_build_datasheet_rebinds_on_new_source(tmp_path):
    """A second build with a different source switches the active document."""
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    _make_pdf(pdf_b, text="Bravo marker two")
    defs = _defs_by_name()

    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_b), "output_dir": str(tmp_path / "out_b")},
    )

    alpha = _run(defs["search_text"].handler, {"query": "Alpha"})
    bravo = _run(defs["search_text"].handler, {"query": "Bravo"})

    assert json.loads(alpha["content"][0]["text"])["results"] == []
    assert json.loads(bravo["content"][0]["text"])["results"]


def test_failed_switch_preserves_working_document(tmp_path):
    """A failed switch to a bad source must leave the working document intact."""
    pdf_a = tmp_path / "a.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    defs = _defs_by_name()

    ok = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    assert ok["is_error"] is False

    # Switch to a source that cannot be opened -> the build must fail.
    bad = _run(
        defs["build_datasheet"].handler,
        {
            "pdf_source": str(tmp_path / "does_not_exist.pdf"),
            "output_dir": str(tmp_path / "out_b"),
        },
    )
    assert bad["is_error"] is True

    # ...but document A must still be bound and queryable, not closed.
    section = _run(defs["get_section_text"].handler, {"start_page": 1, "end_page": 1})
    assert section["is_error"] is False
    assert "Alpha marker" in json.loads(section["content"][0]["text"])["text"]


def test_failed_switch_closes_fresh_instance(tmp_path, monkeypatch):
    """A failed switch closes the fresh instance (no leak) and leaves A open."""
    import datasheetindex.tools.defs as defs_mod

    created: list = []
    real_tools = defs_mod.DatasheetTools

    class TrackingTools(real_tools):
        def __init__(self, pdf_path):
            super().__init__(pdf_path)
            self.close_calls = 0
            created.append(self)

        def close(self):
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(defs_mod, "DatasheetTools", TrackingTools)

    pdf_a = tmp_path / "a.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    defs = _defs_by_name()

    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    bad_source = str(tmp_path / "does_not_exist.pdf")
    bad = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": bad_source, "output_dir": str(tmp_path / "out_b")},
    )
    assert bad["is_error"] is True

    bad_instances = [t for t in created if t.pdf_path == bad_source]
    a_instances = [t for t in created if t.pdf_path == str(pdf_a)]
    # The fresh instance built for the bad source must have been closed...
    assert bad_instances and all(t.close_calls >= 1 for t in bad_instances)
    # ...and the still-bound document A must NOT have been closed.
    assert a_instances and all(t.close_calls == 0 for t in a_instances)


def test_successful_switch_survives_old_close_failure(tmp_path, monkeypatch):
    """A successful switch must bind the new document even if closing the old raises."""
    import datasheetindex.tools.defs as defs_mod

    real_tools = defs_mod.DatasheetTools
    fail_close_for: dict = {}

    class FlakyClose(real_tools):
        def close(self):
            if fail_close_for.get("path") == self.pdf_path:
                raise OSError("temp file vanished during cleanup")
            super().close()

    monkeypatch.setattr(defs_mod, "DatasheetTools", FlakyClose)

    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    _make_pdf(pdf_a, text="Alpha marker one")
    _make_pdf(pdf_b, text="Bravo marker two")
    defs = _defs_by_name()

    _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out_a")},
    )
    # Now make closing A blow up, then switch to B.
    fail_close_for["path"] = str(pdf_a)
    res = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(pdf_b), "output_dir": str(tmp_path / "out_b")},
    )
    # The switch succeeds and B is bound, despite A's close() raising.
    assert res["is_error"] is False
    bravo = _run(defs["search_text"].handler, {"query": "Bravo"})
    assert json.loads(bravo["content"][0]["text"])["results"]
    alpha = _run(defs["search_text"].handler, {"query": "Alpha"})
    assert json.loads(alpha["content"][0]["text"])["results"] == []


def test_build_datasheet_requires_pdf_source():
    defs = _defs_by_name()
    result = _run(defs["build_datasheet"].handler, {})
    assert result["is_error"] is True
    assert "pdf_source is required" in result["content"][0]["text"]


def test_each_defs_call_is_an_independent_session(tmp_path):
    """Two factory calls own separate state -- binding one must not affect the other."""
    pdf_path = tmp_path / "test.pdf"
    _make_pdf(pdf_path)

    session_a = {d.name: d for d in create_datasheet_tool_defs()}
    session_b = {d.name: d for d in create_datasheet_tool_defs()}

    _run(
        session_a["build_datasheet"].handler,
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )

    # session_b never built -> still unbound.
    result_b = _run(session_b["search_text"].handler, {"query": "voltage"})
    assert result_b["is_error"] is True
    assert "No datasheet loaded" in result_b["content"][0]["text"]


def test_create_tool_session_exposes_defs_and_close():
    session = create_datasheet_tool_session()
    assert isinstance(session, DatasheetToolSession)
    assert {d.name for d in session.defs} == EXPECTED_TOOL_NAMES
    assert all(isinstance(d, DatasheetToolDef) for d in session.defs)
    assert callable(session.close)


def test_create_datasheet_tool_defs_matches_session_defs():
    """The list factory stays backward-compatible: same six defs a session exposes."""
    defs = create_datasheet_tool_defs()
    assert {d.name for d in defs} == EXPECTED_TOOL_NAMES


def test_session_close_closes_bound_document(tmp_path, monkeypatch):
    """session.close() must close the currently bound DatasheetTools (temp cleanup)."""
    import datasheetindex.tools.defs as defs_mod

    created: list = []
    real_tools = defs_mod.DatasheetTools

    class TrackingTools(real_tools):
        def __init__(self, pdf_path):
            super().__init__(pdf_path)
            self.close_calls = 0
            created.append(self)

        def close(self):
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(defs_mod, "DatasheetTools", TrackingTools)

    pdf_a = tmp_path / "a.pdf"
    _make_pdf(pdf_a)
    session = create_datasheet_tool_session()
    handlers = {d.name: d for d in session.defs}

    _run(
        handlers["build_datasheet"].handler,
        {"pdf_source": str(pdf_a), "output_dir": str(tmp_path / "out")},
    )
    bound = created[-1]
    assert bound.close_calls == 0
    assert bound._doc is not None  # building opened the document

    session.close()
    assert bound.close_calls >= 1
    assert bound._doc is None  # the underlying document handle was released


def test_session_close_is_safe_when_unbound_and_idempotent():
    session = create_datasheet_tool_session()
    # No document ever bound -> must not raise; and calling twice is safe.
    session.close()
    session.close()


def test_get_section_text_description_mentions_the_continuation_note():
    from datasheetindex.tools.defs import create_datasheet_tool_defs

    defs = {d.name: d for d in create_datasheet_tool_defs()}
    description = defs["get_section_text"].description
    assert "=== NOTE:" in description
    assert "continued" in description
    # Both header forms, not just the plural one.
    assert "=== Page X of N ===" in description
    assert "=== Pages X-Y of N ===" in description
    # Content-level: this signal does not prove the content is a table.
    assert "cuts a table" not in description
    # The absence of a note guarantees nothing.
    assert "not a guarantee of completeness" in description


def test_manifest_hints_at_search_when_toc_is_empty(tmp_path):
    """A bookmark-less PDF must tell the agent how to navigate without a ToC.

    Roughly a third of real datasheets have no bookmarks, and without the [llm]
    extra there is no fallback. The other tools all still work -- but the tool
    descriptions assume a ToC exists, so the agent must be told to grep instead.
    """
    pdf = tmp_path / "no_toc.pdf"
    _make_pdf(pdf)
    defs = _defs_by_name()

    manifest = json.loads(
        _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})["content"][0][
            "text"
        ]
    )

    assert manifest["toc"] == []
    hint = manifest["hint"]
    assert "search_text" in hint
    assert "get_section_text" in hint


def test_manifest_has_no_hint_when_toc_is_present(tmp_path):
    """The good-ToC path must stay byte-identical -- the hint is degraded-only."""
    pdf = tmp_path / "with_toc.pdf"
    _make_pdf(pdf)
    doc = pymupdf.open(str(pdf))
    doc.set_toc([[1, "Absolute maximum ratings", 1]])
    doc.saveIncr()
    doc.close()
    defs = _defs_by_name()

    manifest = json.loads(
        _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})["content"][0][
            "text"
        ]
    )

    assert manifest["toc"] != []
    assert "hint" not in manifest


def test_build_datasheet_schema_bounds_max_figure_captions():
    defs = {d.name: d for d in create_datasheet_tool_defs()}
    props = defs["build_datasheet"].input_schema["properties"]

    assert props["caption_figures"]["type"] == "boolean"
    assert props["max_figure_captions"]["type"] == "integer"
    assert props["max_figure_captions"]["minimum"] == 0


def test_build_datasheet_schema_documents_the_real_default():
    """A bumped constant must not strand the agent-visible schema on 20."""
    from datasheetindex.llm.figure_captions import DEFAULT_MAX_FIGURE_CAPTIONS

    props = _defs_by_name()["build_datasheet"].input_schema["properties"]

    assert (
        f"default {DEFAULT_MAX_FIGURE_CAPTIONS}"
        in props["max_figure_captions"]["description"]
    )


def _figure_pdf(path):
    """A one-page PDF with a raster region and a text-layer figure caption."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (10, 20, 30))
    page.insert_image(pymupdf.Rect(50, 200, 545, 600), pixmap=pix)
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Figure 3. Functional block diagram")
    writer.write_text(page)
    doc.save(str(path))
    doc.close()


def test_manifest_carries_the_toc_source(tmp_path):
    """The agent is handed the manifest and nothing else.

    A reconstructed ToC has page numbers the model inferred from body text,
    and one read from the PDF's own outline does not. The agent has to weigh
    ``start_page`` differently in the two cases, and ``toc_quality`` cannot
    tell them apart -- a reconstruction that scores well looks identical.
    """
    pdf = tmp_path / "figs.pdf"
    _figure_pdf(pdf)
    defs = _defs_by_name()

    manifest = json.loads(
        _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})["content"][0][
            "text"
        ]
    )

    # This fixture has no bookmarks and no LLM client is configured under the
    # hermetic env, so there is nothing to reconstruct from either.
    assert manifest["toc_source"] == "none"


def test_manifest_carries_the_figure_digest(tmp_path):
    """The agent is handed the manifest and nothing else.

    Without a figure block in it, the consumer this branch was built for never
    learns the document has raster content: it could read ``json_path`` off
    disk, but nothing tells it to, and per the WSL namespace gotcha the
    server's filesystem may not be the agent's. This pins the exact shape the
    README documents.
    """
    pdf = tmp_path / "figs.pdf"
    _figure_pdf(pdf)
    defs = _defs_by_name()

    manifest = json.loads(
        _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})["content"][0][
            "text"
        ]
    )

    digest = manifest["figures"]
    assert set(digest) == {
        "total",
        "raster",
        "captioned",
        "pages_with_figures",
        "pages",
        "truncated",
    }
    # One raster placement plus one text-layer caption entry, both on page 1.
    assert digest["total"] == 2
    assert digest["raster"] == 1
    assert digest["captioned"] == 1
    assert digest["pages_with_figures"] == 1
    assert digest["truncated"] is False
    assert digest["pages"] == [
        {"page": 1, "figures": 2, "caption": "Figure 3. Functional block diagram"}
    ]
    # A digest, not a copy: no regions, bboxes or pixel dimensions here.
    assert "region" not in json.dumps(digest)


def test_manifest_figure_digest_is_coherent_without_figures(tmp_path):
    """Always present, so zero is distinguishable from "predates the feature"."""
    pdf = tmp_path / "plain.pdf"
    _make_pdf(pdf)
    defs = _defs_by_name()

    manifest = json.loads(
        _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})["content"][0][
            "text"
        ]
    )

    assert manifest["figures"] == {
        "total": 0,
        "raster": 0,
        "captioned": 0,
        "pages_with_figures": 0,
        "pages": [],
        "truncated": False,
    }


def test_figure_digest_is_bounded_by_constants_not_by_the_document():
    """A scanned datasheet must not put its whole figure index in every reply."""
    from datasheetindex.tools.bound import (
        _MANIFEST_CAPTION_CHARS,
        _MANIFEST_FIGURE_PAGES,
        _figure_digest,
    )

    figures = [
        {
            "page": page,
            "kind": "raster",
            "caption": "A very long caption. " * 40,
        }
        for page in range(1, 101)
        for _ in range(3)
    ]

    digest = _figure_digest(figures)

    rows = digest["pages"]
    assert isinstance(rows, list)
    assert digest["total"] == 300
    assert digest["pages_with_figures"] == 100
    assert len(rows) == _MANIFEST_FIGURE_PAGES
    assert digest["truncated"] is True
    listed_pages = []
    for row in rows:
        assert isinstance(row, dict)
        listed_pages.append(row.get("page"))
        caption = row.get("caption")
        assert isinstance(caption, str)
        assert len(caption) <= _MANIFEST_CAPTION_CHARS
        assert caption.endswith("..."), "a clipped caption must say so"
    # Ascending page order, and the first listed pages are the first pages.
    assert listed_pages == list(range(1, _MANIFEST_FIGURE_PAGES + 1))


def test_figure_digest_tolerates_a_malformed_or_absent_array():
    """An artifact is worth serving even when its figure index is not."""
    from datasheetindex.tools.bound import _figure_digest

    empty = {
        "total": 0,
        "raster": 0,
        "captioned": 0,
        "pages_with_figures": 0,
        "pages": [],
        "truncated": False,
    }

    assert _figure_digest(None) == empty
    assert _figure_digest("not an array") == empty
    assert _figure_digest([None, 7, {"kind": "raster"}, {"page": "three"}]) == empty


def test_figure_digest_picks_the_largest_area_caption_not_the_first():
    """Regression guard for the motivating failure.

    A TI product-change notice has a 7.5%-of-page product-label photo above
    a 25.5%-of-page "Product Attributes" table on the same page, in that
    document order. Picking the first captioned entry in array order names
    the photo and silently drops the table that actually answers a question
    like "does this document mention SUMITOMO" -- text search finds nothing
    because the table is pixels, so the digest is the only place an agent can
    learn it exists. Written so it fails against first-in-array-order
    selection: the smaller figure is listed first here on purpose.
    """
    from datasheetindex.tools.bound import _figure_digest

    photo_caption = "a photo of a product label"
    table_caption = (
        "a table titled Product Attributes with row labels including Mount "
        "Compound Supplier and Mold Compound Supplier"
    )
    figures = [
        {
            "page": 5,
            "kind": "raster",
            "page_area_pct": 7.5,
            "caption": photo_caption,
        },
        {
            "page": 5,
            "kind": "raster",
            "page_area_pct": 25.5,
            "caption": table_caption,
        },
    ]

    digest = _figure_digest(figures)

    assert digest["pages"] == [{"page": 5, "figures": 2, "caption": table_caption}]


def test_figure_digest_tie_break_is_deterministic():
    """Equal-area captioned entries must resolve the same way every call.

    The digest feeds an artifact that is fingerprinted for on-disk reuse, so
    a tie whose winner depended on dict or set iteration order would make the
    same build produce different bytes across runs.
    """
    from datasheetindex.tools.bound import _figure_digest

    figures = [
        {"page": 2, "kind": "raster", "page_area_pct": 10.0, "caption": "first"},
        {"page": 2, "kind": "raster", "page_area_pct": 10.0, "caption": "second"},
    ]

    results = [_figure_digest(figures) for _ in range(5)]

    assert all(result == results[0] for result in results)
    pages = results[0]["pages"]
    assert isinstance(pages, list)
    first_row = pages[0]
    assert isinstance(first_row, dict)
    assert first_row.get("caption") == "first"


def test_figure_digest_caption_clip_is_350_and_clips_at_350():
    """The clip bound is 350: a longer caption is truncated, a shorter is not."""
    from datasheetindex.tools.bound import _MANIFEST_CAPTION_CHARS, _figure_digest

    assert _MANIFEST_CAPTION_CHARS == 350

    long_caption = "a" * 400
    short_caption = "b" * 300
    figures = [
        {"page": 1, "kind": "raster", "caption": long_caption},
        {"page": 2, "kind": "raster", "caption": short_caption},
    ]

    digest = _figure_digest(figures)
    digest_pages = digest["pages"]
    assert isinstance(digest_pages, list)
    rows: dict[object, object] = {}
    for row in digest_pages:
        assert isinstance(row, dict)
        rows[row.get("page")] = row.get("caption")

    row_1 = rows[1]
    assert isinstance(row_1, str)
    assert len(row_1) == 350
    assert row_1 == long_caption[:347] + "..."
    assert rows[2] == short_caption


def test_build_datasheet_description_points_at_the_figure_digest():
    """An agent that is not told about the digest will not read it."""
    description = _defs_by_name()["build_datasheet"].description

    assert "figures" in description
    assert "inspect_page" in description
    assert "json_path" in description


def test_search_text_description_discloses_the_raster_blindness():
    """A zero-hit search must not read as "the document does not say this".

    The limitation is already on ``build_datasheet``, but that description is
    read once, pages of transcript before the search that trips over it. An
    agent deciding what a ``[]`` means looks at the tool it just called, so the
    caveat has to be on ``search_text`` itself -- along with the remedy, since
    naming a limitation without an action just stops the agent.
    """
    description = _defs_by_name()["search_text"].description

    assert "no text layer" in description
    # The remedy, not just the caveat: the two surfaces that do see pixels.
    assert "figures" in description
    assert "inspect_page" in description
    # The inference the agent has to make, spelled out.
    assert "absence of a match does not prove" in description


def test_empty_search_on_a_document_with_figures_returns_a_note(tmp_path):
    """The nudge has to arrive at the moment the search fails, not before."""
    pdf = tmp_path / "figs.pdf"
    _figure_pdf(pdf)
    defs = _defs_by_name()
    _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})

    payload = json.loads(
        _run(defs["search_text"].handler, {"query": "SUMITOMO"})["content"][0]["text"]
    )

    assert payload["results"] == []
    note = payload["note"]
    assert "figures" in note
    assert "inspect_page" in note


def test_search_note_is_absent_when_there_are_hits(tmp_path):
    """A note on every successful search is noise the agent has to read past."""
    pdf = tmp_path / "figs.pdf"
    _figure_pdf(pdf)
    defs = _defs_by_name()
    _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})

    payload = json.loads(
        _run(defs["search_text"].handler, {"query": "Figure"})["content"][0]["text"]
    )

    assert payload["results"]
    assert "note" not in payload


def test_search_note_is_absent_when_the_document_has_no_figures(tmp_path):
    """Pointing at a figure digest that is empty sends the agent nowhere."""
    pdf = tmp_path / "plain.pdf"
    _make_pdf(pdf)
    defs = _defs_by_name()
    _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})

    payload = json.loads(
        _run(defs["search_text"].handler, {"query": "SUMITOMO"})["content"][0]["text"]
    )

    assert payload["results"] == []
    assert "note" not in payload


def test_search_note_is_absent_for_a_caption_only_document(tmp_path):
    """A text-layer figure caption is searchable, so nothing is hidden.

    ``figures`` mixes two kinds. Keying the note off the array being non-empty
    would fire on a document whose only entries are captions the search already
    reads -- pointing the agent at pixels that do not exist.
    """
    pdf = tmp_path / "caption_only.pdf"
    _make_pdf(pdf, text="Figure 4. Timing diagram")
    defs = _defs_by_name()
    manifest = json.loads(
        _run(defs["build_datasheet"].handler, {"pdf_source": str(pdf)})["content"][0][
            "text"
        ]
    )
    assert manifest["figures"]["captioned"] == 1
    assert manifest["figures"]["raster"] == 0

    payload = json.loads(
        _run(defs["search_text"].handler, {"query": "SUMITOMO"})["content"][0]["text"]
    )

    assert payload["results"] == []
    assert "note" not in payload


def test_tool_descriptions_stay_within_a_budget():
    """Tool definitions are re-sent on every request, so length is a real cost.

    These budgets are not aspirational -- they are set just above what the
    current text needs, so a description drifting back toward an essay fails
    here rather than quietly taxing every turn. Raising one is allowed; doing
    it deliberately is the point. `build_datasheet` gets the largest budget
    because it is the only tool that has to explain the manifest it returns.

    Raised from 1300 in 0.31.0 for the ``toc_source`` sentence: the manifest
    gained a field, and a field the description does not explain is one the
    agent has to guess at.

    Raised from 1500 for the ``regenerate_toc`` nudge: the tool gained a
    parameter that only helps if the description tells the agent when to
    reach for it.

    Raised from 1800 in 0.36.0 for ``figure_captions_blocked``: the manifest
    gained a key, and the whole point of publishing it is that the agent stops
    asking for captions that cannot arrive -- which it can only do if the
    description says so. Same argument as ``toc_source`` above.
    """
    budgets = {
        "build_datasheet": 2150,
        "get_section_text": 800,
        "search_text": 700,
        "inspect_page": 400,
        "extract_table_markdown": 400,
    }
    oversize = {
        d.name: len(d.description)
        for d in create_datasheet_tool_defs()
        if len(d.description) > budgets[d.name]
    }
    assert not oversize, f"over budget: {oversize} (budgets {budgets})"


def test_every_tool_parameter_carries_a_description():
    """A parameter explained nowhere is one the agent has to guess at.

    Guidance about an argument belongs on the argument, not in the tool's
    prose: it stays attached to what it describes, and the description is
    free to answer only "when do I call this, and what comes back".
    """
    undocumented = [
        f"{d.name}.{param}"
        for d in create_datasheet_tool_defs()
        for param, schema in d.input_schema["properties"].items()
        if not schema.get("description")
    ]
    assert undocumented == []


def test_tool_descriptions_do_not_shout():
    """Claude 4.5/4.6 respond to emphasis by over-triggering, not by obeying.

    The published guidance is explicit: prompts written to stop an older model
    under-triggering ("CRITICAL: You MUST use this tool when...") now push the
    other way, and the fix is ordinary prose. These five markers are the ones
    this surface actually accumulated.
    """
    shouting = {"IMPORTANT", "CRITICAL", "MUST", "CALL THIS FIRST", "Do NOT"}
    found = {
        f"{d.name}: {marker}"
        for d in create_datasheet_tool_defs()
        for marker in shouting
        if marker in d.description
    }
    assert found == set()


def test_build_datasheet_handler_forwards_regenerate_toc(
    tmp_path, toc_pdf, monkeypatch
):
    """The MCP argument must reach ``DatasheetTools.build_datasheet``.

    A schema key and a Python parameter are two ends of a wire, and nothing in
    the structural tests looks at the wire itself: deleting the ``args.get``
    line from this handler left the parameter documented, accepted, validated
    -- and inert. ``toc_pdf`` is above the quality threshold and carries no
    figures, so the error the ``True`` case produces can only come from the
    request having arrived.
    """
    import datasheetindex.tools.defs as defs_mod

    seen: list[dict] = []
    real_tools = defs_mod.DatasheetTools

    class RecordingTools(real_tools):
        def build_datasheet(self, *args, **kwargs):
            seen.append(kwargs)
            return super().build_datasheet(*args, **kwargs)

    monkeypatch.setattr(defs_mod, "DatasheetTools", RecordingTools)
    defs = _defs_by_name()

    ok = _run(
        defs["build_datasheet"].handler,
        {"pdf_source": str(toc_pdf), "output_dir": str(tmp_path / "off")},
    )
    assert ok.get("is_error") is not True
    assert seen[-1].get("regenerate_toc") is False, (
        "omitting the argument must forward the documented default"
    )

    escalated = _run(
        defs["build_datasheet"].handler,
        {
            "pdf_source": str(toc_pdf),
            "output_dir": str(tmp_path / "on"),
            "regenerate_toc": True,
        },
    )
    assert seen[-1].get("regenerate_toc") is True
    # And it was acted on: credential-free, the only honest answer is a failure
    # naming the parameter. A handler that dropped the argument would build
    # happily and return a manifest here.
    assert escalated["is_error"] is True
    assert "regenerate_toc" in escalated["content"][0]["text"]


def test_zero_hit_note_does_not_steer_by_captions_that_cannot_exist():
    """The standing note tells the agent to read a caption. On a blocked build
    every caption is null and no rebuild will change that, so that advice sends
    it to look at nothing. The remedy survives; only the route changes."""
    from datasheetindex.tools.defs import (
        _EMPTY_SEARCH_BLOCKED_CAPTIONS_NOTE,
        _EMPTY_SEARCH_RASTER_NOTE,
    )

    assert "whose caption" in _EMPTY_SEARCH_RASTER_NOTE
    assert "whose caption" not in _EMPTY_SEARCH_BLOCKED_CAPTIONS_NOTE
    # Both must still name inspect_page: a limitation stated without a remedy
    # just stops the agent.
    assert "inspect_page" in _EMPTY_SEARCH_BLOCKED_CAPTIONS_NOTE
    assert "figure_captions_blocked" in _EMPTY_SEARCH_BLOCKED_CAPTIONS_NOTE
    # It must only name keys the digest rows actually carry. Rows are
    # {page, figures, caption}; `raster` is a document-level total only, so an
    # earlier draft telling the agent to rank by "per-page raster counts" sent
    # it to a field that does not exist.
    from datasheetindex.tools.bound import _figure_digest

    digest = _figure_digest(
        [{"page": 2, "kind": "raster", "caption": None, "page_area_pct": 9.0}]
    )
    pages = digest["pages"]
    assert isinstance(pages, list)
    row = pages[0]
    assert isinstance(row, dict)
    assert set(row) == {"page", "figures", "caption"}
    assert "raster count" not in _EMPTY_SEARCH_BLOCKED_CAPTIONS_NOTE
    assert "'pages'" in _EMPTY_SEARCH_BLOCKED_CAPTIONS_NOTE
