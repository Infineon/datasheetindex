"""Tests for tool registry."""

import sys
import types
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from datasheetindex.core import engine
from datasheetindex.tools.registry import (
    DatasheetTools,
    create_datasheet_tools_server,
)

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


def test_datasheet_tools_inspect_page(tmp_path):
    """DatasheetTools.inspect_page should work with a valid PDF."""
    # Create a minimal test PDF
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Registry test")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    result = tools.inspect_page(page=1)
    tools.close()

    assert len(result) == 1
    assert result[0]["type"] == "image"


def test_datasheet_tools_build_and_query_artifacts(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Supply voltage 4.5V to 5.5V")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    output_dir = tmp_path / "out"
    tools = DatasheetTools(str(pdf_path))
    artifacts = tools.build_datasheet(output_dir=str(output_dir))

    section_text = tools.get_section_text(1, 1)
    matches = tools.search_text("5.5v")
    tools.close()

    assert artifacts.json_path is not None
    assert artifacts.text_path is not None
    assert section_text.startswith("=== Page 1 of 1 ===")
    assert "--- PAGE 1 ---" in section_text
    assert "Supply voltage" in section_text
    # No ToC in this synthetic PDF, so no breadcrumb is attached.
    assert matches == [
        {
            "page": 1,
            "start": 23,
            "end": 27,
            "snippet": "Supply voltage 4.5V to 5.5V",
        }
    ]


def test_get_section_text_multi_page_header(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))
    section_text = tools.get_section_text(1, 2)
    tools.close()

    assert section_text.startswith("=== Pages 1-2 of 3 ===")


def test_search_text_attaches_breadcrumb_and_multi_pattern(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    for label in ("Absolute maximum ratings here", "Thermal resistance value here"):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, 72), label)
        writer.write_text(page)
    doc.set_toc(
        [
            [1, "5 Electrical Characteristics", 1],
            [2, "5.1 Absolute Maximum Ratings", 1],
            [1, "6 Thermal", 2],
        ]
    )
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))

    single = tools.search_text("maximum ratings")
    multi = tools.search_text(["maximum ratings", "Thermal resistance"])
    tools.close()

    assert len(single) == 1
    assert single[0]["page"] == 1
    assert single[0]["breadcrumb"] == (
        "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
    )

    by_page = {m["page"]: m for m in multi}
    assert by_page[1]["pattern"] == "maximum ratings"
    assert by_page[1]["breadcrumb"] == (
        "5 Electrical Characteristics > 5.1 Absolute Maximum Ratings"
    )
    assert by_page[2]["pattern"] == "Thermal resistance"
    assert by_page[2]["breadcrumb"] == "6 Thermal"


def test_search_text_resolves_breadcrumb_once_per_page(tmp_path, monkeypatch):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    # Three matches for "voltage", all on the same page.
    writer.append((72, 72), "voltage here, voltage there, voltage everywhere")
    writer.write_text(page)
    doc.set_toc([[1, "1 Supply", 1]])
    doc.save(str(pdf_path))
    doc.close()

    import datasheetindex.tools.bound as bound_module

    calls: list[int] = []
    real = bound_module.find_breadcrumb_for_page

    def counting(toc, page_number):
        calls.append(page_number)
        return real(toc, page_number)

    monkeypatch.setattr(bound_module, "find_breadcrumb_for_page", counting)

    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))
    matches = tools.search_text("voltage")
    tools.close()

    assert len(matches) == 3
    assert all(m["breadcrumb"] == "1 Supply" for m in matches)
    # The ToC is walked once for the single distinct page, not once per match.
    assert calls == [1]


def test_build_datasheet_omitted_output_dir_uses_resolver(monkeypatch, tmp_path):
    """DatasheetTools.build_datasheet(output_dir=None) writes to resolver default."""
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Hello")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    pinned = tmp_path / "env-pinned"
    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(pinned))

    tools = DatasheetTools(str(pdf_path))
    try:
        artifacts = tools.build_datasheet()
    finally:
        tools.close()

    assert artifacts.json_path is not None
    assert artifacts.json_path.parent == pinned


def test_build_datasheet_cache_invalidated_when_resolver_changes(monkeypatch, tmp_path):
    """Cache must miss if env var (and thus resolver default) changed between calls."""
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Hello")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    first = tmp_path / "first"
    second = tmp_path / "second"

    tools = DatasheetTools(str(pdf_path))
    try:
        monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(first))
        a1 = tools.build_datasheet()
        monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(second))
        a2 = tools.build_datasheet()
    finally:
        tools.close()

    assert a1.json_path is not None and a1.json_path.parent == first
    assert a2.json_path is not None and a2.json_path.parent == second


def test_datasheet_tools_artifact_queries_require_build(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    with pytest.raises(RuntimeError, match="build_datasheet"):
        tools.get_section_text(1, 1)
    with pytest.raises(RuntimeError, match="build_datasheet"):
        tools.search_text("foo")
    tools.close()


def test_datasheet_tools_lazy_doc(tmp_path):
    """Document should be lazy-opened."""
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    assert tools._doc is None
    _ = tools.doc
    assert tools._doc is not None
    tools.close()
    assert tools._doc is None


def test_create_server_raises_without_sdk(monkeypatch):
    """create_datasheet_tools_server should raise ImportError without SDK.

    Force the import to fail rather than relying on claude-agent-sdk being
    absent from the environment. That precondition held implicitly until the
    optional `sdk` group was added, at which point this test asserted the
    library's behaviour in one lane and the environment's in the other -- it
    passed under a plain `uv sync` and failed under `uv sync --group sdk`,
    despite nothing about the code under test having changed. A None entry in
    sys.modules makes `import claude_agent_sdk` raise ImportError either way.
    """
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)

    with pytest.raises(ImportError, match="claude-agent-sdk"):
        create_datasheet_tools_server()


def test_sdk_server_wraps_neutral_defs_verbatim(monkeypatch):
    """The SDK server must expose exactly the neutral defs' names/descriptions/schemas.

    This locks the two surfaces together: the SDK adapter is a thin wrapper over
    create_datasheet_tool_defs(), so a non-SDK host and an SDK host see identical
    tool metadata.
    """
    from datasheetindex.tools.defs import create_datasheet_tool_defs

    captured: list[tuple] = []

    def fake_tool(name, description, params):
        def decorator(func):
            captured.append((name, description, params))
            func._tool_name = name
            return func

        return decorator

    def fake_create_sdk_mcp_server(name, version, tools):
        return types.SimpleNamespace(name=name, version=version, tools=tools)

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(
            tool=fake_tool, create_sdk_mcp_server=fake_create_sdk_mcp_server
        ),
    )

    create_datasheet_tools_server()

    expected = [
        (d.name, d.description, d.input_schema) for d in create_datasheet_tool_defs()
    ]
    assert captured == expected


def test_create_server_registers_tools(monkeypatch, tmp_path):
    """Server factory should register 6 agent-ready tools via SDK pattern."""
    import asyncio

    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Registry MCP test")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    def fake_tool(name, description, params):
        def decorator(func):
            func._tool_name = name
            return func

        return decorator

    def fake_create_sdk_mcp_server(name, version, tools):
        return types.SimpleNamespace(
            name=name,
            version=version,
            tools={t._tool_name: t for t in tools},
        )

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(
            tool=fake_tool,
            create_sdk_mcp_server=fake_create_sdk_mcp_server,
        ),
    )

    server = create_datasheet_tools_server()

    assert set(server.tools) == {
        "build_datasheet",
        "get_section_text",
        "search_text",
        "inspect_page",
        "extract_table_markdown",
    }

    build_result = asyncio.run(
        server.tools["build_datasheet"](
            {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")}
        )
    )
    assert build_result["is_error"] is False

    section_result = asyncio.run(
        server.tools["get_section_text"]({"start_page": 1, "end_page": 1})
    )
    assert section_result["is_error"] is False

    search_result = asyncio.run(server.tools["search_text"]({"query": "registry"}))
    assert search_result["is_error"] is False

    # The SDK tool accepts a list query (multi-pattern) just like a string.
    multi_search_result = asyncio.run(
        server.tools["search_text"]({"query": ["registry", "test"]})
    )
    assert multi_search_result["is_error"] is False

    inspect_result = asyncio.run(server.tools["inspect_page"]({"page": 1}))
    assert inspect_result["is_error"] is False

    # extract_table_markdown requires pymupdf4llm; verify graceful error
    table_md_result = asyncio.run(server.tools["extract_table_markdown"]({"page": 1}))
    # Will be is_error=True if pymupdf4llm not installed, False if it is
    assert isinstance(table_md_result["is_error"], bool)


def test_mcp_build_datasheet_omits_output_dir(monkeypatch, tmp_path):
    """When the MCP caller omits output_dir, the library default is used."""
    import asyncio

    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Default output_dir test")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    def fake_tool(name, description, params):
        def decorator(func):
            func._tool_name = name
            return func

        return decorator

    def fake_create_sdk_mcp_server(name, version, tools):
        return types.SimpleNamespace(
            name=name, version=version, tools={t._tool_name: t for t in tools}
        )

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(
            tool=fake_tool, create_sdk_mcp_server=fake_create_sdk_mcp_server
        ),
    )
    # Pin the resolver so the test stays hermetic
    pinned = tmp_path / "resolved-out"
    monkeypatch.setenv("DATASHEETINDEX_OUTPUT_DIR", str(pinned))

    server = create_datasheet_tools_server()
    result = asyncio.run(server.tools["build_datasheet"]({"pdf_source": str(pdf_path)}))
    assert result["is_error"] is False
    assert pinned.exists() and any(pinned.iterdir())


@pytest.mark.real_pdf
def test_real_pdf_tools():
    """DatasheetTools should work with the real test PDF."""
    if not TLE9350_PATH.exists():
        pytest.skip("Test PDF not found")

    tools = DatasheetTools(str(TLE9350_PATH))
    result = tools.inspect_page(page=1)
    tools.close()

    assert result[0]["type"] == "image"
    assert len(result[0]["data"]) > 0


def test_datasheet_tools_supports_url_source(monkeypatch):
    from tests.conftest import DummyDoc, FakeResponse

    opened_paths: list[str] = []

    def fake_urlopen(url: str, timeout: int):
        assert url == "https://example.com/test.pdf"
        assert timeout > 0
        return FakeResponse(b"%PDF-1.7\nmock")

    def fake_open(path: str):
        opened_paths.append(path)
        return DummyDoc()

    monkeypatch.setattr("datasheetindex.index.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("datasheetindex.index.pymupdf.open", fake_open)

    tools = DatasheetTools("https://example.com/test.pdf")
    _ = tools.doc
    assert len(opened_paths) == 1
    temp_path = Path(opened_paths[0])
    assert temp_path.exists()

    tools.close()
    assert tools._doc is None
    assert not temp_path.exists()


def test_datasheet_tools_locate_text_without_build(tmp_path):
    pdf_path = tmp_path / "locate.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Hello world")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    results = tools.locate_text("Hello")  # no build_datasheet first
    tools.close()

    assert len(results) == 1
    assert results[0]["page"] == 1
    assert results[0]["match_method"] == "search_for"


def test_datasheet_tools_reexported_for_backward_compat():
    """DatasheetTools now lives in tools.bound but must stay importable everywhere."""
    from datasheetindex import DatasheetTools as top_level
    from datasheetindex.tools import DatasheetTools as pkg
    from datasheetindex.tools.bound import DatasheetTools as canonical
    from datasheetindex.tools.registry import DatasheetTools as via_registry

    # Every historical import path must resolve to the one canonical class.
    assert canonical is via_registry is pkg is top_level


@pytest.mark.parametrize(
    "module",
    [
        "datasheetindex",
        "datasheetindex.tools",
        "datasheetindex.tools.bound",
        "datasheetindex.tools.defs",
        "datasheetindex.tools.registry",
    ],
)
def test_module_cold_imports_cleanly(module):
    """Each module imports cleanly as the entry point of a fresh interpreter.

    Catches a cycle that raises during import; the ``timeout`` turns a cycle
    that *deadlocks* (rather than raising) into a clean failure instead of a
    hung subprocess.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_neutral_tool_modules_do_not_import_registry():
    """Lock the layering: the neutral tool modules must not depend on the SDK adapter.

    A cold-import test only catches a reintroduced ``bound``/``defs`` -> ``registry``
    dependency if it happens to *raise*; a benign one (late import, function-local,
    or under ``TYPE_CHECKING``) would import cleanly and silently invert the layers
    again. Parsing the source for an actual import statement locks the invariant
    directly, regardless of load order or package-init side effects.
    """
    import ast
    from pathlib import Path

    import datasheetindex.tools.bound as bound_mod
    import datasheetindex.tools.defs as defs_mod

    for module in (bound_mod, defs_mod):
        source = Path(module.__file__).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        offending = {name for name in imported if "tools.registry" in name}
        assert not offending, (
            f"{module.__name__} must not import the SDK adapter registry "
            f"(found: {sorted(offending)}) -- it would reintroduce the layer inversion"
        )


def _continued_table_pdf(tmp_path):
    """Two pages; page 2 opens with a continuation marker."""
    pdf_path = tmp_path / "continued.pdf"
    doc = pymupdf.open()
    first = doc.new_page()
    writer = pymupdf.TextWriter(first.rect)
    writer.append((72, 72), "6.4 Recommended Operating Conditions")
    writer.append((72, 96), "IOH(RXD) Devices with VIO -1.5 mA")
    writer.write_text(first)

    second = doc.new_page()
    writer = pymupdf.TextWriter(second.rect)
    writer.append((72, 72), "6.4 Recommended Operating Conditions (continued)")
    writer.append((72, 96), "TJ Operating junction temperature -40")
    writer.write_text(second)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_get_section_text_warns_when_range_cuts_a_continuation(tmp_path):
    pdf_path = _continued_table_pdf(tmp_path)
    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))
    section_text = tools.get_section_text(1, 1)
    tools.close()

    # Header form is unchanged.
    assert section_text.startswith("=== Page 1 of 2 ===")
    expected = (
        '=== NOTE: "6.4 Recommended Operating Conditions" is continued on '
        "page 2, which is outside this range. ==="
    )
    assert expected in section_text
    # A page-1 read has no head cut.
    assert "opens inside" not in section_text
    # No completeness claim, ever.
    assert "complete" not in section_text.lower()


def test_get_section_text_warns_when_range_opens_mid_continuation(tmp_path):
    pdf_path = _continued_table_pdf(tmp_path)
    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))
    section_text = tools.get_section_text(2, 2)
    tools.close()

    assert section_text.startswith("=== Page 2 of 2 ===")
    expected = (
        '=== NOTE: this range opens inside "6.4 Recommended Operating '
        'Conditions", which is continued from page 1. ==='
    )
    assert expected in section_text
    # Page 2 is the last page, so there is no tail cut.
    assert "is continued on page" not in section_text


def test_get_section_text_is_silent_when_the_range_covers_the_whole_table(tmp_path):
    pdf_path = _continued_table_pdf(tmp_path)
    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))
    section_text = tools.get_section_text(1, 2)
    tools.close()

    assert section_text.startswith("=== Pages 1-2 of 2 ===")
    assert "=== NOTE:" not in section_text


def _spanning_table_pdf(tmp_path):
    """Three pages; the table runs across both breaks, so page 2 is cut at BOTH
    boundaries -- it opens mid-continuation and continues onward."""
    pdf_path = tmp_path / "spanning.pdf"
    doc = pymupdf.open()
    rows = [
        "6.8 Electrical Characteristics",
        "6.8 Electrical Characteristics (continued)",
        "6.8 Electrical Characteristics (continued)",
    ]
    for heading in rows:
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        writer.append((72, 72), heading)
        writer.append((72, 96), "VOH output high voltage 2.4 V")
        writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_get_section_text_warns_at_both_boundaries(tmp_path):
    """A range cut at the head AND the tail gets exactly one note for each,
    head first, with no duplication."""
    pdf_path = _spanning_table_pdf(tmp_path)
    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))
    section_text = tools.get_section_text(2, 2)
    tools.close()

    lines = section_text.splitlines()
    notes = [line for line in lines if line.startswith("=== NOTE:")]
    assert len(notes) == 2, notes

    # The head note comes first -- it describes where the range began.
    assert notes[0] == (
        '=== NOTE: this range opens inside "6.8 Electrical Characteristics", '
        "which is continued from page 1. ==="
    )
    assert notes[1] == (
        '=== NOTE: "6.8 Electrical Characteristics" is continued on page 3, '
        "which is outside this range. ==="
    )
    # Notes sit directly under the header, above the page text.
    assert lines[0] == "=== Page 2 of 3 ==="
    assert lines[1].startswith("=== NOTE:")


def test_document_note_lines_are_not_mistaken_for_the_signal(tmp_path):
    """A datasheet's own body text can contain a literal "NOTE:" line that has
    nothing to do with our truncation signal -- found live on page 26 of the
    TI TCAN1044A-Q1 revision-history section. Our warning must be framed as
    "=== NOTE: ... ===" specifically so it cannot be confused with, or
    falsely triggered by, that kind of document content."""
    pdf_path = tmp_path / "document_note.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "8 Revision History")
    writer.append(
        (72, 96),
        "NOTE: Page numbers for previous revisions may differ from page "
        "numbers in the current version.",
    )
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    tools = DatasheetTools(str(pdf_path))
    tools.build_datasheet(output_dir=str(tmp_path / "out"))
    section_text = tools.get_section_text(1, 1)
    tools.close()

    # The library stays silent -- no continuation marker exists anywhere.
    assert not any(line.startswith("=== NOTE:") for line in section_text.splitlines())
    # The document's own NOTE: line is still there, verbatim -- we frame our
    # signal, we do not filter the document's content.
    assert (
        "NOTE: Page numbers for previous revisions may differ from page "
        "numbers in the current version." in section_text
    )


def test_sdk_server_reports_installed_version(monkeypatch):
    """The SDK surface must report the same real version as the local server.

    Both surfaces exist so there is a single source of truth for the tool
    definitions; one of them reporting a stale literal breaks that property.
    """
    from datasheetindex._version import package_version

    def fake_tool(name, description, params):
        def decorator(func):
            return func

        return decorator

    def fake_create_sdk_mcp_server(name, version, tools):
        return types.SimpleNamespace(name=name, version=version, tools=tools)

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(
            tool=fake_tool, create_sdk_mcp_server=fake_create_sdk_mcp_server
        ),
    )

    server = create_datasheet_tools_server()

    assert server.version == package_version()
    assert server.version != "1.0.0"


def test_sdk_content_conversion_survives_every_tool(monkeypatch, tmp_path):
    """Drive each tool's envelope through the SDK's real conversion rules.

    The other fakes in this file accept the envelope and never read a key out of
    it, so they cannot see a key-name mismatch between what a handler emits and
    what the SDK reads. That is precisely the blind spot #13 lived in: for two
    months inspect_page raised KeyError('mimeType') inside the SDK's converter
    on every call, and this file stayed green.

    `sdk_envelope_to_content` mirrors that converter key for key (see its
    docstring); tests/test_sdk_integration.py pins the mirror against the real
    SDK when the optional 'sdk' group is installed.
    """
    import asyncio

    from tests.conftest import sdk_envelope_to_content

    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Supply voltage 4.5V to 5.5V")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()

    def fake_tool(name, description, params):
        def decorator(func):
            func._tool_name = name
            return func

        return decorator

    def fake_create_sdk_mcp_server(name, version, tools):
        return types.SimpleNamespace(
            name=name, version=version, tools={t._tool_name: t for t in tools}
        )

    monkeypatch.setitem(
        sys.modules,
        "claude_agent_sdk",
        types.SimpleNamespace(
            tool=fake_tool, create_sdk_mcp_server=fake_create_sdk_mcp_server
        ),
    )

    server = create_datasheet_tools_server()

    def call(name, args):
        # Convert exactly as the SDK does -- a mismatched key raises KeyError
        # here rather than silently passing.
        envelope = asyncio.run(server.tools[name](args))
        return envelope, sdk_envelope_to_content(envelope)

    build_envelope, build_content = call(
        "build_datasheet",
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )
    assert build_envelope["is_error"] is False
    assert build_content[0]["type"] == "text"

    image_envelope, image_content = call("inspect_page", {"page": 1})
    assert image_envelope["is_error"] is False
    assert image_content[0]["type"] == "image"
    assert image_content[0]["mimeType"] == "image/png"
    assert image_content[0]["data"]

    # Error envelopes convert too -- they are text blocks, and a handler that
    # raised must still produce something the SDK can render.
    error_envelope, error_content = call("get_section_text", {"start_page": 1})
    assert error_envelope["is_error"] is True
    assert error_content[0]["text"] == "KeyError: 'end_page'"

    for name, args in (
        ("search_text", {"query": "voltage"}),
        ("extract_table_markdown", {"page": 1}),
    ):
        envelope, content = call(name, args)
        assert isinstance(envelope["is_error"], bool)
        assert content and content[0]["type"] == "text"


def _tools_on_a_one_page_pdf(tmp_path: Path) -> "DatasheetTools":
    pdf_path = tmp_path / "kwargs.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Layout kwarg gating")
    writer.write_text(page)
    doc.save(str(pdf_path))
    doc.close()
    return DatasheetTools(str(pdf_path))


def _fake_pymupdf4llm(*, use_layout: bool, seen: dict) -> Any:
    """A stand-in whose to_markdown records the keywords it was handed."""
    module: Any = types.ModuleType("pymupdf4llm")
    module._use_layout = use_layout
    module.use_layout = lambda yes: None

    def to_markdown(_doc, **kwargs):
        seen.update(kwargs)
        return "md"

    module.to_markdown = to_markdown
    return module


@pytest.mark.parametrize(
    ("use_layout", "expect_furniture"),
    [(True, True), (False, False)],
)
def test_extract_table_markdown_passes_layout_kwargs_only_on_the_layout_branch(
    tmp_path, monkeypatch, use_layout, expect_furniture
):
    """`header=`/`footer=` must not reach the classic renderer.

    `pymupdf4llm.to_markdown` dispatches on the module global `_use_layout` at
    call time, so importing the package successfully does not mean the layout
    branch is taken: a pymupdf4llm whose own `import pymupdf.layout` failed
    (an unimportable onnxruntime is enough) routes to
    `helpers/pymupdf_rag.py`, which swallows unknown keywords into `**kwargs`
    and `print()`s a notice naming them to *stdout* -- the channel the MCP
    stdio transport carries JSON-RPC on. `mcp_server`'s existing
    `_silence_pymupdf_stdout_notice` does not cover it; it silences PyMuPDF's
    own advertisement, a different print in a different package.

    Runs on the default lane, which is the point: tests/test_layout_integration.py
    skips without the [layout] extra, i.e. in exactly the environment where the
    classic branch is taken.
    """
    seen: dict = {}
    module = _fake_pymupdf4llm(use_layout=use_layout, seen=seen)
    monkeypatch.setattr(
        engine, "importlib", types.SimpleNamespace(import_module=lambda name: module)
    )
    tools = _tools_on_a_one_page_pdf(tmp_path)
    try:
        assert tools.extract_table_markdown(1) == "md"
    finally:
        tools.close()

    assert ("header" in seen) is expect_furniture, seen
    assert ("footer" in seen) is expect_furniture, seen
    if expect_furniture:
        assert seen["header"] is False and seen["footer"] is False, seen
    # The keywords the classic renderer does accept are passed either way.
    assert seen["pages"] == [0] and seen["show_progress"] is False, seen
