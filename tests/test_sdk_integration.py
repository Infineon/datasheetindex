"""Tests that exercise the real ``claude-agent-sdk`` tool server.

Skipped unless the optional ``sdk`` dependency group is installed
(``uv sync --group sdk``). It is not in ``dev`` because the wheel unpacks to
~263 MB, nearly all of it a bundled ``claude`` CLI binary these tests never
invoke -- see the group's comment in pyproject.toml.

Everything else in the suite drives ``create_datasheet_tools_server`` through a
fake ``create_sdk_mcp_server``. These are the only tests that would notice the
real SDK disagreeing with what our handlers emit -- which is exactly what #13
was: for two months ``inspect_page`` raised ``KeyError('mimeType')`` inside the
SDK's own content converter, and the whole suite stayed green because no test
ever ran that converter.
"""

import asyncio

import pymupdf
import pytest

from datasheetindex import create_datasheet_tools_server
from tests.conftest import sdk_envelope_to_content

pytest.importorskip("claude_agent_sdk")
pytest.importorskip("mcp")

pytestmark = [pytest.mark.sdk, pytest.mark.integration]


@pytest.fixture
def pdf_path(tmp_path):
    path = tmp_path / "probe.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    writer = pymupdf.TextWriter(page.rect)
    writer.append((72, 72), "Supply voltage 4.5V to 5.5V")
    writer.write_text(page)
    doc.save(str(path))
    doc.close()
    return path


def _call_tool(server, name, args):
    """Dispatch through the SDK server's real CallToolRequest handler."""
    from mcp import types

    handler = server.request_handlers[types.CallToolRequest]
    request = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name=name, arguments=args),
    )
    return asyncio.run(handler(request)).root


def test_inspect_page_returns_an_image_through_the_real_sdk(pdf_path, tmp_path):
    """The #13 regression, end to end against the shipped SDK surface.

    Before the fix this returned ``isError=True`` with the entire result text
    being ``'mimeType'`` -- no exception type, no tool name. The bare key name
    read like truncated output rather than an error, which is why it went
    untraced for so long.
    """
    server = create_datasheet_tools_server()["instance"]

    build = _call_tool(
        server,
        "build_datasheet",
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )
    assert build.isError is False

    result = _call_tool(server, "inspect_page", {"page": 1})

    assert result.isError is False, (
        f"inspect_page failed through the real SDK: "
        f"{[getattr(c, 'text', c) for c in result.content]}"
    )
    assert result.content[0].type == "image"
    assert result.content[0].mimeType == "image/png"
    assert result.content[0].data


def test_error_results_name_the_exception_type_through_the_real_sdk(pdf_path, tmp_path):
    """A handler-caught exception must reach the wire as 'TypeName: message'.

    Three error paths reach an agent through this surface and only the middle
    one is ours to shape:

    1. Schema validation, *before* dispatch -- the low-level MCP server rejects
       a call missing a required property and the handler never runs (asserted
       below, because it means a missing argument cannot demonstrate the prefix
       here; ``tests/test_defs.py`` covers that on the neutral surface, which
       non-validating hosts use directly).
    2. An exception inside a handler, caught by its blanket ``except`` and
       returned as an envelope. This is where the type prefix applies.
    3. An exception raised *after* the handler returns -- inside the SDK's own
       converter -- which escapes to the MCP low-level server and is
       stringified there with a bare ``str(e)`` we do not control. That is where
       the original ``'mimeType'`` text came from, and why fixing ``_err`` alone
       would never have made #13 legible.
    """
    server = create_datasheet_tools_server()["instance"]
    _call_tool(
        server,
        "build_datasheet",
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )

    # Path 1: rejected before our code runs, with a message we do not author.
    missing_arg = _call_tool(server, "get_section_text", {"start_page": 1})
    assert missing_arg.isError is True
    assert "required property" in missing_arg.content[0].text

    # Path 2: schema-valid, raises inside the handler, prefixed on the way out.
    out_of_range = _call_tool(server, "inspect_page", {"page": 9999})
    assert out_of_range.isError is True
    assert out_of_range.content[0].text.startswith("ValueError: ")
    assert "out of range" in out_of_range.content[0].text


def test_conftest_mirror_matches_the_real_sdk_converter(pdf_path, tmp_path):
    """Pin ``sdk_envelope_to_content`` against the converter it duplicates.

    The default lane relies on that mirror to catch key-name drift. If the SDK
    ever renames a field, this fails and tells us the mirror is stale --
    otherwise the mirror would keep asserting a contract the SDK no longer has.
    """
    server = create_datasheet_tools_server()["instance"]
    _call_tool(
        server,
        "build_datasheet",
        {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out")},
    )
    real = _call_tool(server, "inspect_page", {"page": 1})

    from datasheetindex.tools.defs import create_datasheet_tool_session

    session = create_datasheet_tool_session()
    handlers = {d.name: d for d in session.defs}
    try:
        asyncio.run(
            handlers["build_datasheet"].handler(
                {"pdf_source": str(pdf_path), "output_dir": str(tmp_path / "out2")}
            )
        )
        envelope = asyncio.run(handlers["inspect_page"].handler({"page": 1}))
    finally:
        session.close()

    mirrored = sdk_envelope_to_content(envelope)

    assert [block["type"] for block in mirrored] == [c.type for c in real.content]
    assert mirrored[0]["mimeType"] == real.content[0].mimeType
