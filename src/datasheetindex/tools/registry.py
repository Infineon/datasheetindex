"""Tool registration for Agent SDK / MCP.

The document-bound :class:`DatasheetTools` now lives in
:mod:`datasheetindex.tools.bound`; it is re-exported here for backward
compatibility (existing code imports it from this module). This module itself is
the thin Claude Agent SDK adapter over the framework-neutral tool defs.
"""

from __future__ import annotations

from datasheetindex._version import package_version
from datasheetindex.tools.bound import DatasheetTools
from datasheetindex.tools.defs import create_datasheet_tool_defs

__all__ = ["DatasheetTools", "create_datasheet_tools_server"]


def create_datasheet_tools_server():
    """Create the MCP/tool server that a consuming agent can mount.

    Requires ``claude-agent-sdk`` to be installed. Raises ``ImportError``
    if the SDK is not available. The server starts without a bound PDF;
    call ``build_datasheet`` with a ``pdf_source`` to load a document.

    This is a thin adapter over
    :func:`datasheetindex.tools.defs.create_datasheet_tool_defs`: it wraps each
    framework-neutral tool def with the SDK ``@tool`` decorator and hands them to
    ``create_sdk_mcp_server``. Tool names, descriptions, and JSON schemas are
    therefore identical to the neutral defs -- non-SDK hosts can realize the same
    tools without importing ``claude-agent-sdk`` (see ``create_datasheet_tool_defs``).
    """
    try:
        from claude_agent_sdk import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
            create_sdk_mcp_server,
            tool,
        )
    except ImportError:
        raise ImportError(
            "claude-agent-sdk is required for tool server creation. "
            "Install it with: uv pip install claude-agent-sdk"
        ) from None

    return create_sdk_mcp_server(
        name="datasheetindex",
        version=package_version(),
        tools=[
            tool(d.name, d.description, d.input_schema)(d.handler)
            for d in create_datasheet_tool_defs()
        ],
    )
