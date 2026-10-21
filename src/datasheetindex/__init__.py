"""Agent-first parameter extraction from technical datasheets.

Re-exports are resolved lazily via ``__getattr__`` (PEP 562), as in
``datasheetindex.tools``, so importing this package -- which every submodule
import does first -- does not import PyMuPDF. That is load-bearing for the
table scan: its worker processes resolve ``datasheetindex._scan_worker`` before
they may import PyMuPDF, and an eager import here would load the ONNX layout
engine in each of them (see that module). The quad-corrections reset that
``import datasheetindex`` used to imply now comes with ``datasheetindex.core``,
which every page-reading path imports.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasheetindex.batch import BatchResult, build_batch
    from datasheetindex.index import DatasheetIndex
    from datasheetindex.llm.client import (
        LlmIncompleteResponseError,
        LlmTlsVerificationError,
    )
    from datasheetindex.mcp_server import create_local_mcp_server, run_mcp_server
    from datasheetindex.tools import (
        DatasheetToolDef,
        DatasheetTools,
        DatasheetToolSession,
        create_datasheet_tool_defs,
        create_datasheet_tool_session,
        create_datasheet_tools_server,
    )

__all__ = [
    "BatchResult",
    "DatasheetIndex",
    "DatasheetToolDef",
    "DatasheetToolSession",
    "DatasheetTools",
    "LlmIncompleteResponseError",
    "LlmTlsVerificationError",
    "build_batch",
    "create_datasheet_tool_defs",
    "create_datasheet_tool_session",
    "create_datasheet_tools_server",
    "create_local_mcp_server",
    "run_mcp_server",
]

_ATTR_MODULES = {
    "BatchResult": "datasheetindex.batch",
    "build_batch": "datasheetindex.batch",
    "DatasheetIndex": "datasheetindex.index",
    # Exported because `build()` raises it: catching it must not require
    # reaching into `datasheetindex.llm.client`, whose neighbours are private.
    "LlmTlsVerificationError": "datasheetindex.llm.client",
    "LlmIncompleteResponseError": "datasheetindex.llm.client",
    "create_local_mcp_server": "datasheetindex.mcp_server",
    "run_mcp_server": "datasheetindex.mcp_server",
    # Delegated to the tools package's own lazy table, so each of these names
    # has one home to update when a tool class moves between its modules.
    "DatasheetTools": "datasheetindex.tools",
    "DatasheetToolDef": "datasheetindex.tools",
    "DatasheetToolSession": "datasheetindex.tools",
    "create_datasheet_tool_defs": "datasheetindex.tools",
    "create_datasheet_tool_session": "datasheetindex.tools",
    "create_datasheet_tools_server": "datasheetindex.tools",
}


def __dir__() -> list[str]:
    """Keep ``dir()`` listing the re-exports, absent from ``globals()`` until used."""
    return sorted(__all__)


def __getattr__(name: str) -> object:
    module_name = _ATTR_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
