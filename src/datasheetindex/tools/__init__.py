"""Tooling surfaces for bound datasheet inspection and MCP handoff.

Re-exports are resolved lazily via ``__getattr__`` (PEP 562) rather than
eager top-level imports. Eager imports here would make ``datasheetindex.tools``
un-importable as a mere namespace: any submodule import (e.g.
``datasheetindex.tools.vision``) first runs this package's ``__init__.py``,
and an eager ``from datasheetindex.tools.bound import DatasheetTools`` pulls in
``datasheetindex.index`` -- which is exactly the module that reaches
``datasheetindex.tools.vision`` (via ``llm/figure_captions.py``) while it is
still initializing. Lazy resolution means a plain submodule import never
touches ``bound.py`` at all, so the cycle cannot form.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasheetindex.tools.bound import DatasheetTools
    from datasheetindex.tools.defs import (
        DatasheetToolDef,
        DatasheetToolSession,
        create_datasheet_tool_defs,
        create_datasheet_tool_session,
    )
    from datasheetindex.tools.registry import create_datasheet_tools_server
    from datasheetindex.tools.vision import inspect_page

__all__ = [
    "DatasheetToolDef",
    "DatasheetToolSession",
    "DatasheetTools",
    "create_datasheet_tool_defs",
    "create_datasheet_tool_session",
    "create_datasheet_tools_server",
    "inspect_page",
]

_ATTR_MODULES = {
    "DatasheetTools": "datasheetindex.tools.bound",
    "DatasheetToolDef": "datasheetindex.tools.defs",
    "DatasheetToolSession": "datasheetindex.tools.defs",
    "create_datasheet_tool_defs": "datasheetindex.tools.defs",
    "create_datasheet_tool_session": "datasheetindex.tools.defs",
    "create_datasheet_tools_server": "datasheetindex.tools.registry",
    "inspect_page": "datasheetindex.tools.vision",
}


def __getattr__(name: str) -> object:
    module_name = _ATTR_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
