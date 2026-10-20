"""Page readers, the table scan, and the engine state they depend on.

Importing this package imports :mod:`datasheetindex.core.engine` first, which
resets the quad-corrections flag that ``import pymupdf`` turns on. So no
module here can read a page before that reset, whichever one a host imports
first -- ``from datasheetindex.core.textfile import extract_page_text`` alone
is a supported entry point (datasheet-agent uses it). The package root used to
guarantee this by importing everything; it resolves its exports lazily now.

Nothing that must run before ``import pymupdf`` may live in this package; see
``datasheetindex._scan_worker``.
"""

from datasheetindex.core import engine as engine
