"""Nothing published here may turn TLS verification off on a reader's behalf.

Seven ``os.environ.setdefault("DISABLE_TLS_VERIFY", "true")`` calls once ran
before any network call in this package, so a reader who pointed the harness
at an ``https://`` gateway shipped their API key over an unverified connection
and was told nothing -- an unverified connection succeeds exactly like a
verified one. They were deleted with no test behind the deletion, which left
reintroducing a single one of them perfectly green. The variable is still
read, still documented and still the right thing for a reader to set in their
own shell, so the shape is an easy one to restore by accident.

This lives in its own module rather than beside the producer-argument guard
because it is not about producers: five of the seven calls were in
``scripts/``, two were in library modules under ``src/``, and a guard watching
only the entry points would have caught five of them.
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = BENCHMARK_ROOT / "scripts"
SRC = BENCHMARK_ROOT / "src"

#: The variable ``credentials.tls_verify_disabled()`` reads. Reading it is the
#: whole point of the escape hatch; *writing* it is what this guard forbids.
_TLS_ENV_VAR = "DISABLE_TLS_VERIFY"


def _shipped_python_sources() -> list[Path]:
    """Every ``.py`` file a reader installs or runs, across both trees."""
    return sorted(
        p
        for tree in (SCRIPTS, SRC)
        for p in tree.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _tls_env_writes(path: Path) -> list[str]:
    """Every statement in ``path`` that *writes* ``DISABLE_TLS_VERIFY``.

    Three shapes are recognised, all of which set the variable for the
    process: subscript assignment (``os.environ["..."] = "true"``),
    ``setdefault`` -- the exact form that was deleted -- and ``os.putenv``.

    The check is on the parsed statement rather than on the source text
    because the text is unavoidably full of legitimate mentions: the reading
    helper, its docstring, and ``gateway/README.md``'s instructions all name
    the variable, so a substring guard would have to be either blind or
    permanently silenced. ``os.environ.get(...)`` is therefore untouched --
    ``credentials.py`` must keep reading it, and a guard that could not tell a
    read from a write would be deleted the first time it fired.

    Returns rendered source lines, so a failure names the offending statement
    rather than only the file.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []

    def _is_var(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and node.value == _TLS_ENV_VAR

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and _is_var(target.slice):
                    offenders.append(ast.unparse(node))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"setdefault", "putenv"}
                and node.args
                and _is_var(node.args[0])
            ):
                offenders.append(ast.unparse(node))
    return offenders


def test_no_shipped_source_disables_tls_verification():
    """Nothing under ``scripts/`` or ``src/`` may set ``DISABLE_TLS_VERIFY``."""
    offenders = {
        str(path.relative_to(BENCHMARK_ROOT)): writes
        for path in _shipped_python_sources()
        if (writes := _tls_env_writes(path))
    }
    assert not offenders, (
        f"published code sets {_TLS_ENV_VAR}, disabling certificate "
        f"verification for a reader who never asked: {offenders}"
    )


def test_the_tls_guard_can_actually_fail():
    """The predicate above must reject the exact shape that was deleted.

    Its sibling asserts an absence, which is the failure mode that quietly
    stops working: ``_tls_env_writes`` returning ``[]`` for everything would
    look identical to a clean tree. This drives it with the three write forms
    and with the read that has to stay allowed.
    """
    written = (
        "import os\n"
        'os.environ.setdefault("DISABLE_TLS_VERIFY", "true")\n'
        'os.environ["DISABLE_TLS_VERIFY"] = "1"\n'
        'os.putenv("DISABLE_TLS_VERIFY", "yes")\n'
    )
    read_only = (
        "import os\n"
        'flag = os.environ.get("DISABLE_TLS_VERIFY", "")\n'
        'other = os.environ.setdefault("SOMETHING_ELSE", "true")\n'
    )
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.py"
        bad.write_text(written, encoding="utf-8")
        assert len(_tls_env_writes(bad)) == 3

        good = Path(tmp) / "good.py"
        good.write_text(read_only, encoding="utf-8")
        assert _tls_env_writes(good) == []


def test_the_guard_reads_both_trees():
    """A tree that stopped being walked is the other way this goes quiet."""
    scanned = {p.resolve() for p in _shipped_python_sources()}
    assert (SCRIPTS / "variance.py").resolve() in scanned
    assert (SRC / "chamberbench" / "credentials.py").resolve() in scanned
