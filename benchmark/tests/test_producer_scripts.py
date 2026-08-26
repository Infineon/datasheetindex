"""Every published producer imports and offers --help. No network calls."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Importing or running a producer needs the harness extra; the source-level
#: AST checks in this file do not, so only the tests that actually execute
#: producer code carry this.
_NEEDS_HARNESS = pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None
    or importlib.util.find_spec("datasheetindex") is None,
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)

PRODUCERS = [
    "variance.py",
    "fault_injection.py",
    "fault_injection_multimodel.py",
    "null_tool.py",
    "perturbation.py",
    "fourth_component.py",
]


@_NEEDS_HARNESS
@pytest.mark.parametrize("name", PRODUCERS)
def test_producer_offers_help(name):
    """Running a producer needs the harness extra; the two source-level checks
    below do not, so only this one is skipped on a Tier-1 install."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / name), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("name", PRODUCERS)
def test_producer_has_no_private_imports(name):
    text = (SCRIPTS / name).read_text(encoding="utf-8")
    assert "datasheet_agent" not in text


def _producer_tree(name: str) -> ast.Module:
    return ast.parse((SCRIPTS / name).read_text(encoding="utf-8"))


def _mentions_archive(node: ast.AST, tainted: frozenset[str]) -> bool:
    """Does this expression resolve into ``archive/``?

    True for a direct ``archive_dir()`` / ``x.archive_dir()``, for a string
    literal naming the directory, and for any already-tainted module-level
    name -- which is what makes the check survive one level of indirection.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and (sub.id == "archive_dir" or sub.id in tainted):
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == "archive_dir":
            return True
        if (
            isinstance(sub, ast.Constant)
            and isinstance(sub.value, str)
            and "archive" in sub.value.lower()
        ):
            return True
    return False


def _archive_tainted_names(tree: ast.Module) -> frozenset[str]:
    """Module-level names that resolve into ``archive/``, transitively.

    ``default=`` rarely spells the archive out. It is at least as likely to
    name a constant (``default=_ARCHIVE``) or a helper
    (``default=_default_out()``) defined a few lines up -- and reading only
    the ``default=`` expression sees neither. Iterated to a fixed point so a
    chain of constants cannot launder the reference either.
    """
    tainted: frozenset[str] = frozenset()
    while True:
        grown = set(tainted)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if _mentions_archive(node, tainted):
                    grown.add(node.name)
                continue
            else:
                continue
            if not _mentions_archive(value, tainted):
                continue
            for target in targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        grown.add(sub.id)
        if grown == set(tainted):
            return tainted
        tainted = frozenset(grown)


def _add_argument_calls(name: str) -> list[ast.Call]:
    """Every ``parser.add_argument(...)`` call in a producer, as AST nodes."""
    tree = _producer_tree(name)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]


def _out_call(name: str) -> ast.Call:
    """The producer's single ``--out`` declaration."""
    calls = [
        c
        for c in _add_argument_calls(name)
        if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "--out"
    ]
    assert len(calls) == 1, f"{name}: expected exactly one --out, got {len(calls)}"
    return calls[0]


@pytest.mark.parametrize("name", PRODUCERS)
def test_producer_defaults_away_from_the_archive(name):
    """A producer must not default to overwriting the evidence.

    This asserts against the parsed argparse declaration, not against the
    source text. The predicate it replaced was
    ``"archive_dir()" not in text or "--out" in text``: every producer
    declares ``--out``, so the right operand was always true and the guard
    could never fail, whatever default a producer grew. Since it is the guard
    that stands between a re-run and read-only published evidence, it has to
    read what argparse will actually do.
    """
    call = _out_call(name)
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}

    required = kwargs.get("required")
    is_required = isinstance(required, ast.Constant) and required.value is True

    default = kwargs.get("default")
    # Absent, or explicitly None -- either way argparse supplies no value, so
    # a caller who omits --out cannot land on a path nobody chose.
    has_no_default = default is None or (
        isinstance(default, ast.Constant) and default.value is None
    )

    assert is_required or has_no_default, (
        f"{name}: --out is neither required=True nor default-free; a caller "
        "who omits it would silently write to whatever default was chosen"
    )


@pytest.mark.parametrize("name", PRODUCERS)
def test_no_producer_argument_defaults_into_the_archive(name):
    """No argument at all may default into ``archive/``.

    Broader than ``--out`` on purpose: any path argument that resolves to the
    archive can overwrite evidence, and a future producer is as likely to add
    ``--results-dir`` as to change ``--out``.

    That sentence used to be false of the code under it. The check read only
    the ``default=`` expression, so ``_ARCHIVE = archive_dir()`` followed by
    ``default=_ARCHIVE`` -- the ``--results-dir`` case named above, written
    the way anyone would write it -- carried neither the name
    ``archive_dir`` nor the string ``archive`` and passed. Resolution is now
    transitive through module-level constants and helpers; see
    ``_archive_tainted_names``.
    """
    tainted = _archive_tainted_names(_producer_tree(name))
    for call in _add_argument_calls(name):
        for kw in call.keywords:
            if kw.arg != "default":
                continue
            assert not _mentions_archive(kw.value, tainted), (
                f"{name}: {ast.unparse(call.args[0]) if call.args else '?'} "
                f"defaults to {ast.unparse(kw.value)}, which resolves into the "
                "read-only archive"
            )


@_NEEDS_HARNESS
@pytest.mark.parametrize(
    ("model", "preset", "expected_env", "expected_thinking"),
    [
        # The default the docstring pins: qwen, nothing set by the caller.
        ("qwen3.6-27b", None, "false", False),
        ("QWEN3.6-27B", None, "false", False),
        # The archive's `thinking_on` arm was produced by setting this
        # explicitly, so an explicit value must survive untouched.
        ("qwen3.6-27b", "true", "true", True),
        ("qwen3.6-27b", "false", "false", False),
        # A non-qwen model must not have the variable written at all.
        ("claudesonnet4.6", None, None, True),
        ("gpt-5.1", None, None, True),
        # An explicit setting still governs a non-qwen run, unforced.
        ("claudesonnet4.6", "false", "false", False),
    ],
)
def test_null_tool_forces_thinking_off_for_qwen(
    monkeypatch, model, preset, expected_env, expected_thinking
):
    """Upstream QwenLM/Qwen3#1817: thinking on breaks tool calls.

    The archive ships both arms, so the flag must stay reachable AND default
    off -- and this reads the resulting ``os.environ`` value back rather than
    grepping the source for the two substrings, which is what the previous
    version of this test did. That version stayed green when the forced value
    was inverted to ``"true"``, i.e. when the fidelity-critical default the
    docstring pins was reversed.
    """
    monkeypatch.delenv("CHAMBER_QWEN_ENABLE_THINKING", raising=False)
    if preset is not None:
        monkeypatch.setenv("CHAMBER_QWEN_ENABLE_THINKING", preset)

    import null_tool

    thinking = null_tool.resolve_qwen_thinking(model)

    assert os.environ.get("CHAMBER_QWEN_ENABLE_THINKING") == expected_env
    assert thinking is expected_thinking


@_NEEDS_HARNESS
def test_null_tool_thinking_flag_tolerates_whitespace_and_case():
    """``CHAMBER_QWEN_ENABLE_THINKING=" FALSE "`` is off, not on.

    A shell-exported value picks up spacing and capitalisation; reading it
    with a bare ``== "false"`` would silently re-enable reasoning on the very
    arm that exists to keep it off.
    """
    import null_tool

    env = {"CHAMBER_QWEN_ENABLE_THINKING": " FALSE "}
    assert null_tool.resolve_qwen_thinking("qwen3.6-27b", env) is False
    assert env["CHAMBER_QWEN_ENABLE_THINKING"] == " FALSE "


@_NEEDS_HARNESS
def test_null_tool_records_the_thinking_flag_it_resolved():
    """``main`` must report the value ``resolve_qwen_thinking`` returned.

    The artifact's ``qwen_enable_thinking`` field is the only record of which
    configuration produced a number, so the helper being right is not enough
    -- ``main`` has to be the thing calling it.
    """
    tree = ast.parse((SCRIPTS / "null_tool.py").read_text(encoding="utf-8"))
    main = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef) and n.name == "main"
    )
    called = {
        n.func.id
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "resolve_qwen_thinking" in called, (
        "null_tool.main no longer resolves the qwen thinking flag; the "
        "artifact would record a value nothing forced"
    )


def test_annotator_guide_gives_away_no_answer():
    """The guide ships as the blind bundle's README, so it must hold no needle.

    ``prepare_rederivation.py --bundle`` copies ``docs/annotator_guide.md`` in
    as ``README.md`` and then leak-audits the folder, so this is in principle
    already covered -- except that the bundle path crashed on a stale pre-port
    datasheet directory, so the audit had never once run. It found a
    formatting example written as ``'1.7'``, which is
    ``dps310-supply-voltage-vdd``'s actual needle. Checked here as well
    because this one runs offline, on every suite, without the corpus.
    """
    import re

    import yaml

    from chamberbench.claimsio import BENCHMARK_ROOT, claims_path

    claims = yaml.safe_load(claims_path().read_text(encoding="utf-8"))["claims"]
    needles = {
        str(n)
        for c in claims
        for n in (c.get("value_contains") or [])
        if len(str(n)) >= 3 and re.search(r"\d", str(n))
    }
    guide = (BENCHMARK_ROOT / "docs" / "annotator_guide.md").read_text(encoding="utf-8")
    # Same digit-boundary rule the bundle's own leak audit uses.
    found = sorted(
        n for n in needles if re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", guide)
    )
    assert not found, f"docs/annotator_guide.md hands the annotator {found}"


# ---------------------------------------------------------------------------
# fourth_component.py's --pdf pre-flight
# ---------------------------------------------------------------------------


@_NEEDS_HARNESS
def test_fourth_component_rewrites_every_claim_to_the_given_pdf(tmp_path):
    """``--pdf`` must reach every claim, not just the first.

    ``data/claims_a4988.yaml`` carries the bare part label ``A4988`` as its
    ``pdf_source``, which resolves to no file at all -- so a claim the rewrite
    misses is a claim that runs against nothing.
    """
    import fourth_component
    from chamberbench.claimsio import A4988_CLAIMS_FILENAME, load_claims

    pdf = tmp_path / "A4988.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    claims = load_claims(A4988_CLAIMS_FILENAME)
    assert claims, "the A4988 claim file is empty; this test would prove nothing"
    assert all(c.pdf_source == "A4988" for c in claims)

    resolved = fourth_component._resolve_corpus(
        claims, str(pdf), argparse.ArgumentParser()
    )

    assert len(resolved) == len(claims)
    assert {c.pdf_source for c in resolved} == {str(pdf)}


@_NEEDS_HARNESS
def test_fourth_component_rejects_a_pdf_path_that_does_not_exist(tmp_path, capsys):
    """A typo'd ``--pdf`` is a setup mistake, and must exit before billing.

    The message is asserted, not just the exit code. Delete the ``is_file()``
    check and every claim is rewritten to the missing path, which the
    unresolvable-corpus branch below then rejects with the *same* exit code --
    so an exit-code-only assertion cannot tell the two apart, and the reader
    is told the claim file is wrong when their ``--pdf`` argument is.
    """
    import fourth_component
    from chamberbench.claimsio import A4988_CLAIMS_FILENAME, load_claims

    missing = tmp_path / "nope.pdf"
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit) as excinfo:
        fourth_component._resolve_corpus(
            load_claims(A4988_CLAIMS_FILENAME), str(missing), parser
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--pdf: no such file" in err, err
    assert str(missing) in err


@_NEEDS_HARNESS
def test_fourth_component_refuses_to_run_with_an_unresolvable_corpus(capsys):
    """No ``--pdf`` at all: the whole point of the pre-flight.

    Left to the run loop, twelve ``FileNotFoundError`` cells are swallowed
    into ``engine_error``, summarised as twelve failures, and exit 0 -- an
    artifact that looks like an experiment and reports nothing. The message
    has to name ``--pdf``, because "12 of 12 claims failed" does not tell a
    reader what to do next.
    """
    import fourth_component
    from chamberbench.claimsio import A4988_CLAIMS_FILENAME, load_claims

    claims = load_claims(A4988_CLAIMS_FILENAME)
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit) as excinfo:
        fourth_component._resolve_corpus(claims, "", parser)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"{len(claims)} of {len(claims)} claims" in err, err
    assert "--pdf" in err, err


@_NEEDS_HARNESS
def test_fourth_component_preflight_leaves_urls_and_real_files_alone(
    monkeypatch, tmp_path
):
    """A resolvable corpus passes through untouched, and downloads nothing.

    A URL is resolved and cached by the engine on first use; checking it here
    would mean fetching during pre-flight. ``requests.get`` is made to explode
    -- ``anthropic_path`` imports it lazily, inside the download branch, so
    patching the module itself is what covers that branch -- and a future
    rewrite that starts downloading therefore fails loudly instead of quietly
    adding a network call to a pre-flight check.
    """
    import requests

    import fourth_component
    from chamberbench.claimsio import A4988_CLAIMS_FILENAME, load_claims

    def _no_network(*args, **kwargs):  # pragma: no cover -- the failure it guards
        raise AssertionError("the pre-flight must not reach the network")

    monkeypatch.setattr(requests, "get", _no_network)

    local = tmp_path / "local.pdf"
    local.write_bytes(b"%PDF-1.7\n")
    base = load_claims(A4988_CLAIMS_FILENAME)
    claims = [
        base[0].model_copy(update={"pdf_source": "https://example.invalid/a.pdf"}),
        base[1].model_copy(update={"pdf_source": str(local)}),
    ]

    resolved = fourth_component._resolve_corpus(claims, "", argparse.ArgumentParser())

    assert [c.pdf_source for c in resolved] == [
        "https://example.invalid/a.pdf",
        str(local),
    ]
