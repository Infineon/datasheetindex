"""The harness package's configuration surface."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from chamberbench import claimsio
from chamberbench.claims import TraceStep
from chamberbench.harness import (
    CHAMBER_MODEL_CONFIG,
    model_config,
    rollup_cell_usage,
)


def test_corpus_dir_defaults_under_benchmark_root():
    assert claimsio.corpus_dir() == claimsio.BENCHMARK_ROOT / "corpus"


def test_corpus_dir_honours_env_override(monkeypatch):
    monkeypatch.setenv("CHAMBERBENCH_CORPUS_DIR", "/tmp/somewhere")
    assert claimsio.corpus_dir() == Path("/tmp/somewhere")


def test_archived_aliases_are_present():
    """The three aliases the archive was produced under must resolve."""
    for alias in ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"):
        assert alias in CHAMBER_MODEL_CONFIG


def test_unknown_model_falls_through_to_safe_defaults():
    """An unknown alias must get the Sonnet-shaped ceiling, not a guess."""
    cfg = model_config("some-model-we-never-heard-of")
    assert cfg["max_turns"] == 30
    assert cfg["inspect_page_detail"] == "high"


def test_qwen_uses_low_detail_tier():
    assert model_config("qwen3.6-27b")["inspect_page_detail"] == "low"


def test_rollup_does_not_double_count_within_a_turn():
    """Usage is duplicated onto every tool_call in a turn; count it once."""
    # TraceStep requires run_id/claim_id/engine/step; filled with placeholder
    # values here since the rollup logic under test only reads kind,
    # turn_idx, and the token counters.
    steps = [
        TraceStep(
            run_id="run-1",
            claim_id="claim-1",
            engine="agentic",
            step=0,
            kind="tool_call",
            turn_idx=0,
            input_tokens=100,
            output_tokens=10,
        ),
        TraceStep(
            run_id="run-1",
            claim_id="claim-1",
            engine="agentic",
            step=1,
            kind="tool_call",
            turn_idx=0,
            input_tokens=100,
            output_tokens=10,
        ),
        TraceStep(
            run_id="run-1",
            claim_id="claim-1",
            engine="agentic",
            step=2,
            kind="tool_call",
            turn_idx=1,
            input_tokens=200,
            output_tokens=20,
        ),
        TraceStep(
            run_id="run-1",
            claim_id="claim-1",
            engine="agentic",
            step=3,
            kind="final_output",
            turn_idx=2,
            input_tokens=50,
            output_tokens=5,
        ),
    ]
    assert rollup_cell_usage(steps) == {
        "input_tokens": 350,
        "output_tokens": 35,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }


# Directories the guard walks recursively, relative to the benchmark root.
# Deliberately omits the dotdirs (``.venv``, ``.pytest_cache``,
# ``.ruff_cache``), which hold third-party or generated files rather than
# anything we authored.
# ``data/`` joined this list after four internal sprint codenames (``Day 10``,
# ``Day 11`` x2, ``Day 13+``) survived undetected in ``data/claims.yaml`` --
# the guard had never walked the claim data files at all, only the code and
# docs trees.
# ``archive/`` joined it for its prose only; see ``_CODENAME_GUARD_DIR_GLOBS``.
_CODENAME_GUARD_DIRS = (
    "src",
    "scripts",
    "tests",
    "docs",
    "gateway",
    "data",
    "archive",
)
_CODENAME_GUARD_GLOBS = ("*.py", "*.md", "*.yaml", "*.yml", "*.toml")

#: Directories scanned with a narrower glob than ``_CODENAME_GUARD_GLOBS``.
#:
#: ``archive/`` is scanned as ``*.md`` ONLY, and the exemption of its
#: ``.json``/``.jsonl`` files is deliberate rather than an oversight to be
#: tidied up later. Those files are recorded outputs of runs that actually
#: happened: they are the evidence this benchmark rests on, a session fixture
#: in ``conftest.py`` digests every byte of the directory to prove no test
#: mutated them, and editing a recorded run so that it reads more cleanly is
#: the one thing an archive may never do -- it would make the record a
#: description of what we wish had run. The archive's *prose* carries no such
#: status, so ``archive/README.md`` is held to the same standard as every
#: other document a reader sees.
#:
#: One topology mention therefore remains inside the recorded outputs and
#: is expected to stay there: the superseded
#: ``null_tool_injection.qwen3.6-27b.thinking_on.json`` run explains itself
#: as predating a tool-parser change on the gateway that served it, and
#: names that gateway while doing so. It is left exactly as written. A
#: second known mention sat in ``archive/README.md`` and was reworded
#: rather than exempted, which is precisely the line this glob draws:
#: prose describing the evidence is ours to write, the evidence is not.
#: Anyone auditing this scrub with a bare grep will still find the one in
#: the JSON; this comment is what tells them it is a decision, not a miss.
_CODENAME_GUARD_DIR_GLOBS = {"archive": ("*.md",)}
# Files at the benchmark root that no glob above reaches -- the root itself is
# walked only as ``*.md``. They are clean today; they were also unscanned,
# which is how every previous escape happened.
# ``pyproject.toml`` is here because ``*.toml`` in ``_CODENAME_GUARD_GLOBS``
# matched *zero files*: the globs apply only inside ``_CODENAME_GUARD_DIRS``,
# and there is no ``.toml`` under any of them. The extension was listed and
# uncovered, and a topology string appended to ``pyproject.toml`` passed the
# whole suite. ``test_the_codename_guard_scans_what_it_claims_to`` below now
# fails if any listed extension or root file goes back to matching nothing.
_CODENAME_GUARD_ROOT_FILES = ("NOTICE", "LICENSE", "CITATION.cff", "pyproject.toml")


#: Paths under ``benchmark/`` that hold nothing we authored: dotdirs
#: (``.venv``, ``.pytest_cache``, ``.ruff_cache``), the fetched third-party
#: corpus and the regenerable figures -- all three gitignored. Everything
#: else with a guarded extension must be reachable by the guard.
_UNAUTHORED_DIRS = frozenset({"corpus", "figures", "results", "eval_results"})


def _is_generated_or_fetched(rel: Path) -> bool:
    return rel.parts[0].startswith(".") or rel.parts[0] in _UNAUTHORED_DIRS


def _codename_guard_paths(benchmark_root: Path) -> list[Path]:
    """Every file the codename guard scans.

    Originally this only walked ``src/chamberbench/harness`` as ``*.py`` --
    narrow enough that a new directory (like ``gateway/``, added for the
    reference LiteLLM config) would join the tree with no coverage at all.
    Two internal sprint codenames (``Layer 2``, ``Day 4``) already survived
    a deliberate scrub and a code review once because the guard's reach was
    too narrow; this widens it to the whole tree a reader actually sees --
    ``src/``, ``scripts/``, ``tests/``, ``docs/``, ``gateway/``, ``data/`` and
    ``archive/`` recursively, plus the benchmark root's own top-level ``*.md``
    files (``README.md``) -- and to ``.md``/``.yaml``/``.yml``/``.toml`` in
    addition to ``.py``, plus the root files no glob reaches (``NOTICE``,
    ``LICENSE``, ``CITATION.cff``, ``pyproject.toml``). ``archive/`` is the
    one directory scanned at a narrower glob; ``_CODENAME_GUARD_DIR_GLOBS``
    says why.

    Listing an extension is not the same as covering it --
    ``test_the_codename_guard_scans_what_it_claims_to`` is what ties the two
    together, and it exists because ``*.toml`` was listed while the tree's
    only ``.toml`` sat at the root, unscanned.
    """
    paths: list[Path] = []
    for dirname in _CODENAME_GUARD_DIRS:
        base = benchmark_root / dirname
        if not base.exists():
            continue
        for glob in _CODENAME_GUARD_DIR_GLOBS.get(dirname, _CODENAME_GUARD_GLOBS):
            paths.extend(base.rglob(glob))
    paths.extend(benchmark_root.glob("*.md"))
    # Deliberately NOT gated on ``.exists()``: a root file that is renamed or
    # removed used to drop out of the scan silently, which is coverage loss
    # dressed up as a passing suite. Missing entries are reported by
    # ``test_the_codename_guard_scans_what_it_claims_to``.
    paths.extend(benchmark_root / name for name in _CODENAME_GUARD_ROOT_FILES)
    # This file necessarily quotes the codenames it guards against, in its
    # own docstrings and literals -- exclude it, not the pattern.
    this_file = Path(__file__).resolve()
    return [p for p in paths if p.resolve() != this_file]


#: Internal sprint codenames. Case-insensitive and separator-optional: the
#: earlier ``\b(Day|Layer)[- ]\d+`` required a capital and a separator, so it
#: read ``Day 4`` but not ``TODO(day3)`` -- which is exactly the form that
#: survived in the largest file's module docstring. Anchored on the literal
#: words ``day`` / ``layer`` and never on ``phase``, so it cannot match the
#: ``PHASE 1`` / ``PHASE 2`` prompt text -- see
#: ``test_phase_markers_survive_the_codename_guard`` below.
_CODENAME_PATTERN = re.compile(r"(?i)\b(day|layer)[- ]?\d+")

#: Deployment topology. A public research artifact must not describe the
#: private cluster it happened to be developed against: a reader cannot act on
#: "the prod gateway" or "reaches oc logs", and naming an internal environment
#: leaks a fact about our infrastructure for no reader's benefit. Say the
#: portable thing instead -- "a gateway that serves qwen3.6-27b".
#:
#: The qualifier and the word ``gateway`` need not be adjacent, and requiring
#: that was this guard's real defect rather than its reach: ``prod[- ]gateway``
#: read "the prod gateway" but not "the prod LiteLLM gateway", which is how the
#: string is actually written. Seven sites in directories the guard was already
#: walking -- ``src/``, ``scripts/``, ``docs/``, ``data/`` -- were live and
#: green, because a middle word is the natural way to name the thing. Up to two
#: intervening words are allowed, which also catches "internal self-signed
#: gateway".
_TOPOLOGY_PATTERN = re.compile(
    r"(?i)\b(?:prod|production|internal)\b[-\s](?:\w+[-\s]){0,2}gateway"
    r"|staging|oc logs|openshift|playground"
)


def test_no_internal_codenames_or_hosts():
    """The scrub must hold across the whole ``benchmark/`` tree.

    See the spec's Global Constraints. Three separate escapes are pinned here:
    the literal ``connectorapp``; sprint codenames (``_CODENAME_PATTERN``,
    which two ``anthropic_path.py`` comments carrying ``Layer 2`` and ``Day 4``
    once cleared, and a third carrying ``TODO(day3)`` cleared again after the
    pattern was widened only for case); and deployment topology
    (``_TOPOLOGY_PATTERN``), which had no coverage at all until six new sites
    naming a private cluster shipped on one branch.
    """
    benchmark_root = Path(__file__).resolve().parents[1]
    for path in _codename_guard_paths(benchmark_root):
        text = path.read_text(encoding="utf-8")
        assert "connectorapp" not in text, path
        match = _CODENAME_PATTERN.search(text)
        assert match is None, (path, match.group(0) if match else None)
        match = _TOPOLOGY_PATTERN.search(text)
        assert match is None, (path, match.group(0) if match else None)


def test_the_codename_guard_scans_what_it_claims_to():
    """Every extension and root file the guard lists must match a real file.

    ``_CODENAME_GUARD_GLOBS`` gained ``*.toml`` and covered nothing: the
    globs are applied only inside ``_CODENAME_GUARD_DIRS`` and no ``.toml``
    lives under any of them, so the only one in the tree --
    ``pyproject.toml``, at the root -- went unscanned while the commit
    message read as though it had been covered. Appending
    "runs on openshift against the prod gateway" to it left the suite green.

    This is the guard on the guard: a listed extension that matches nothing,
    or a root file that has been renamed out from under the tuple, is
    coverage that exists only on paper.
    """
    benchmark_root = Path(__file__).resolve().parents[1]
    scanned = _codename_guard_paths(benchmark_root)

    missing = [
        name
        for name in _CODENAME_GUARD_ROOT_FILES
        if not (benchmark_root / name).is_file()
    ]
    assert not missing, (
        f"_CODENAME_GUARD_ROOT_FILES names files that do not exist: {missing}; "
        "a renamed root file must update the tuple, not silently stop being scanned"
    )

    guarded = {g.removeprefix("*") for g in _CODENAME_GUARD_GLOBS}
    # This file is excluded from the scan by design -- it quotes every
    # codename it guards against -- so it is the one expected absence.
    scanned_set = {p.resolve() for p in scanned} | {Path(__file__).resolve()}
    unscanned = sorted(
        str(path.relative_to(benchmark_root))
        for path in benchmark_root.rglob("*")
        if path.is_file()
        and path.suffix in guarded
        and not _is_generated_or_fetched(path.relative_to(benchmark_root))
        and path.resolve() not in scanned_set
    )
    assert not unscanned, (
        f"{unscanned} carry an extension _CODENAME_GUARD_GLOBS lists but sit "
        "somewhere the guard does not walk; the extension is covered on paper "
        "only. Add the directory to _CODENAME_GUARD_DIRS, or the file to "
        "_CODENAME_GUARD_ROOT_FILES."
    )


def test_phase_markers_survive_the_codename_guard():
    """PHASE 1 / PHASE 2 must never be caught by the codename regex above.

    They are the two-pass-freeze methodology, live inside the agent's own
    prompt text in anthropic_path.py, and must be preserved exactly. This
    pins their survival so a future widening of the codename pattern cannot
    silently start scrubbing them.
    """
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "chamberbench"
        / "harness"
        / "anthropic_path.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "PHASE 1" in text
    assert "PHASE 2" in text
    pattern = re.compile(r"\b(Day|Layer)[- ]\d+")
    assert pattern.search("PHASE 1") is None
    assert pattern.search("PHASE 2") is None
    assert pattern.search("phase 1") is None
    assert pattern.search("phase 2") is None


def test_archive_prose_is_guarded_but_recorded_outputs_are_not():
    """``archive/`` is scanned for ``*.md`` and for nothing else.

    Both halves matter and neither is incidental. Dropping the README would
    leave a document a reader actually reads outside every scrub the rest of
    the tree gets. Adding the ``.json``/``.jsonl`` runs would invite a future
    contributor to edit recorded evidence so that it reads more cleanly --
    which is the one edit an archive may not carry, and which the session
    digest in ``conftest.py`` exists to make loud. Pinning the split here
    means a change to either has to be argued for rather than typed.
    """
    benchmark_root = Path(__file__).resolve().parents[1]
    archive = benchmark_root / "archive"
    recorded = [p for p in archive.rglob("*") if p.suffix in {".json", ".jsonl"}]
    assert recorded, "archive holds no recorded runs; this guard is testing nothing"

    scanned = {p.resolve() for p in _codename_guard_paths(benchmark_root)}
    assert (archive / "README.md").resolve() in scanned
    assert not [p for p in recorded if p.resolve() in scanned]

    # The exemption must be doing the work, not the default glob list happening
    # to omit ``*.json`` today. Widen the default the way a future change
    # wanting to scan some other directory's JSON would, and the recorded runs
    # must still be out of reach.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            sys.modules[__name__],
            "_CODENAME_GUARD_GLOBS",
            (*_CODENAME_GUARD_GLOBS, "*.json", "*.jsonl"),
        )
        widened = {p.resolve() for p in _codename_guard_paths(benchmark_root)}
    assert (archive / "README.md").resolve() in widened
    assert not [p for p in recorded if p.resolve() in widened], (
        "archive/ is exempt from *.json only by accident of the default glob "
        "list; _CODENAME_GUARD_DIR_GLOBS must be what keeps recorded runs out"
    )
