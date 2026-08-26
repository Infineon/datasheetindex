"""The runner and its CLI. No test here makes a network call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

# See tests/test_harness_anthropic_path.py: Tier 1 does not install the harness
# extra, and the runner imports both engines.
pytest.importorskip(
    "anthropic",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)
pytest.importorskip(
    "openai",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)

from chamberbench.claims import ClaimResult, ClaimSpec
from chamberbench.harness import anthropic_path, openai_path, run

SRC = (
    Path(__file__).resolve().parents[1] / "src" / "chamberbench" / "harness" / "run.py"
)


def test_claims_load_through_claimsio():
    claims = run.load_chamber_claims()
    assert len(claims) == 25
    assert all(c.id for c in claims)


def test_no_hardcoded_private_paths():
    text = SRC.read_text(encoding="utf-8")
    for bad in (
        "eval_results/chamber",
        "eval/chamber/claims.yaml",
        "parents[2]",
        "parents[3]",
    ):
        assert bad not in text, bad


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        run.main(["--help"])
    assert exc.value.code == 0
    assert "--model" in capsys.readouterr().out


def test_cli_rejects_unknown_engine():
    with pytest.raises(SystemExit):
        run.main(["--engine", "telepathy"])


def test_collector_writes_where_told(tmp_path):
    """Never into archive/. The conftest fixture enforces that globally,
    but the collector must take an explicit destination."""
    collector = run.ChamberResultsCollector(results_dir=tmp_path, model="testmodel")
    summary = collector.write_summary()
    collector.close()
    assert summary.parent == tmp_path
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["model"] == "testmodel"


def test_collector_rejects_the_archive_as_a_destination():
    """No default may point at the archive, and no explicit pass-through
    should either -- the archive is committed evidence, not a scratch dir."""
    from chamberbench.claimsio import archive_dir

    with pytest.raises(ValueError):
        run.ChamberResultsCollector(results_dir=archive_dir(), model="testmodel")


def test_summary_is_self_describing(tmp_path):
    """A reader must be able to tell which datasheetindex tool surface
    produced a run without cross-referencing a separate changelog."""
    collector = run.ChamberResultsCollector(results_dir=tmp_path, model="testmodel")
    payload = collector.to_dict()
    collector.close()
    assert "datasheetindex_version" in payload
    assert "timestamp" in payload
    assert "model" in payload


def _minimal_claim(claim_id: str, chamber_dataset: str = "wt_validate_v1") -> ClaimSpec:
    return ClaimSpec(
        id=claim_id,
        pdf_source="https://example.invalid/datasheet.pdf",
        parameter="test parameter",
        expected_unit="V",
        claim_kind="typical",
        chamber_dataset=chamber_dataset,
        chamber_protocol="chamberbench.protocols._common",
        primary_chamber_variable="test_variable",
    )


def test_router_dispatches_gpt_models_to_the_openai_path(monkeypatch):
    """The Anthropic-shape router (`anthropic_path.extract_chamber_*`) must
    dispatch a `gpt-*` model into the Responses-API engine
    (`openai_path.extract_chamber_*_openai`), and must NOT dispatch a
    Claude model there. Nothing exercises this today except live, paid
    gateway traffic -- the keyword arguments match by inspection, but a
    future drift in either signature would only surface as a real-money
    surprise. Fully mocked; makes no network call."""
    claim = _minimal_claim("router-seam-claim")
    stub_result = ClaimResult(
        claim_id=claim.id,
        found=True,
        confidence=0.9,
        extracted={"parameter": "test parameter", "found": True, "confidence": 0.9},
    )

    openai_baseline_calls: list[str] = []
    openai_agentic_calls: list[str] = []
    anthropic_baseline_calls: list[str] = []
    anthropic_agentic_calls: list[str] = []

    async def fake_openai_baseline(_claim, *, model, **_kwargs):
        openai_baseline_calls.append(model)
        return stub_result

    async def fake_openai_agentic(_claim, *, model, **_kwargs):
        openai_agentic_calls.append(model)
        return stub_result

    async def fake_anthropic_baseline(_claim, *, model, **_kwargs):
        anthropic_baseline_calls.append(model)
        return stub_result

    async def fake_anthropic_agentic(_claim, *, model, **_kwargs):
        anthropic_agentic_calls.append(model)
        return stub_result

    monkeypatch.setattr(
        openai_path, "extract_chamber_baseline_openai", fake_openai_baseline
    )
    monkeypatch.setattr(
        openai_path, "extract_chamber_agentic_openai", fake_openai_agentic
    )
    monkeypatch.setattr(
        anthropic_path, "_extract_chamber_baseline_anthropic", fake_anthropic_baseline
    )
    monkeypatch.setattr(
        anthropic_path, "_extract_chamber_agentic_anthropic", fake_anthropic_agentic
    )

    # A gpt-* model reaches the openai_path entry points, not the Anthropic
    # ones.
    asyncio.run(anthropic_path.extract_chamber_baseline(claim, model="gpt-5.1"))
    asyncio.run(anthropic_path.extract_chamber_agentic(claim, model="gpt-5.1"))
    assert openai_baseline_calls == ["gpt-5.1"]
    assert openai_agentic_calls == ["gpt-5.1"]
    assert anthropic_baseline_calls == []
    assert anthropic_agentic_calls == []

    # A Claude model reaches the Anthropic-shape entry points, never the
    # openai_path ones.
    asyncio.run(anthropic_path.extract_chamber_baseline(claim, model="claudesonnet4.6"))
    asyncio.run(anthropic_path.extract_chamber_agentic(claim, model="claudesonnet4.6"))
    assert anthropic_baseline_calls == ["claudesonnet4.6"]
    assert anthropic_agentic_calls == ["claudesonnet4.6"]
    assert openai_baseline_calls == ["gpt-5.1"]
    assert openai_agentic_calls == ["gpt-5.1"]


def test_run_py_has_no_pytest_collectable_functions():
    """The safety property behind the rename: no function in run.py may be
    named test_* (it makes live, billable API calls)."""
    text = SRC.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("def test_"), line
        assert not stripped.startswith("async def test_"), line
