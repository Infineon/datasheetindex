"""The agent's tool surface survives the port."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

# See tests/test_harness_anthropic_path.py: Tier 1 does not install the harness
# extra, and both tool modules import `anthropic.beta_async_tool`.
pytest.importorskip(
    "anthropic",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)

from chamberbench.harness import chamber_tools, datasheet_tools

HARNESS = Path(__file__).resolve().parents[1] / "src" / "chamberbench" / "harness"


def test_no_private_project_imports():
    """Nothing may reach back into the private package."""
    for path in HARNESS.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "datasheet_agent" not in text, path


def test_dead_schema_hook_is_gone():
    """`_build_output_schema` was only reachable from the private repository's
    production engine, which this release does not contain."""
    assert not hasattr(datasheet_tools, "_build_output_schema")


def test_tool_factories_are_present():
    assert callable(chamber_tools._make_chamber_tools)
    assert callable(datasheet_tools._make_large_pdf_tools)


def test_tool_docstrings_are_intact(tmp_path):
    """The docstrings ARE the tool schema; an empty one is a silent break.

    `_make_large_pdf_tools` returns a 3-tuple (tools, cleanup, getter) -- only
    the first element is the tool list.
    """
    tools, _cleanup, _getter = datasheet_tools._make_large_pdf_tools(
        pdf_path=str(tmp_path / "absent.pdf")
    )
    assert tools, "no tools returned"
    for tool in tools:
        doc = inspect.getdoc(tool) or getattr(tool, "description", "")
        assert doc and len(doc) > 40, getattr(tool, "__name__", repr(tool))


def test_parse_json_from_text_tolerates_fenced_output():
    out = datasheet_tools._parse_json_from_text('```json\n{"a": 1}\n```')
    assert out == {"a": 1}


def test_create_client_honors_disable_tls_verify(monkeypatch):
    """`_create_client` must gate its httpx bypass on `tls_verify_disabled()`.

    A stale `NODE_TLS_REJECT_UNAUTHORIZED` check (the private repo's
    `setup_sdk_environment` translated `DISABLE_TLS_VERIFY` into that Node
    variable; Tier 1's `setup_credentials` deliberately does not) would make
    `DISABLE_TLS_VERIFY=true` silently do nothing. `setup_credentials` is
    monkeypatched to a no-op, and `AsyncAnthropic` to a plain kwargs-recording
    stub, so this needs no real credentials, no network, and no dependence on
    the installed anthropic SDK's own httpx-client type checking (an
    unrelated, separately-scoped concern from the gating logic under test).

    NOT SUFFICIENT ON ITS OWN: because `AsyncAnthropic` is stubbed out, this
    test cannot see whether the real SDK would even accept the http_client
    object being passed to it. It passed while `_create_client` built a
    plain `httpx.AsyncClient` that the locked anthropic==1.0.0 rejects at
    construction with `TypeError: Invalid http_client argument; Expected an
    instance of httpx2.AsyncClient` -- a real regression this test never
    observed. See `test_create_client_constructs_real_anthropic_client`
    below, which builds the actual `AsyncAnthropic` object and would have
    caught it.
    """

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("chamberbench.credentials.setup_credentials", lambda: None)
    monkeypatch.setattr(datasheet_tools, "AsyncAnthropic", _FakeAsyncAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    monkeypatch.setenv("DISABLE_TLS_VERIFY", "true")
    _client, http_client = datasheet_tools._create_client()
    assert http_client is not None

    monkeypatch.delenv("DISABLE_TLS_VERIFY", raising=False)
    _client, http_client = datasheet_tools._create_client()
    assert http_client is None


def test_create_client_constructs_real_anthropic_client(monkeypatch):
    """`_create_client` must build a REAL `AsyncAnthropic` under
    `DISABLE_TLS_VERIFY=true` without the SDK constructor itself raising.

    Deliberately does NOT monkeypatch `AsyncAnthropic` (unlike the test
    above): the locked anthropic==1.0.0 requires its `http_client` override
    to be an `httpx2.AsyncClient`, and rejects a plain `httpx.AsyncClient`
    with a `TypeError` at construction time -- no network call involved, no
    event loop needed. Only `setup_credentials` is stubbed (so no real
    credentials or .env file are needed) and only a fake API key is used;
    the client construction itself is completely real.
    """
    monkeypatch.setattr("chamberbench.credentials.setup_credentials", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("DISABLE_TLS_VERIFY", "true")

    from anthropic import AsyncAnthropic

    client, http_client = datasheet_tools._create_client()

    assert isinstance(client, AsyncAnthropic)
    assert http_client is not None
    assert type(http_client).__module__.startswith("httpx2")
