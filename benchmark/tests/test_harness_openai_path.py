"""The Responses-API engine survives the port."""

from __future__ import annotations

from pathlib import Path

import pytest

# See tests/test_harness_anthropic_path.py: Tier 1 does not install the harness
# extra, and this module's import chain reaches `openai` (and `anthropic`,
# which openai_path imports from).
pytest.importorskip(
    "openai",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)
pytest.importorskip(
    "anthropic",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)

from chamberbench.harness import openai_path

SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "chamberbench"
    / "harness"
    / "openai_path.py"
)


def test_uses_the_responses_api():
    """Reasoning summaries come from /v1/responses, not chat completions."""
    text = SRC.read_text(encoding="utf-8")
    assert "responses.create" in text


def test_requires_a_gateway_base_url(monkeypatch):
    """Gateway-only replication: refusing to run without one is correct,
    not a bug. See the spec, section 1."""
    for var in (
        "LITELLM_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "LITELLM_MASTER_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-key")
    monkeypatch.setattr("chamberbench.credentials.setup_credentials", lambda: None)
    with pytest.raises(ValueError, match="base URL"):
        openai_path._create_openai_client()


def test_missing_key_is_reported_clearly(monkeypatch):
    for var in ("LITELLM_MASTER_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("chamberbench.credentials.setup_credentials", lambda: None)
    with pytest.raises(ValueError, match="API key"):
        openai_path._create_openai_client()


def test_create_openai_client_constructs_real_client_under_tls_bypass(monkeypatch):
    """`_create_openai_client` must build a REAL `OpenAI` client under
    `DISABLE_TLS_VERIFY=true` without the SDK constructor raising.

    Unlike anthropic==1.0.0 (which rejects a plain `httpx.Client`),
    openai==3.3.1 accepts either httpx or httpx2 -- so this site was never
    actually broken. It is switched to httpx2 anyway, for one consistent
    client library across all three construction sites rather than two, and
    this test locks that choice in with a real, unmonkeypatched
    `OpenAI(...)` construction (no network call: construction alone does not
    touch the network).
    """
    monkeypatch.setattr("chamberbench.credentials.setup_credentials", lambda: None)
    for var in ("LITELLM_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test-fake")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example.invalid")
    monkeypatch.setenv("DISABLE_TLS_VERIFY", "true")

    from openai import OpenAI

    client, http_client = openai_path._create_openai_client()

    assert isinstance(client, OpenAI)
    assert http_client is not None
    assert type(http_client).__module__.startswith("httpx2")
