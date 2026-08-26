"""The classifier's LLM-assist client construction.

`classifier.py` was already published in this repository with no test
covering the path that builds an Anthropic client for LLM-assist
classification -- see `_create_llm_assist_client`. That path shares the
same `DISABLE_TLS_VERIFY` bug as `datasheet_tools._create_client`: the
locked anthropic==1.0.0 rejects a plain `httpx.Client` passed as
`http_client`, raising `TypeError` at construction, before any network
call happens.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

from chamberbench import classifier

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_classify_extra_declares_httpx2_explicitly():
    """`_create_llm_assist_client` imports `httpx2` directly under the
    `classify` extra -- declare it rather than relying on it arriving
    transitively through `anthropic`'s own dependency graph (the root
    pyproject.toml makes the identical argument about `jsonschema`)."""
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    classify_extra = " ".join(cfg["project"]["optional-dependencies"]["classify"])
    assert "httpx2" in classify_extra


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="needs the classify or harness extra: uv pip install -e '.[harness]'",
)
def test_create_llm_assist_client_constructs_real_client_under_tls_bypass(
    monkeypatch,
):
    """`_create_llm_assist_client` must build a REAL `Anthropic` client
    under `DISABLE_TLS_VERIFY=true` without the SDK constructor raising.

    Deliberately does NOT monkeypatch `Anthropic` -- construction is exactly
    where the locked anthropic==1.0.0 raises `TypeError: Invalid http_client
    argument; Expected an instance of httpx2.Client` when handed a plain
    `httpx.Client`. No network call is needed or made: only the client
    object is built. Only `setup_credentials` is stubbed (so no real
    credentials or .env file are needed) and only a fake API key is used.
    """
    monkeypatch.setattr("chamberbench.credentials.setup_credentials", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("DISABLE_TLS_VERIFY", "true")

    from anthropic import Anthropic

    client = classifier._create_llm_assist_client(Anthropic)

    assert isinstance(client, Anthropic)


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is None,
    reason="needs the classify or harness extra: uv pip install -e '.[harness]'",
)
def test_create_llm_assist_client_skips_http_client_override_when_tls_verify_stays_on(
    monkeypatch,
):
    """Sanity check on the other branch: no TLS bypass, no custom http_client,
    and the SDK constructor still does not raise."""
    monkeypatch.setattr("chamberbench.credentials.setup_credentials", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("DISABLE_TLS_VERIFY", raising=False)

    from anthropic import Anthropic

    client = classifier._create_llm_assist_client(Anthropic)

    assert isinstance(client, Anthropic)
