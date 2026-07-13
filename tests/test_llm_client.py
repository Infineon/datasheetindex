"""Tests for the LLM client factory."""

from __future__ import annotations

import sys
import types

import pytest


def _noop_load_dotenv(*args, **kwargs):
    pass


def _install_fake_dotenv(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=_noop_load_dotenv),
    )


class _TrackedHttpxClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _patch_fake_clients(
    monkeypatch,
    seen_httpx_kwargs: dict[str, object],
    seen_openai_kwargs: dict[str, object],
    httpx_clients: list[_TrackedHttpxClient] | None = None,
) -> None:
    class _FakeOpenAI:
        def __init__(self, **kwargs):
            seen_openai_kwargs.clear()
            seen_openai_kwargs.update(kwargs)
            self.responses = types.SimpleNamespace(
                create=lambda **_kwargs: types.SimpleNamespace(output_text="ok")
            )

    def _fake_httpx_client(**kwargs):
        seen_httpx_kwargs.clear()
        seen_httpx_kwargs.update(kwargs)
        client = _TrackedHttpxClient()
        if httpx_clients is not None:
            httpx_clients.append(client)
        return client

    fake_httpx = types.SimpleNamespace(Client=_fake_httpx_client)
    fake_openai = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)


def test_create_llm_client_raises_without_env(monkeypatch):
    """Should raise ValueError when env vars are missing."""
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    _install_fake_dotenv(monkeypatch)

    from datasheetindex.llm.client import create_llm_client

    with pytest.raises(ValueError, match="LITELLM_BASE_URL"):
        create_llm_client()


def test_create_llm_client_raises_partial_env(monkeypatch):
    """Should raise ValueError when only one env var is set."""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    _install_fake_dotenv(monkeypatch)

    from datasheetindex.llm.client import create_llm_client

    with pytest.raises(ValueError, match="LITELLM_BASE_URL"):
        create_llm_client()


def test_create_llm_client_tls_verify_defaults_false(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.delenv("LITELLM_TLS_VERIFY", raising=False)
    _install_fake_dotenv(monkeypatch)

    seen_httpx_kwargs: dict[str, object] = {}
    _seen_openai_kwargs: dict[str, object] = {}
    _patch_fake_clients(monkeypatch, seen_httpx_kwargs, _seen_openai_kwargs)

    from datasheetindex.llm.client import create_llm_client

    llm = create_llm_client()
    assert callable(llm)
    assert seen_httpx_kwargs["verify"] is False


def test_create_llm_client_tls_verify_can_be_enabled(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("LITELLM_TLS_VERIFY", "true")
    _install_fake_dotenv(monkeypatch)

    seen_httpx_kwargs: dict[str, object] = {}
    _seen_openai_kwargs: dict[str, object] = {}
    _patch_fake_clients(monkeypatch, seen_httpx_kwargs, _seen_openai_kwargs)

    from datasheetindex.llm.client import create_llm_client

    llm = create_llm_client()
    assert callable(llm)
    assert seen_httpx_kwargs["verify"] is True


def test_create_llm_client_timeout_and_retries_defaults(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.delenv("LITELLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LITELLM_MAX_RETRIES", raising=False)
    _install_fake_dotenv(monkeypatch)

    seen_httpx_kwargs: dict[str, object] = {}
    seen_openai_kwargs: dict[str, object] = {}
    _patch_fake_clients(monkeypatch, seen_httpx_kwargs, seen_openai_kwargs)

    from datasheetindex.llm.client import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_TIMEOUT_SECONDS,
        create_llm_client,
    )

    llm = create_llm_client()
    assert callable(llm)
    assert seen_httpx_kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS
    assert seen_openai_kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS
    assert seen_openai_kwargs["max_retries"] == DEFAULT_MAX_RETRIES


def test_create_llm_client_timeout_and_retries_override(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("LITELLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LITELLM_MAX_RETRIES", "5")
    _install_fake_dotenv(monkeypatch)

    seen_httpx_kwargs: dict[str, object] = {}
    seen_openai_kwargs: dict[str, object] = {}
    _patch_fake_clients(monkeypatch, seen_httpx_kwargs, seen_openai_kwargs)

    from datasheetindex.llm.client import create_llm_client

    llm = create_llm_client()
    assert callable(llm)
    assert seen_httpx_kwargs["timeout"] == 12.5
    assert seen_openai_kwargs["timeout"] == 12.5
    assert seen_openai_kwargs["max_retries"] == 5


def test_create_llm_client_invalid_timeout_raises(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("LITELLM_TIMEOUT_SECONDS", "0")
    _install_fake_dotenv(monkeypatch)

    from datasheetindex.llm.client import create_llm_client

    with pytest.raises(ValueError, match="LITELLM_TIMEOUT_SECONDS"):
        create_llm_client()


def test_create_llm_client_invalid_retries_raises(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("LITELLM_MAX_RETRIES", "-1")
    _install_fake_dotenv(monkeypatch)

    from datasheetindex.llm.client import create_llm_client

    with pytest.raises(ValueError, match="LITELLM_MAX_RETRIES"):
        create_llm_client()


def test_close_llm_client_closes_httpx_client(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    _install_fake_dotenv(monkeypatch)

    seen_httpx_kwargs: dict[str, object] = {}
    seen_openai_kwargs: dict[str, object] = {}
    httpx_clients: list[_TrackedHttpxClient] = []
    _patch_fake_clients(
        monkeypatch,
        seen_httpx_kwargs,
        seen_openai_kwargs,
        httpx_clients=httpx_clients,
    )

    from datasheetindex.llm.client import close_llm_client, create_llm_client

    llm = create_llm_client()
    close_llm_client(llm)

    assert len(httpx_clients) == 1
    assert httpx_clients[0].closed is True


def test_get_structured_output_client_returns_none_for_plain_callable():
    from datasheetindex.llm.client import get_structured_output_client

    assert get_structured_output_client(lambda _system, _user: "ok") is None


def test_get_structured_output_client_exposes_schema_calls(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    _install_fake_dotenv(monkeypatch)

    seen_request: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = types.SimpleNamespace(
                create=lambda **kwargs: (
                    seen_request.update(kwargs)
                    or types.SimpleNamespace(
                        output_text='{"entries":[]}',
                        status="completed",
                        incomplete_details=None,
                    )
                )
            )

    def _fake_httpx_client(**_kwargs):
        return _TrackedHttpxClient()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        types.SimpleNamespace(Client=_fake_httpx_client),
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=_FakeOpenAI),
    )

    from datasheetindex.llm.client import (
        create_llm_client,
        get_structured_output_client,
    )

    llm = create_llm_client()
    structured = get_structured_output_client(llm)
    assert structured is not None

    result = structured.structured_json(
        "sys",
        "user",
        name="probe-schema",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
        max_output_tokens=123,
    )

    assert seen_request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "probe-schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }
    }
    assert seen_request["max_output_tokens"] == 123
    assert result.output_text == '{"entries":[]}'
    assert result.status == "completed"


def test_call_with_retry_retries_on_429(monkeypatch):
    """Should retry on rate limit errors and succeed."""
    from datasheetindex.llm.client import _call_with_retry

    monkeypatch.setattr("datasheetindex.llm.client._RETRY_BASE_DELAY", 0.01)

    call_count = 0

    class _RateLimitError(Exception):
        status_code = 429

    def fake_create(*, model, instructions, input):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _RateLimitError("Too Many Requests")
        return types.SimpleNamespace(output_text="success")

    api = types.SimpleNamespace(create=fake_create)
    result = _call_with_retry(api, "model", "sys", "user")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert result == "success"
    assert call_count == 3


def test_call_structured_with_retry_retries_on_429(monkeypatch):
    """Structured calls should retry on rate limit errors and succeed."""
    from datasheetindex.llm.client import _call_structured_with_retry

    monkeypatch.setattr("datasheetindex.llm.client._RETRY_BASE_DELAY", 0.01)

    call_count = 0

    class _RateLimitError(Exception):
        status_code = 429

    def fake_create(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _RateLimitError("Too Many Requests")
        return types.SimpleNamespace(
            output_text='{"entries":[]}',
            status="completed",
            incomplete_details=None,
        )

    api = types.SimpleNamespace(create=fake_create)
    result = _call_structured_with_retry(
        api,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        "model",
        "sys",
        "user",
        name="probe-schema",
        schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    assert result.output_text == '{"entries":[]}'
    assert result.status == "completed"
    assert call_count == 3


def test_call_with_retry_raises_on_non_retryable(monkeypatch):
    """Should raise immediately on non-retryable errors."""
    from datasheetindex.llm.client import _call_with_retry

    monkeypatch.setattr("datasheetindex.llm.client._RETRY_BASE_DELAY", 0.01)

    def fake_create(*, model, instructions, input):
        raise ValueError("bad input")

    api = types.SimpleNamespace(create=fake_create)
    with pytest.raises(ValueError, match="bad input"):
        _call_with_retry(api, "model", "sys", "user")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_call_with_retry_raises_after_max_attempts(monkeypatch):
    """Should raise after exhausting all retry attempts."""
    from datasheetindex.llm.client import _call_with_retry

    monkeypatch.setattr("datasheetindex.llm.client._RETRY_BASE_DELAY", 0.01)

    class _RateLimitError(Exception):
        status_code = 429

    def fake_create(*, model, instructions, input):
        raise _RateLimitError("Too Many Requests")

    api = types.SimpleNamespace(create=fake_create)
    with pytest.raises(_RateLimitError):
        _call_with_retry(api, "model", "sys", "user")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


@pytest.mark.usefixtures("_has_env")
@pytest.mark.integration
def test_create_llm_client_integration():
    """Integration: verify a simple LLM call returns non-empty text."""
    from datasheetindex.llm.client import close_llm_client, create_llm_client

    llm = create_llm_client()
    try:
        result = llm("You are a helpful assistant.", "Say hello in one word.")
        assert isinstance(result, str)
        assert len(result) > 0
    finally:
        close_llm_client(llm)
