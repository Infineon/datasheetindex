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


class _FakeResponses:
    def create(self, **kwargs):
        raise AssertionError("describe_image must not use the Responses API")


class _FakeChat:
    """Records the request and returns one chat-completion-shaped reply."""

    def __init__(
        self,
        content: str | None = "a schematic of the output stage",
        *,
        finish_reason: str | None = "stop",
        choices: int = 1,
    ):
        self.captured: dict = {}
        self._content = content
        self._finish_reason = finish_reason
        self._choices = choices

    def create(self, **kwargs):
        self.captured.update(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=self._content),
                    finish_reason=self._finish_reason,
                )
            ]
            * self._choices
        )


def _patch_fake_clients(
    monkeypatch,
    seen_httpx_kwargs: dict[str, object],
    seen_openai_kwargs: dict[str, object],
    httpx_clients: list[_TrackedHttpxClient] | None = None,
) -> _FakeChat:
    """Install fake ``httpx``/``openai`` modules; return the chat fake.

    The chat fake is returned rather than discarded so a test can assert what
    ``create_llm_client`` wired into it. Before it was real, ``chat.completions``
    was a stub returning ``None`` -- which meant no test had ever reached
    ``describe_image`` through the factory, and the two lines that connect the
    vision path could both have been deleted with the suite still green.
    """
    chat = _FakeChat()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            seen_openai_kwargs.clear()
            seen_openai_kwargs.update(kwargs)
            self.responses = types.SimpleNamespace(
                create=lambda **_kwargs: types.SimpleNamespace(output_text="ok")
            )
            self.chat = types.SimpleNamespace(completions=chat)

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
    return chat


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
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_kwargs: None)
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


def test_get_vision_client_detects_describe_image():
    from datasheetindex.llm.client import get_vision_client

    class WithVision:
        def describe_image(self, system, image_base64, *, media_type="image/png"):
            return "a block diagram"

    class WithoutVision:
        def __call__(self, system, user):
            return "text"

    assert get_vision_client(WithVision()) is not None
    assert get_vision_client(WithoutVision()) is None
    assert get_vision_client(None) is None


def _image_part(captured: dict) -> dict:
    user = next(m for m in captured["messages"] if m["role"] == "user")
    return next(p for p in user["content"] if p["type"] == "image_url")


def test_describe_image_uses_chat_completions_not_responses():
    # Not a style choice. On a LiteLLM gateway the Responses API is a bridge
    # for any non-OpenAI model, and that bridge can file the model's answer as
    # a reasoning item -- which output_text ignores, yielding an empty caption
    # with the text sitting in the payload. Measured against qwen3.6-27b over
    # 16 real figure regions: 8 to 12 of 16 empty over five runs, a different
    # subset each run. Chat Completions bypasses the bridge: 0 empty in 144.
    #
    # image_url is an OBJECT here. On the Responses API it is a plain string;
    # the wrong form type-checks and fails at the gateway.
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat()
    client = _ManagedLlmClient(_FakeResponses(), object(), "gpt-4.1", chat_api=chat)
    result = client.describe_image("describe it", "QUJD", media_type="image/png")

    assert result == "a schematic of the output stage"
    system = next(m for m in chat.captured["messages"] if m["role"] == "system")
    user = next(m for m in chat.captured["messages"] if m["role"] == "user")
    text_part = next(p for p in user["content"] if p["type"] == "text")
    assert system["content"] == "describe it"
    assert text_part["text"] == "describe it"
    assert _image_part(chat.captured)["image_url"] == {
        "url": "data:image/png;base64,QUJD",
        "detail": "high",
    }


def test_describe_image_requests_high_detail():
    # Measured on the PCN's page-5 table: at "low" (512x512 downscale) the
    # model confidently invented row headings it never received; at "high"
    # it returned 19 of 20 verbatim correct. This guard stops "detail"
    # silently regressing back to "low", which would resume the
    # fabrication -- a prompt fix alone cannot bound what the model can
    # actually read.
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(
        _FakeResponses(), object(), "gpt-4.1", chat_api=chat
    ).describe_image("s", "QUJD")

    assert _image_part(chat.captured)["image_url"]["detail"] == "high"


def test_describe_image_caps_output_tokens():
    # A model that ignores the prompt's 60-word limit is the reason: qwen
    # answered a 128-pin pinout by listing all 128 pins (667 tokens) where
    # gpt-4.1 used 134. The cap is above every compliant answer measured, so
    # it binds on runaways only.
    from datasheetindex.llm.client import VISION_MAX_TOKENS, _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(
        _FakeResponses(), object(), "gpt-4.1", chat_api=chat
    ).describe_image("s", "QUJD")

    assert chat.captured["max_tokens"] == VISION_MAX_TOKENS


def test_describe_image_returns_empty_string_when_the_model_says_nothing():
    # The SDK types content as str | None. None must not reach the caller as
    # the literal "None": caption_figures_in_place strips the reply and treats
    # a blank one as a failed call, which is the correct outcome here.
    from datasheetindex.llm.client import _ManagedLlmClient

    client = _ManagedLlmClient(
        _FakeResponses(), object(), "gpt-4.1", chat_api=_FakeChat(content=None)
    )

    assert client.describe_image("s", "QUJD") == ""


def test_describe_image_uses_the_vision_model_when_one_is_configured():
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(
        _FakeResponses(), object(), "gpt-4.1", chat_api=chat, vision_model="qwen"
    ).describe_image("s", "QUJD")

    assert chat.captured["model"] == "qwen"


def test_describe_image_follows_the_text_model_without_a_vision_model():
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(
        _FakeResponses(), object(), "gpt-4.1", chat_api=chat
    ).describe_image("s", "QUJD")

    assert chat.captured["model"] == "gpt-4.1"


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("qwen3.6-27b", "qwen3.6-27b"),
        # Unset and empty must both mean "follow the text model". An empty
        # value is what an absent entry in a Kubernetes ConfigMap or a
        # commented-out .env line leaves behind, and it must be a no-op rather
        # than a request for the model named "".
        ("", None),
        ("   ", None),
    ],
)
def test_vision_model_env_reads_the_knob(monkeypatch, env_value, expected):
    from datasheetindex.llm.client import vision_model_env

    monkeypatch.setenv("DATASHEETINDEX_VISION_MODEL", env_value)
    assert vision_model_env() == expected


def test_vision_model_env_is_none_when_unset(monkeypatch):
    from datasheetindex.llm.client import vision_model_env

    monkeypatch.delenv("DATASHEETINDEX_VISION_MODEL", raising=False)
    assert vision_model_env() is None


def test_create_llm_client_wires_chat_completions_and_the_vision_model(monkeypatch):
    # The join between the two halves, which each have their own tests and
    # neither of which covers this: that the factory hands _ManagedLlmClient
    # ``client.chat.completions`` (not ``client.responses``) and the resolved
    # env knob. Both are single lines a refactor can drop silently.
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("DATASHEETINDEX_VISION_MODEL", "qwen3.6-27b")
    _install_fake_dotenv(monkeypatch)
    chat = _patch_fake_clients(monkeypatch, {}, {})

    from datasheetindex.llm.client import create_llm_client, get_vision_client

    vision = get_vision_client(create_llm_client(model="gpt-4.1"))
    assert vision is not None
    vision.describe_image("describe it", "QUJD")

    assert chat.captured["model"] == "qwen3.6-27b"
    assert _image_part(chat.captured)["image_url"]["detail"] == "high"


def test_create_llm_client_vision_follows_the_text_model_without_the_knob(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.delenv("DATASHEETINDEX_VISION_MODEL", raising=False)
    _install_fake_dotenv(monkeypatch)
    chat = _patch_fake_clients(monkeypatch, {}, {})

    from datasheetindex.llm.client import create_llm_client, get_vision_client

    vision = get_vision_client(create_llm_client(model="gpt-4.1"))
    assert vision is not None
    vision.describe_image("describe it", "QUJD")

    assert chat.captured["model"] == "gpt-4.1"


def test_describe_image_logs_why_a_caption_came_back_empty(caplog):
    # The silence is what let the transport bug hide: an empty reply and a
    # render failure both reached the caller as "failed". The two things that
    # tell the causes apart are the model and finish_reason, so both must be in
    # the message. "length" with no text is the reasoning-model case --
    # gpt-5-mini billed 300 of 300 tokens as reasoning and wrote nothing.
    import logging

    from datasheetindex.llm.client import _ManagedLlmClient

    client = _ManagedLlmClient(
        _FakeResponses(),
        object(),
        "gpt-4.1",
        chat_api=_FakeChat(content=None, finish_reason="length"),
        vision_model="gpt-5-mini",
    )

    with caplog.at_level(logging.WARNING, logger="datasheetindex.llm.client"):
        assert client.describe_image("s", "QUJD") == ""

    assert "gpt-5-mini" in caplog.text
    assert "length" in caplog.text


def test_describe_image_returns_empty_when_the_gateway_sends_no_choices(caplog):
    # A content filter or an error envelope can return a well-formed 200 with
    # an empty choices list. The caller already degrades an empty caption to a
    # failed call; an IndexError would instead surface a traceback whose text
    # says nothing about captioning.
    import logging

    from datasheetindex.llm.client import _ManagedLlmClient

    client = _ManagedLlmClient(
        _FakeResponses(), object(), "gpt-4.1", chat_api=_FakeChat(choices=0)
    )

    with caplog.at_level(logging.WARNING, logger="datasheetindex.llm.client"):
        assert client.describe_image("s", "QUJD") == ""

    assert "no choices" in caplog.text


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
