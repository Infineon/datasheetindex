"""Tests for the LLM client factory."""

from __future__ import annotations

import os
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


class _TrackedHttpx2Client:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ForbiddenResponses:
    """Wired in wherever a real ``client.responses`` would be.

    Nothing in this package may use the Responses API any more -- text,
    structured and vision all go over Chat Completions. Failing loudly here
    turns that from a convention into something the suite enforces, so a
    partial revert cannot pass green.
    """

    def create(self, **kwargs):
        raise AssertionError("no call may use the Responses API")


def _chat_reply(
    content: str | None,
    *,
    finish_reason: str | None = "stop",
    choices: int = 1,
    refusal: str | None = None,
):
    """One chat-completion-shaped response object.

    ``refusal`` is a real field on the OpenAI message object and defaults to
    ``None`` on every ordinary reply, which is what it does here.
    """
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content, refusal=refusal),
                finish_reason=finish_reason,
            )
        ]
        * choices
    )


class _FakeChat:
    """Records every request and returns one chat-completion-shaped reply.

    ``requests`` keeps them in order and ``captured`` is the most recent. Both
    matter now that text and vision share this transport: a test that makes a
    text call and then a caption call has to be able to tell the two requests
    apart, which a single merged dict cannot do -- it would report the vision
    model for both.
    """

    def __init__(
        self,
        content: str | None = "a schematic of the output stage",
        *,
        finish_reason: str | None = "stop",
        choices: int = 1,
        refusal: str | None = None,
    ):
        self.requests: list[dict] = []
        self.captured: dict = {}
        self._content = content
        self._finish_reason = finish_reason
        self._choices = choices
        self._refusal = refusal

    def create(self, **kwargs):
        self.requests.append(kwargs)
        self.captured = kwargs
        return _chat_reply(
            self._content,
            finish_reason=self._finish_reason,
            choices=self._choices,
            refusal=self._refusal,
        )


def _patch_fake_clients(
    monkeypatch,
    seen_httpx2_kwargs: dict[str, object],
    seen_openai_kwargs: dict[str, object],
    httpx2_clients: list[_TrackedHttpx2Client] | None = None,
) -> _FakeChat:
    """Install fake ``httpx2``/``openai`` modules; return the chat fake.

    The chat fake is returned rather than discarded so a test can assert what
    ``create_llm_client`` wired into it. Before it was real, ``chat.completions``
    was a stub returning ``None`` -- which meant no test had ever reached
    ``describe_image`` through the factory, and the two lines that connect the
    vision path could both have been deleted with the suite still green.

    Every request is now observable through it, text included. ``client.responses``
    is wired to a fake that raises, so a call that reaches for the old transport
    fails rather than quietly working.
    """

    chat = _FakeChat()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            seen_openai_kwargs.clear()
            seen_openai_kwargs.update(kwargs)
            self.responses = _ForbiddenResponses()
            self.chat = types.SimpleNamespace(completions=chat)

    def _fake_httpx2_client(**kwargs):
        seen_httpx2_kwargs.clear()
        seen_httpx2_kwargs.update(kwargs)
        client = _TrackedHttpx2Client()
        if httpx2_clients is not None:
            httpx2_clients.append(client)
        return client

    fake_httpx2 = types.SimpleNamespace(Client=_fake_httpx2_client)
    fake_openai = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "httpx2", fake_httpx2)
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


def _verify_arg_for(monkeypatch, tls_verify: str | None) -> object:
    """Build a client with ``LITELLM_TLS_VERIFY`` set (or unset) and report
    the ``verify=`` argument that reached ``httpx2.Client``."""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    if tls_verify is None:
        monkeypatch.delenv("LITELLM_TLS_VERIFY", raising=False)
    else:
        monkeypatch.setenv("LITELLM_TLS_VERIFY", tls_verify)
    _install_fake_dotenv(monkeypatch)

    seen_httpx2_kwargs: dict[str, object] = {}
    _seen_openai_kwargs: dict[str, object] = {}
    _patch_fake_clients(monkeypatch, seen_httpx2_kwargs, _seen_openai_kwargs)

    from datasheetindex.llm.client import create_llm_client

    llm = create_llm_client()
    assert callable(llm)
    return seen_httpx2_kwargs["verify"]


def test_create_llm_client_tls_verify_defaults_true(monkeypatch):
    """An unset ``LITELLM_TLS_VERIFY`` must verify the certificate.

    This is the security-relevant direction and the one that regressed: the
    default used to be ``False``, so an ordinary install shipped the master
    key to the gateway over an unauthenticated channel with nothing in the
    call, the log or the result saying so. Asserting on the ``verify=``
    argument that reaches ``httpx2.Client`` -- rather than on
    ``_parse_tls_verify_env`` alone -- keeps the wiring in scope, since a
    correct parse that is never passed through fixes nothing.
    """
    assert _verify_arg_for(monkeypatch, None) is True


@pytest.mark.parametrize("spelling", ["0", "false", "no", "off"])
def test_create_llm_client_tls_verify_opt_out_spellings(monkeypatch, spelling):
    """Every documented opt-out spelling must still disable verification.

    An escape hatch that honours one spelling and silently ignores the others
    looks broken to exactly the reader who needs it -- a self-signed internal
    gateway is the reason this knob exists, and that reader has no way to tell
    "ignored" from "did not help".
    """
    assert _verify_arg_for(monkeypatch, spelling) is False


@pytest.mark.parametrize("spelling", ["FALSE", "Off", " no "])
def test_create_llm_client_tls_verify_opt_out_is_case_and_space_tolerant(
    monkeypatch, spelling
):
    assert _verify_arg_for(monkeypatch, spelling) is False


@pytest.mark.parametrize("spelling", ["true", "1", "yes", "", "  "])
def test_create_llm_client_tls_verify_anything_else_verifies(monkeypatch, spelling):
    """Only the four opt-out words turn verification off.

    An empty or whitespace value is the accident case -- ``export
    LITELLM_TLS_VERIFY=`` in a shell profile -- and it must fail towards
    verifying rather than away from it.
    """
    assert _verify_arg_for(monkeypatch, spelling) is True


def test_create_llm_client_timeout_and_retries_defaults(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.delenv("LITELLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LITELLM_MAX_RETRIES", raising=False)
    _install_fake_dotenv(monkeypatch)

    seen_httpx2_kwargs: dict[str, object] = {}
    seen_openai_kwargs: dict[str, object] = {}
    _patch_fake_clients(monkeypatch, seen_httpx2_kwargs, seen_openai_kwargs)

    from datasheetindex.llm.client import (
        DEFAULT_MAX_RETRIES,
        DEFAULT_TIMEOUT_SECONDS,
        create_llm_client,
    )

    llm = create_llm_client()
    assert callable(llm)
    assert seen_httpx2_kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS
    assert seen_openai_kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS
    assert seen_openai_kwargs["max_retries"] == DEFAULT_MAX_RETRIES


def test_create_llm_client_timeout_and_retries_override(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("LITELLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LITELLM_MAX_RETRIES", "5")
    _install_fake_dotenv(monkeypatch)

    seen_httpx2_kwargs: dict[str, object] = {}
    seen_openai_kwargs: dict[str, object] = {}
    _patch_fake_clients(monkeypatch, seen_httpx2_kwargs, seen_openai_kwargs)

    from datasheetindex.llm.client import create_llm_client

    llm = create_llm_client()
    assert callable(llm)
    assert seen_httpx2_kwargs["timeout"] == 12.5
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


def test_close_llm_client_closes_httpx2_client(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    _install_fake_dotenv(monkeypatch)

    seen_httpx2_kwargs: dict[str, object] = {}
    seen_openai_kwargs: dict[str, object] = {}
    httpx2_clients: list[_TrackedHttpx2Client] = []
    _patch_fake_clients(
        monkeypatch,
        seen_httpx2_kwargs,
        seen_openai_kwargs,
        httpx2_clients=httpx2_clients,
    )

    from datasheetindex.llm.client import close_llm_client, create_llm_client

    llm = create_llm_client()
    close_llm_client(llm)

    assert len(httpx2_clients) == 1
    assert httpx2_clients[0].closed is True


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
            self.responses = _ForbiddenResponses()
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kwargs: (
                        seen_request.update(kwargs) or _chat_reply('{"entries":[]}')
                    )
                )
            )

    def _fake_httpx2_client(**_kwargs):
        return _TrackedHttpx2Client()

    monkeypatch.setitem(
        sys.modules,
        "httpx2",
        types.SimpleNamespace(Client=_fake_httpx2_client),
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

    # Chat Completions nests the schema one level deeper than the Responses API
    # did (``response_format.json_schema.name``, not ``text.format.name``), and
    # the wrong nesting is accepted by the SDK and rejected at the gateway.
    assert seen_request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "probe-schema",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    # max_tokens, not max_output_tokens: the spelling vLLM is certain to know.
    assert seen_request["max_tokens"] == 123
    assert result.output_text == '{"entries":[]}'
    assert result.status == "completed"


def test_call_with_retry_retries_on_429(monkeypatch):
    """Should retry on rate limit errors and succeed."""
    from datasheetindex.llm.client import _call_with_retry

    monkeypatch.setattr("datasheetindex.llm.client._RETRY_BASE_DELAY", 0.01)

    call_count = 0

    class _RateLimitError(Exception):
        status_code = 429

    def fake_create(*, model, messages):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise _RateLimitError("Too Many Requests")
        return _chat_reply("success")

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
        return _chat_reply('{"entries":[]}')

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

    def fake_create(*, model, messages):
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

    def fake_create(*, model, messages):
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
    client = _ManagedLlmClient(chat, object(), "gpt-4.1")
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


def test_text_calls_send_the_system_prompt_exactly_once():
    # describe_image deliberately sends the prompt twice (system role AND an
    # input_text part) -- parity with the Responses shape it replaced, and its
    # measured token counts include it. That is vision-specific. Copying it to
    # the text path would silently inflate every ToC-fallback and summary call,
    # which on a 15-chunk datasheet is 15 duplicated system prompts.
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat(content="ok")
    _ManagedLlmClient(chat, object(), "gpt-4.1")("the system prompt", "the user text")

    messages = chat.captured["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "the system prompt"
    assert messages[1]["content"] == "the user text"


def test_text_calls_log_when_the_reply_comes_back_empty(caplog):
    # The point of the whole exercise. A well-formed 200 carrying no text used
    # to reach the ToC fallback as "", which it scores as "no entries" and
    # answers by retrying the entire document over the same transport -- two
    # full passes, no ToC, and one log line that says "retrying" and nothing
    # about why. The model and finish_reason are what separate the causes.
    import logging

    from datasheetindex.llm.client import _ManagedLlmClient

    client = _ManagedLlmClient(
        _FakeChat(content="", finish_reason="length"), object(), "some-gateway-model"
    )

    with caplog.at_level(logging.WARNING, logger="datasheetindex.llm.client"):
        assert client("s", "u") == ""

    assert "some-gateway-model" in caplog.text
    assert "length" in caplog.text


def test_a_refusal_is_logged_as_a_refusal_and_not_as_an_empty_reply(caplog):
    # A refusal arrives as content=None with the reason in `refusal` and
    # finish_reason="stop". Without checking it first, the blank-content branch
    # logs "came back empty (finish_reason=stop)" -- which names the wrong cause
    # in the one message whose whole job is to name the right one. OpenAI's
    # guidance is to inspect `refusal` before parsing content, since a refusal
    # deliberately does not follow the supplied schema.
    import logging

    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat(content=None, refusal="I can't help with that.")
    client = _ManagedLlmClient(chat, object(), "gpt-4.1")

    with caplog.at_level(logging.WARNING, logger="datasheetindex.llm.client"):
        assert client("s", "u") == ""

    assert "refused" in caplog.text
    assert "I can't help with that." in caplog.text
    # The misleading message must not also fire.
    assert "came back empty" not in caplog.text


def test_a_backend_that_omits_refusal_still_works():
    # `refusal` is OpenAI's field. An OpenAI-compatible backend such as vLLM
    # need not send it, so reading it must not require it to exist.
    from datasheetindex.llm.client import _ManagedLlmClient

    class _NoRefusalField:
        def create(self, **_kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ]
            )

    assert (
        _ManagedLlmClient(_NoRefusalField(), object(), "vllm-model")("s", "u") == "ok"
    )


def test_structured_calls_report_truncation_as_an_incomplete_status():
    # finish_reason is the only completion signal Chat Completions gives, so it
    # is what now feeds StructuredLlmResult.status. toc_fallback raises on a
    # non-"completed" status rather than parsing half a JSON document, and that
    # behaviour must survive the transport change.
    from datasheetindex.llm.client import _ManagedLlmClient

    client = _ManagedLlmClient(
        _FakeChat(content='{"entries": [', finish_reason="length"),
        object(),
        "gpt-4.1",
    )

    result = client.structured_json("s", "u", name="n", schema={"type": "object"})

    assert result.status == "incomplete"
    assert result.incomplete_details == {"reason": "length"}


def test_structured_calls_leave_status_unset_when_the_gateway_omits_it():
    # A gateway that reports no finish_reason says nothing about the payload,
    # and _parse_structured_chunk_response already treats None that way and
    # parses anyway. Mapping the absence to "incomplete" would discard every
    # good chunk such a gateway returns.
    from datasheetindex.llm.client import _ManagedLlmClient

    client = _ManagedLlmClient(
        _FakeChat(content='{"entries": []}', finish_reason=None),
        object(),
        "gpt-4.1",
    )

    result = client.structured_json("s", "u", name="n", schema={"type": "object"})

    assert result.status is None
    assert result.incomplete_details is None
    assert result.output_text == '{"entries": []}'


def test_describe_image_requests_high_detail():
    # Measured on the PCN's page-5 table: at "low" (512x512 downscale) the
    # model confidently invented row headings it never received; at "high"
    # it returned 19 of 20 verbatim correct. This guard stops "detail"
    # silently regressing back to "low", which would resume the
    # fabrication -- a prompt fix alone cannot bound what the model can
    # actually read.
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(chat, object(), "gpt-4.1").describe_image("s", "QUJD")

    assert _image_part(chat.captured)["image_url"]["detail"] == "high"


def test_describe_image_caps_output_tokens():
    # A model that ignores the prompt's 60-word limit is the reason: qwen
    # answered a 128-pin pinout by listing all 128 pins (667 tokens) where
    # gpt-4.1 used 134. The cap is above every compliant answer measured, so
    # it binds on runaways only.
    from datasheetindex.llm.client import VISION_MAX_TOKENS, _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(chat, object(), "gpt-4.1").describe_image("s", "QUJD")

    assert chat.captured["max_tokens"] == VISION_MAX_TOKENS


def test_describe_image_returns_empty_string_when_the_model_says_nothing():
    # The SDK types content as str | None. None must not reach the caller as
    # the literal "None": caption_figures_in_place strips the reply and treats
    # a blank one as a failed call, which is the correct outcome here.
    from datasheetindex.llm.client import _ManagedLlmClient

    client = _ManagedLlmClient(_FakeChat(content=None), object(), "gpt-4.1")

    assert client.describe_image("s", "QUJD") == ""


def test_describe_image_uses_the_vision_model_when_one_is_configured():
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(chat, object(), "gpt-4.1", vision_model="qwen").describe_image(
        "s", "QUJD"
    )

    assert chat.captured["model"] == "qwen"


def test_describe_image_follows_the_text_model_without_a_vision_model():
    from datasheetindex.llm.client import _ManagedLlmClient

    chat = _FakeChat()
    _ManagedLlmClient(chat, object(), "gpt-4.1").describe_image("s", "QUJD")

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


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("gpt-5-mini", "gpt-5-mini"),
        # Same contract as the vision knob: unset and empty both mean "use the
        # built-in default", so a commented-out .env line or a blank entry in
        # the MCP host's env block is a no-op rather than a request for the
        # model named "".
        ("", None),
        ("   ", None),
    ],
)
def test_text_model_env_reads_the_knob(monkeypatch, env_value, expected):
    from datasheetindex.llm.client import text_model_env

    monkeypatch.setenv("DATASHEETINDEX_MODEL", env_value)
    assert text_model_env() == expected


def test_text_model_env_is_none_when_unset(monkeypatch):
    from datasheetindex.llm.client import text_model_env

    monkeypatch.delenv("DATASHEETINDEX_MODEL", raising=False)
    assert text_model_env() is None


@pytest.mark.parametrize(
    "reader_name,variable",
    [
        ("text_model_env", "DATASHEETINDEX_MODEL"),
        ("vision_model_env", "DATASHEETINDEX_VISION_MODEL"),
    ],
)
def test_each_env_reader_sees_dotenv_without_a_client(
    monkeypatch, reader_name, variable
):
    """Each reader must load ``.env`` itself, with no client constructed first.

    ``_BuildOptions`` -- the artifact cache key -- calls both readers before any
    client exists, and ``load_dotenv`` used to run only inside the factory. The
    key therefore recorded ``None`` for a ``.env``-configured model while the
    build ran on the ``.env`` value.

    Parametrized per reader on purpose. The end-to-end test in
    ``tests/test_reuse.py`` cannot pin this one: the two readers are called on
    adjacent lines, so whichever runs first loads ``.env`` for the other, and
    dropping the load from just one of them leaves that test green.
    """
    import datasheetindex.llm.client as client_module

    def _load_dotenv(*_args, **_kwargs):
        if variable not in os.environ:
            monkeypatch.setenv(variable, "from-dotenv")

    monkeypatch.delenv(variable, raising=False)
    monkeypatch.setitem(
        sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=_load_dotenv)
    )

    reader = getattr(client_module, reader_name)
    assert reader() == "from-dotenv"


def test_create_llm_client_uses_the_text_model_env_when_no_model_is_named(monkeypatch):
    """The knob's whole point: the caller that names nothing is the common one.

    ``build_datasheet`` omits ``model`` unless the agent asks for summaries or
    judges the ToC poor, so the auto ToC fallback is what actually runs on the
    default -- and before this knob that default was a hardcoded ``gpt-4.1``
    with no deployment override anywhere. A gateway that does not serve
    ``gpt-4.1`` had no way to make the fallback work.
    """
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("DATASHEETINDEX_MODEL", "some-gateway-model")
    _install_fake_dotenv(monkeypatch)
    chat = _patch_fake_clients(monkeypatch, {}, {})

    from datasheetindex.llm.client import create_llm_client

    create_llm_client()("system", "user")

    assert chat.captured["model"] == "some-gateway-model"


def test_an_explicit_model_wins_over_the_text_model_env(monkeypatch):
    """Precedence is arg > env > default, and this is the half that can regress.

    The ``model`` tool argument is per-call and per-document; the env var is the
    deployment's default. Resolving the env var over an explicit argument would
    make ``build_datasheet(model=...)`` inert wherever the knob happens to be
    set, which is exactly where an agent asking for a different model is most
    likely to be running.
    """
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("DATASHEETINDEX_MODEL", "deployment-default")
    _install_fake_dotenv(monkeypatch)
    chat = _patch_fake_clients(monkeypatch, {}, {})

    from datasheetindex.llm.client import create_llm_client

    create_llm_client(model="gpt-5.2")("system", "user")

    assert chat.captured["model"] == "gpt-5.2"


def test_create_llm_client_falls_back_to_the_default_model(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.delenv("DATASHEETINDEX_MODEL", raising=False)
    _install_fake_dotenv(monkeypatch)
    chat = _patch_fake_clients(monkeypatch, {}, {})

    from datasheetindex.llm.client import DEFAULT_TEXT_MODEL, create_llm_client

    create_llm_client()("system", "user")

    assert chat.captured["model"] == DEFAULT_TEXT_MODEL


def test_the_vision_knob_still_wins_over_the_text_model_env(monkeypatch):
    """Two knobs, and the specific one must not be swallowed by the general one.

    With both set, text goes to one model and captions to the other. Nothing
    else pins that the vision resolution reads the *env* text model rather than
    the hardcoded default it used to fall back to.
    """
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("DATASHEETINDEX_MODEL", "text-only-model")
    monkeypatch.setenv("DATASHEETINDEX_VISION_MODEL", "qwen3.6-27b")
    _install_fake_dotenv(monkeypatch)
    chat = _patch_fake_clients(monkeypatch, {}, {})

    from datasheetindex.llm.client import create_llm_client, get_vision_client

    llm = create_llm_client()
    llm("system", "user")
    vision = get_vision_client(llm)
    assert vision is not None
    vision.describe_image("describe it", "QUJD")

    # Read per request, not from the merged view: both calls now go out over
    # the same chat transport, so the only thing separating them is order.
    text_request, vision_request = chat.requests
    assert text_request["model"] == "text-only-model"
    assert vision_request["model"] == "qwen3.6-27b"


def test_vision_follows_the_text_model_env_when_no_vision_model_is_set(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("DATASHEETINDEX_MODEL", "text-only-model")
    monkeypatch.delenv("DATASHEETINDEX_VISION_MODEL", raising=False)
    _install_fake_dotenv(monkeypatch)
    chat = _patch_fake_clients(monkeypatch, {}, {})

    from datasheetindex.llm.client import create_llm_client, get_vision_client

    vision = get_vision_client(create_llm_client())
    assert vision is not None
    vision.describe_image("describe it", "QUJD")

    assert chat.captured["model"] == "text-only-model"


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
        _FakeChat(content=None, finish_reason="length"),
        object(),
        "gpt-4.1",
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

    client = _ManagedLlmClient(_FakeChat(choices=0), object(), "gpt-4.1")

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


# --- TLS verification failures are named, not swallowed ----------------------
# A gateway whose certificate does not verify is the one LLM failure the
# transport describes uselessly: openai raises `APIConnectionError("Connection
# error.")` and the `ssl.SSLCertVerificationError` naming the real cause sits
# three links down the `__cause__` chain (openai -> httpx2 -> httpcore2 -> ssl,
# measured against a self-signed local server). Every caller of this client
# wraps its calls in a blanket `except Exception` that logs a warning, so
# without a distinguishable type the symptom is a silently empty ToC -- exactly
# what `LITELLM_TLS_VERIFY` defaulting to verify made possible for anyone whose
# gateway is fronted by a private CA.


def _cert_error(msg="certificate verify failed: self-signed certificate"):
    """Rebuild the real chain openai hands us on a verification failure."""
    import ssl

    root = ssl.SSLCertVerificationError(f"[SSL: CERTIFICATE_VERIFY_FAILED] {msg}")
    transport = ConnectionError("[SSL: CERTIFICATE_VERIFY_FAILED] " + msg)
    transport.__cause__ = root
    api = RuntimeError("Connection error.")
    api.__cause__ = transport
    return api


def test_tls_failure_is_found_through_the_whole_cause_chain():
    from datasheetindex.llm.client import _tls_verification_failure

    assert _tls_verification_failure(_cert_error()) is not None


def test_tls_failure_is_found_through_context_as_well_as_cause():
    """An `except`-and-raise without `from` links via `__context__`, not `__cause__`."""
    import ssl

    from datasheetindex.llm.client import _tls_verification_failure

    outer = RuntimeError("Connection error.")
    outer.__context__ = ssl.SSLCertVerificationError("nope")
    assert _tls_verification_failure(outer) is not None


def test_a_cyclic_cause_chain_terminates():
    """`raise ... from` can build a cycle; walking it must not hang the build."""
    from datasheetindex.llm.client import _tls_verification_failure

    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert _tls_verification_failure(a) is None


def test_a_severed_chain_is_not_followed_past_the_severing():
    """``raise X from None`` cuts the chain on purpose; the walk must respect it.

    Otherwise an unrelated error raised while handling a certificate failure is
    classified as one -- and that classification is neither retryable nor
    degradable, so the build would abort quoting a remedy that cannot fix it.
    """
    import ssl

    from datasheetindex.llm.client import _tls_verification_failure

    try:
        try:
            raise ssl.SSLCertVerificationError("original")
        except ssl.SSLCertVerificationError:
            raise ValueError("unrelated, and the chain is severed") from None
    except ValueError as exc:
        assert _tls_verification_failure(exc) is None


def test_an_ordinary_failure_is_not_mistaken_for_a_tls_one():
    from datasheetindex.llm.client import _tls_verification_failure

    assert _tls_verification_failure(RuntimeError("Connection error.")) is None


def _client_raising(exc):
    """A `_ManagedLlmClient` whose every gateway call raises `exc`."""
    from datasheetindex.llm.client import _ManagedLlmClient

    class _Boom:
        def create(self, **_kwargs):
            raise exc

    return _ManagedLlmClient(
        types.SimpleNamespace(completions=_Boom()).completions, object(), "gpt-4.1"
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda c: c("system", "user"), id="text"),
        pytest.param(
            lambda c: c.structured_json("s", "u", name="n", schema={"type": "object"}),
            id="structured",
        ),
        pytest.param(lambda c: c.describe_image("s", "QUJD"), id="vision"),
    ],
)
def test_every_call_shape_reports_a_tls_failure_as_such(call, monkeypatch):
    """Text, structured and vision all reach the gateway; all three must name it."""
    from datasheetindex.llm.client import LlmTlsVerificationError

    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    with pytest.raises(LlmTlsVerificationError) as excinfo:
        call(_client_raising(_cert_error()))

    message = str(excinfo.value)
    # The message has to carry the whole remedy: which endpoint failed, the
    # preferred fix, and the escape hatch. A named type with a bare "TLS error"
    # would move the problem rather than solve it.
    assert "gateway.example" in message
    assert "LITELLM_TLS_VERIFY" in message
    assert "trust store" in message


def test_the_original_transport_error_is_kept_as_the_cause(monkeypatch):
    """Nothing is hidden: the raw chain stays reachable for a traceback."""
    import ssl

    from datasheetindex.llm.client import LlmTlsVerificationError

    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    with pytest.raises(LlmTlsVerificationError) as excinfo:
        _client_raising(_cert_error())("s", "u")

    from datasheetindex.llm.client import _tls_verification_failure

    original = excinfo.value.__cause__
    assert isinstance(original, RuntimeError)
    # The ssl error is still reachable from the raised error, not just from the
    # one it wrapped -- a traceback shows the whole chain.
    assert isinstance(
        _tls_verification_failure(excinfo.value), ssl.SSLCertVerificationError
    )


def test_a_non_tls_failure_is_left_exactly_as_it_was(monkeypatch):
    """The narrowing must not swallow or reshape any other error."""
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/v1")
    boom = ValueError("something else entirely")
    with pytest.raises(ValueError, match="something else entirely"):
        _client_raising(boom)("s", "u")


def test_the_message_names_the_variable_when_the_url_is_unset(monkeypatch):
    """A client built from an explicit argument may leave the env var unset."""
    from datasheetindex.llm.client import LlmTlsVerificationError

    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    with pytest.raises(LlmTlsVerificationError, match="LITELLM_BASE_URL"):
        _client_raising(_cert_error())("s", "u")
