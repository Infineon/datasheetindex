"""Hermetic tests for Chat Completions settings and payload-free telemetry."""

from __future__ import annotations

import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict

import pytest

from datasheetindex.llm.client import (
    VISION_MAX_TOKENS,
    ChatRequestOptions,
    _ManagedLlmClient,
    create_llm_client,
    text_request_options_env,
    vision_request_options_env,
)
from tests.test_llm_client import _FakeChat, _patch_fake_clients


def invoke(client, operation):
    if operation == "text":
        return client("private prompt", "private document")
    if operation == "structured":
        return client.structured_json(
            "private prompt", "private document", name="toc", schema={"type": "object"}
        )
    return client.describe_image("private prompt", "private image")


@pytest.mark.parametrize("operation", ["text", "structured", "vision"])
def test_default_options_are_modern_private_and_do_not_guess_reasoning(operation):
    chat = _FakeChat()
    invoke(_ManagedLlmClient(chat, object(), "arbitrary-model"), operation)
    assert chat.captured["store"] is False
    assert "reasoning_effort" not in chat.captured
    assert "max_tokens" not in chat.captured
    if operation == "vision":
        assert chat.captured["max_completion_tokens"] == VISION_MAX_TOKENS
    else:
        assert "max_completion_tokens" not in chat.captured


@pytest.mark.parametrize("operation", ["text", "structured", "vision"])
def test_explicit_legacy_options_apply_without_an_automatic_retry(operation):
    chat = _FakeChat()
    options = ChatRequestOptions(
        reasoning_effort="none",
        max_output_tokens=123,
        token_limit_parameter="max_tokens",
        store=None,
    )
    client = _ManagedLlmClient(
        chat, object(), "arbitrary-model", text_options=options, vision_options=options
    )
    invoke(client, operation)
    assert chat.captured["max_tokens"] == 123
    assert chat.captured["reasoning_effort"] == "none"
    assert "max_completion_tokens" not in chat.captured
    assert "store" not in chat.captured
    assert len(chat.requests) == 1


def test_text_reasoning_does_not_leak_into_vision_even_with_the_same_model():
    chat = _FakeChat()
    client = _ManagedLlmClient(
        chat,
        object(),
        "same-model",
        text_options=ChatRequestOptions(reasoning_effort="high"),
    )
    invoke(client, "text")
    invoke(client, "structured")
    invoke(client, "vision")
    assert chat.requests[0]["reasoning_effort"] == "high"
    assert chat.requests[1]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in chat.requests[2]


def test_vision_reasoning_and_budget_are_independent_of_text():
    chat = _FakeChat()
    client = _ManagedLlmClient(
        chat,
        object(),
        "text-model",
        vision_model="vision-model",
        vision_options=ChatRequestOptions(
            reasoning_effort="none", max_output_tokens=450
        ),
    )
    invoke(client, "vision")
    invoke(client, "text")
    assert chat.requests[0]["model"] == "vision-model"
    assert chat.requests[0]["reasoning_effort"] == "none"
    assert chat.requests[0]["max_completion_tokens"] == 450
    assert "reasoning_effort" not in chat.requests[1]
    assert "max_completion_tokens" not in chat.requests[1]


def test_an_explicit_vision_options_object_still_has_the_default_safety_cap():
    chat = _FakeChat()
    client = _ManagedLlmClient(
        chat,
        object(),
        "model",
        vision_options=ChatRequestOptions(reasoning_effort="none"),
    )
    invoke(client, "vision")
    assert chat.captured["max_completion_tokens"] == VISION_MAX_TOKENS


def test_structured_per_call_limit_overrides_configured_limit_without_mutating_it():
    chat = _FakeChat()
    client = _ManagedLlmClient(
        chat, object(), "model", text_options=ChatRequestOptions(max_output_tokens=800)
    )
    client.structured_json("s", "u", name="n", schema={}, max_output_tokens=123)
    invoke(client, "structured")
    assert [r["max_completion_tokens"] for r in chat.requests] == [123, 800]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "300"])
def test_invalid_output_limits_are_rejected_before_any_request(value):
    with pytest.raises(ValueError, match="max_output_tokens"):
        ChatRequestOptions(max_output_tokens=value)
    chat = _FakeChat()
    client = _ManagedLlmClient(chat, object(), "model")
    with pytest.raises(ValueError, match="max_output_tokens"):
        client.structured_json("s", "u", name="n", schema={}, max_output_tokens=value)
    assert not chat.requests


@pytest.mark.parametrize(
    "variable,value",
    [
        ("DATASHEETINDEX_REASONING_EFFORT", "automatic"),
        ("DATASHEETINDEX_VISION_REASONING_EFFORT", "automatic"),
        ("DATASHEETINDEX_MAX_OUTPUT_TOKENS", "0"),
        ("DATASHEETINDEX_VISION_MAX_OUTPUT_TOKENS", "-10"),
        ("DATASHEETINDEX_VISION_MAX_OUTPUT_TOKENS", "300.5"),
        ("DATASHEETINDEX_TOKEN_LIMIT_PARAMETER", "max_output_tokens"),
        ("DATASHEETINDEX_LLM_STORE", "maybe"),
    ],
)
def test_factory_validates_environment_before_allocating_http_clients(
    monkeypatch, variable, value
):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv(variable, value)
    allocated = []
    _patch_fake_clients(monkeypatch, {}, {}, allocated)
    with pytest.raises(ValueError, match=variable):
        create_llm_client()
    assert allocated == []


@pytest.mark.parametrize(
    "reader", [text_request_options_env, vision_request_options_env]
)
def test_option_readers_load_dotenv_before_any_client_exists(monkeypatch, reader):
    def load(*args, **kwargs):
        monkeypatch.setenv("DATASHEETINDEX_REASONING_EFFORT", "low")
        monkeypatch.setenv("DATASHEETINDEX_VISION_REASONING_EFFORT", "none")

    monkeypatch.setitem(sys.modules, "dotenv", types.SimpleNamespace(load_dotenv=load))
    expected = "low" if reader is text_request_options_env else "none"
    assert reader().reasoning_effort == expected


def test_factory_wires_environment_and_callback_for_each_role(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("DATASHEETINDEX_REASONING_EFFORT", " HIGH ")
    monkeypatch.setenv("DATASHEETINDEX_VISION_REASONING_EFFORT", " NONE ")
    monkeypatch.setenv("DATASHEETINDEX_MAX_OUTPUT_TOKENS", "900")
    monkeypatch.setenv("DATASHEETINDEX_VISION_MAX_OUTPUT_TOKENS", "400")
    monkeypatch.setenv("DATASHEETINDEX_TOKEN_LIMIT_PARAMETER", "max_tokens")
    monkeypatch.setenv("DATASHEETINDEX_LLM_STORE", "omit")
    chat = _patch_fake_clients(monkeypatch, {}, {})
    events = []
    client = create_llm_client(on_completion=events.append)
    for operation in ("text", "structured", "vision"):
        invoke(client, operation)
    assert [r["reasoning_effort"] for r in chat.requests] == ["high", "high", "none"]
    assert [r["max_tokens"] for r in chat.requests] == [900, 900, 400]
    assert all("store" not in r for r in chat.requests)
    assert [e.operation for e in events] == ["text", "structured", "vision"]


def test_explicit_options_override_even_invalid_environment_for_that_role(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("DATASHEETINDEX_REASONING_EFFORT", "invalid")
    monkeypatch.setenv("DATASHEETINDEX_VISION_REASONING_EFFORT", "none")
    chat = _patch_fake_clients(monkeypatch, {}, {})
    client = create_llm_client(text_options=ChatRequestOptions(reasoning_effort="low"))
    invoke(client, "text")
    invoke(client, "vision")
    assert [r["reasoning_effort"] for r in chat.requests] == ["low", "none"]


@pytest.mark.parametrize(
    "value,expected",
    [("", False), (" FALSE ", False), ("true", True), (" OMIT ", None)],
)
def test_storage_environment_is_explicit_and_normalized(monkeypatch, value, expected):
    monkeypatch.setenv("DATASHEETINDEX_LLM_STORE", value)
    assert text_request_options_env().store is expected
    assert vision_request_options_env().store is expected


@pytest.mark.parametrize("operation", ["text", "structured", "vision"])
def test_metadata_reports_usage_without_document_payloads(operation):
    class WithMetadata(_FakeChat):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            response.model = "provider-snapshot"
            response._request_id = "request-123"
            response.usage = types.SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=30,
                completion_tokens_details=types.SimpleNamespace(reasoning_tokens=12),
            )
            return response

    events = []
    client = _ManagedLlmClient(
        WithMetadata(content="private answer"),
        object(),
        "alias",
        on_completion=events.append,
    )
    invoke(client, operation)
    assert len(events) == 1
    event = events[0]
    assert event.operation == operation
    assert event.requested_model == "alias"
    assert event.response_model == "provider-snapshot"
    assert event.request_id == "request-123"
    assert event.finish_reason == "stop"
    assert (event.prompt_tokens, event.completion_tokens, event.reasoning_tokens) == (
        100,
        30,
        12,
    )
    assert event.elapsed_seconds >= 0
    assert not event.empty and not event.refused
    assert "private" not in json.dumps(asdict(event))
    with pytest.raises(FrozenInstanceError):
        event.requested_model = "changed"


@pytest.mark.parametrize(
    "choices,content,refusal",
    [(0, None, None), (1, "", None), (1, None, "private refusal")],
)
def test_empty_and_refused_responses_still_emit_metadata(choices, content, refusal):
    events = []
    client = _ManagedLlmClient(
        _FakeChat(choices=choices, content=content, refusal=refusal),
        object(),
        "alias",
        on_completion=events.append,
    )
    assert client("s", "u") == ""
    event = events[0]
    assert event.empty
    assert event.refused is bool(refusal)
    assert event.prompt_tokens is None
    assert event.completion_tokens is None
    assert event.reasoning_tokens is None
    assert event.response_model is None
    assert event.request_id is None
    assert "private refusal" not in repr(event)


def test_callback_failures_do_not_break_success_or_log_callback_payloads(caplog):
    def broken(event):
        raise RuntimeError("private document data")

    client = _ManagedLlmClient(
        _FakeChat(content="ok"), object(), "alias", on_completion=broken
    )
    assert client("s", "u") == "ok"
    assert "callback failed" in caplog.text
    assert "private document data" not in caplog.text


def test_caption_whitespace_is_normalized_without_rewriting_content():
    client = _ManagedLlmClient(
        _FakeChat(content="  Pinout.\n\n Labels:\tVCC, GND.  "), object(), "model"
    )
    assert client.describe_image("s", "QUJD") == "Pinout. Labels: VCC, GND."


def test_plain_text_truncation_cannot_be_cached_as_a_completed_summary():
    from datasheetindex.llm.summarizer import add_summaries
    from tests.test_summarizer import SAMPLE_TEXT, _make_nodes

    nodes = _make_nodes()
    events = []
    client = _ManagedLlmClient(
        _FakeChat(content="Supply voltage ranges from", finish_reason="length"),
        object(),
        "model",
        text_options=ChatRequestOptions(max_output_tokens=5),
        on_completion=events.append,
    )
    with pytest.raises(RuntimeError, match="truncated"):
        add_summaries(nodes, SAMPLE_TEXT, client)
    assert nodes[0].summary == ""
    assert len(events) == 1
    assert events[0].finish_reason == "length"


def test_caption_truncation_still_preserves_the_identifying_prefix():
    client = _ManagedLlmClient(
        _FakeChat(
            content="Table: Supply voltage, Input current", finish_reason="length"
        ),
        object(),
        "model",
    )
    assert client.describe_image("s", "QUJD") == "Table: Supply voltage, Input current"


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_factory_closes_owned_http_when_sdk_initialization_fails(
    monkeypatch, error_type
):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    allocated = []
    _patch_fake_clients(monkeypatch, {}, {}, allocated)
    error = error_type("SDK initialization failed")

    class BrokenSdk:
        def __init__(self, **kwargs):
            raise error

    monkeypatch.setattr(sys.modules["openai"], "OpenAI", BrokenSdk)
    with pytest.raises(error_type) as caught:
        create_llm_client()
    assert caught.value is error
    assert len(allocated) == 1
    assert allocated[0].closed


def test_concurrent_caption_calls_deliver_separate_metadata_records():
    events = []
    client = _ManagedLlmClient(
        _FakeChat(), object(), "alias", on_completion=events.append
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(
            pool.map(lambda _: client.describe_image("s", "QUJD"), range(12))
        )
    assert len(events) == len(replies) == 12
    assert len({id(e) for e in events}) == 12
    assert all(e.operation == "vision" and e.requested_model == "alias" for e in events)
