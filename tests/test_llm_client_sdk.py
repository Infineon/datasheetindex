"""Real OpenAI SDK over an in-memory HTTP transport; never contacts a gateway.

The sdk CI lane installs [llm] and asserts imports so this cannot pass by skipping.
Fakes alone cannot verify SDK retry counts or the actual serialized request body.
"""

from __future__ import annotations

import json

import pytest

from datasheetindex.llm.client import (
    LlmIncompleteResponseError,
    close_llm_client,
    create_llm_client,
    get_structured_output_client,
    get_vision_client,
)

httpx2 = pytest.importorskip("httpx2")
openai = pytest.importorskip("openai")


def _install_transport(monkeypatch, handler, retries):
    base = httpx2.Client

    class MemoryClient(base):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx2, "Client", MemoryClient)
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "test-only-key")
    monkeypatch.setenv("LITELLM_MAX_RETRIES", str(retries))


def _reply(*, finish_reason="stop", content='{"entries":[]}'):
    return httpx2.Response(
        200,
        headers={"x-request-id": "test-request"},
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "provider-snapshot",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        },
    )


def _invoke(client, operation):
    if operation == "text":
        return client("s", "u")
    if operation == "structured":
        structured = get_structured_output_client(client)
        assert structured is not None
        return structured.structured_json(
            "s", "u", name="toc", schema={"type": "object"}, max_output_tokens=500
        )
    vision = get_vision_client(client)
    assert vision is not None
    return vision.describe_image("s", "QUJD")


@pytest.mark.parametrize("operation", ["text", "structured", "vision"])
@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("retries", [0, 2])
def test_sdk_owns_the_exact_retry_budget_for_every_operation(
    monkeypatch, operation, status, retries
):
    calls = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        calls.append(json.loads(request.content))
        return httpx2.Response(
            status,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "test failure", "type": "test_error"}},
        )

    _install_transport(monkeypatch, handler, retries)
    client = create_llm_client()
    try:
        with pytest.raises(openai.APIStatusError):
            _invoke(client, operation)
    finally:
        close_llm_client(client)
    assert len(calls) == retries + 1
    assert all(body["store"] is False for body in calls)


@pytest.mark.parametrize("operation", ["text", "structured", "vision"])
def test_sdk_recovers_from_rate_limiting_and_emits_one_metadata_event(
    monkeypatch, operation
):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx2.Response(
                429,
                headers={"retry-after-ms": "1"},
                json={"error": {"message": "slow down"}},
            )
        return _reply()

    _install_transport(monkeypatch, handler, 2)
    events = []
    client = create_llm_client(on_completion=events.append)
    try:
        _invoke(client, operation)
    finally:
        close_llm_client(client)
    assert len(calls) == 2
    assert len(events) == 1
    assert events[0].request_id == "test-request"
    assert events[0].response_model == "provider-snapshot"
    assert events[0].reasoning_tokens == 3


@pytest.mark.parametrize("status", [400, 401, 403])
def test_permanent_http_errors_are_not_retried(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(status, json={"error": {"message": "bad request"}})

    _install_transport(monkeypatch, handler, 2)
    client = create_llm_client()
    try:
        with pytest.raises(openai.APIStatusError):
            client("s", "u")
    finally:
        close_llm_client(client)
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["text", "structured", "vision"])
def test_truncation_is_not_a_completed_summary_or_an_http_retry(monkeypatch, operation):
    calls = []

    def handler(request):
        calls.append(request)
        return _reply(finish_reason="length", content="Identifying prefix")

    _install_transport(monkeypatch, handler, 2)
    events = []
    client = create_llm_client(on_completion=events.append)
    try:
        if operation == "text":
            with pytest.raises(LlmIncompleteResponseError, match="truncated"):
                _invoke(client, operation)
        elif operation == "structured":
            result = _invoke(client, operation)
            assert result.status == "incomplete"
        else:
            assert _invoke(client, operation) == "Identifying prefix"
    finally:
        close_llm_client(client)
    assert len(calls) == 1
    assert len(events) == 1
    assert events[0].finish_reason == "length"


@pytest.mark.parametrize("operation", ["text", "structured", "vision"])
@pytest.mark.parametrize("parameter", ["max_completion_tokens", "max_tokens"])
@pytest.mark.parametrize("storage", ["false", "omit"])
def test_options_reach_the_wire_through_the_real_sdk(
    monkeypatch, operation, parameter, storage
):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return _reply()

    _install_transport(monkeypatch, handler, 0)
    monkeypatch.setenv("DATASHEETINDEX_REASONING_EFFORT", "low")
    monkeypatch.setenv("DATASHEETINDEX_VISION_REASONING_EFFORT", "none")
    monkeypatch.setenv("DATASHEETINDEX_MAX_OUTPUT_TOKENS", "900")
    monkeypatch.setenv("DATASHEETINDEX_TOKEN_LIMIT_PARAMETER", parameter)
    monkeypatch.setenv("DATASHEETINDEX_LLM_STORE", storage)
    client = create_llm_client()
    try:
        _invoke(client, operation)
    finally:
        close_llm_client(client)
    body = calls[0]
    assert body["reasoning_effort"] == ("none" if operation == "vision" else "low")
    assert body[parameter] == {"text": 900, "structured": 500, "vision": 300}[operation]
    other = (
        "max_tokens"
        if parameter == "max_completion_tokens"
        else "max_completion_tokens"
    )
    assert other not in body
    if storage == "omit":
        assert "store" not in body
    else:
        assert body["store"] is False
