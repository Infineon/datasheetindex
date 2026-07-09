"""LLM client factory using OpenAI Responses API via LiteLLM gateway."""

from __future__ import annotations

import importlib
import logging
import os
import time
import weakref
from typing import Protocol

logger = logging.getLogger(__name__)


class LlmCallable(Protocol):
    """Callable interface used by the indexing pipeline."""

    def __call__(self, system: str, user: str) -> str:
        """Run a prompt pair and return text output."""


class _ResponsesOutput(Protocol):
    @property
    def output_text(self) -> str:
        """Concatenated text output of the response."""


class _ResponsesApi(Protocol):
    def create(self, *, model: str, instructions: str, input: str) -> _ResponsesOutput:
        """Create an LLM response."""


def _close_resource(resource: object | None) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY = 4.0
_RETRY_MAX_DELAY = 60.0


def _is_retryable(exc: Exception) -> bool:
    """Check if an API error is retryable (429 or 5xx)."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "too many" in msg


def _call_with_retry(
    responses_api: _ResponsesApi,
    model: str,
    system: str,
    user: str,
) -> str:
    """Call the LLM API with exponential backoff on retryable errors."""
    last_exc: Exception | None = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            response = responses_api.create(
                model=model,
                instructions=system,
                input=user,
            )
            return response.output_text
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == _RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt + 1,
                _RETRY_MAX_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


class _ManagedLlmClient:
    """Callable wrapper that owns and closes its underlying HTTP client."""

    def __init__(
        self, responses_api: _ResponsesApi, http_client: object, model: str
    ) -> None:
        self._responses_api = responses_api
        self._model = model
        self._finalizer = weakref.finalize(self, _close_resource, http_client)

    def __call__(self, system: str, user: str) -> str:
        return _call_with_retry(self._responses_api, self._model, system, user)

    def close(self) -> None:
        """Release the underlying HTTP client once the callable is no longer needed."""
        if self._finalizer.alive:
            self._finalizer()


DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2


def _parse_tls_verify_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_timeout_seconds_env(value: str | None) -> float:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("LITELLM_TIMEOUT_SECONDS must be a positive number") from exc
    if parsed <= 0:
        raise ValueError("LITELLM_TIMEOUT_SECONDS must be a positive number")
    return parsed


def _parse_max_retries_env(value: str | None) -> int:
    if value is None:
        return DEFAULT_MAX_RETRIES
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("LITELLM_MAX_RETRIES must be an integer >= 0") from exc
    if parsed < 0:
        raise ValueError("LITELLM_MAX_RETRIES must be an integer >= 0")
    return parsed


def create_llm_client(model: str = "gpt-4.1") -> LlmCallable:
    """Create a sync LLM callable backed by the OpenAI Responses API.

    Reads ``LITELLM_BASE_URL`` and ``LITELLM_MASTER_KEY`` from the
    environment (loading ``.env`` via python-dotenv if available).
    TLS verification is disabled by default for compatibility with internal
    endpoints and can be enabled with ``LITELLM_TLS_VERIFY=true``.
    Request timeout and retry policy can be tuned with
    ``LITELLM_TIMEOUT_SECONDS`` and ``LITELLM_MAX_RETRIES``.

    Returns a ``(system: str, user: str) -> str`` callable. The returned object
    also exposes ``close()`` for explicit cleanup when the caller owns it.
    """
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError:
        pass
    else:
        dotenv.load_dotenv()

    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM_MASTER_KEY")

    if not base_url or not api_key:
        raise ValueError(
            "LITELLM_BASE_URL and LITELLM_MASTER_KEY must be set "
            "in the environment or .env file"
        )

    tls_verify = _parse_tls_verify_env(os.environ.get("LITELLM_TLS_VERIFY"))
    timeout_seconds = _parse_timeout_seconds_env(
        os.environ.get("LITELLM_TIMEOUT_SECONDS")
    )
    max_retries = _parse_max_retries_env(os.environ.get("LITELLM_MAX_RETRIES"))
    httpx = importlib.import_module("httpx")
    openai = importlib.import_module("openai")
    http_client = httpx.Client(verify=tls_verify, timeout=timeout_seconds)
    client = openai.OpenAI(
        base_url=base_url.rstrip("/") + "/v1",
        api_key=api_key,
        http_client=http_client,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )
    return _ManagedLlmClient(
        responses_api=client.responses,
        http_client=http_client,
        model=model,
    )


def close_llm_client(llm_callable: object | None) -> None:
    """Close a managed LLM callable if it exposes a ``close()`` method."""
    _close_resource(llm_callable)
