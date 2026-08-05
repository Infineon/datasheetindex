"""LLM client factory using OpenAI Responses API via LiteLLM gateway."""

from __future__ import annotations

import importlib
import logging
import os
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

logger = logging.getLogger(__name__)


class LlmCallable(Protocol):
    """Callable interface used by the indexing pipeline."""

    def __call__(self, system: str, user: str) -> str:
        """Run a prompt pair and return text output."""


@dataclass(frozen=True)
class StructuredLlmResult:
    """Structured Responses payload plus completion metadata."""

    output_text: str
    status: str | None = None
    incomplete_details: object | None = None


class StructuredLlmCallable(Protocol):
    """Optional structured-output interface for schema-constrained calls."""

    def structured_json(
        self,
        system: str,
        user: str,
        *,
        name: str,
        schema: dict[str, object],
        max_output_tokens: int | None = None,
    ) -> StructuredLlmResult:
        """Run a JSON-schema constrained response request."""


class VisionLlmCallable(Protocol):
    """Optional image-input interface for figure captioning."""

    def describe_image(
        self, system: str, image_base64: str, *, media_type: str = "image/png"
    ) -> str:
        """Describe one image in a single line."""


class _ResponsesOutput(Protocol):
    @property
    def output_text(self) -> str:
        """Concatenated text output of the response."""


class _ResponsesApi(Protocol):
    def create(
        self,
        *,
        model: str,
        instructions: str = "",
        input: str | list[dict[str, object]],
        **kwargs: object,
    ) -> _ResponsesOutput:
        """Create an LLM response."""


class _ChatMessage(Protocol):
    @property
    def content(self) -> str | None:
        """The assistant's reply, or None when the model produced no text."""


class _ChatChoice(Protocol):
    @property
    def message(self) -> _ChatMessage:
        """The message of this choice."""


class _ChatCompletion(Protocol):
    @property
    def choices(self) -> Sequence[_ChatChoice]:
        """The completion's choices; we always request and read the first."""


class _ChatCompletionsApi(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        **kwargs: object,
    ) -> _ChatCompletion:
        """Create a chat completion."""


def _close_resource(resource: object | None) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY = 4.0
_RETRY_MAX_DELAY = 60.0

#: Ceiling on one figure caption. The caption prompt asks for under 60 words,
#: which lands around 90 tokens; measured over 16 real figure regions the
#: median is 71-102 and gpt-4.1's worst case is 197. It exists for the models
#: that do not honour the word limit on a dense figure: qwen3.6-27b answered a
#: 128-pin TQFP pinout by enumerating all 128 pins, 667 tokens. Truncation is
#: not a new failure mode here -- the caption prompt already orders its output
#: for it ("your text may be truncated, so identifying labels must come before
#: any description of structure"), so a clipped caption keeps the part that
#: earns its place in the index. Set above every compliant answer observed, so
#: it binds on runaways only.
VISION_MAX_TOKENS = 300


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


def _normalize_incomplete_details(details: object | None) -> object | None:
    if details is None:
        return None
    model_dump = getattr(details, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return details


def _call_structured_with_retry(
    responses_api: _ResponsesApi,
    model: str,
    system: str,
    user: str,
    *,
    name: str,
    schema: dict[str, object],
    max_output_tokens: int | None = None,
) -> StructuredLlmResult:
    """Call the Responses API with a strict JSON schema and retry if needed."""

    last_exc: Exception | None = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            request: dict[str, object] = {
                "model": model,
                "instructions": system,
                "input": user,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
            if max_output_tokens is not None:
                request["max_output_tokens"] = max_output_tokens

            response = responses_api.create(**request)
            return StructuredLlmResult(
                output_text=response.output_text,
                status=getattr(response, "status", None),
                incomplete_details=_normalize_incomplete_details(
                    getattr(response, "incomplete_details", None)
                ),
            )
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == _RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
            logger.warning(
                "Structured LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
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
        self,
        responses_api: _ResponsesApi,
        http_client: object,
        model: str,
        *,
        chat_api: _ChatCompletionsApi,
        vision_model: str | None = None,
    ) -> None:
        self._responses_api = responses_api
        self._chat_api = chat_api
        self._model = model
        #: The text model unless a deployment names a different one. Vision is
        #: the one call whose model is worth separating: it is the only
        #: per-figure cost, and the cheapest capable model on a gateway is
        #: rarely the one you want writing summaries.
        self._vision_model = vision_model or model
        self._finalizer = weakref.finalize(self, _close_resource, http_client)

    def __call__(self, system: str, user: str) -> str:
        return _call_with_retry(self._responses_api, self._model, system, user)

    def structured_json(
        self,
        system: str,
        user: str,
        *,
        name: str,
        schema: dict[str, object],
        max_output_tokens: int | None = None,
    ) -> StructuredLlmResult:
        return _call_structured_with_retry(
            self._responses_api,
            self._model,
            system,
            user,
            name=name,
            schema=schema,
            max_output_tokens=max_output_tokens,
        )

    def describe_image(
        self, system: str, image_base64: str, *, media_type: str = "image/png"
    ) -> str:
        """Describe one image, over **Chat Completions** rather than Responses.

        The transport is deliberate and measured, not a stylistic preference.
        The Responses API is a bridge for any gateway model that is not natively
        an OpenAI one, and on a LiteLLM gateway that bridge can misfile the
        model's answer as a *reasoning* item carrying ``reasoning_text``.
        ``output_text`` reads only ``output_text`` chunks, so the caption comes
        back as the empty string with the text sitting right there in the
        payload. Measured against the self-hosted ``qwen3.6-27b`` (vLLM) on the
        prod gateway over 16 real figure regions: **8 of 16 captions empty**,
        reproduced three times, and a *different* 8 each run -- it is per-call
        sampling, not a property of a figure. Chat Completions is not a
        workaround for that bridge; it bypasses it. Same model, same images,
        same prompt: 0 empty in 112 calls, and the raw message has no reasoning
        channel at all.

        An empty caption is worse here than it sounds: ``caption_figures_in_place``
        counts a blank reply as ``failed``, which marks the artifact incomplete,
        so a coin-flip transport would re-caption the document on every build
        forever.

        The path is single for every model, not branched by model name. gpt-4.1
        was re-measured over the same 16 regions on this transport and is
        indistinguishable from the Responses path -- 1084 median input tokens
        either way, which is also the check that nested ``detail`` is honoured
        (see below); a name-based branch would buy nothing and would need a
        list of model names to maintain.

        Sent at ``detail="high"``, reversing the ``"low"`` this call used
        before. ``"low"`` downscales to 512x512 before the model ever sees
        the image, which sounded like the safer choice for the
        no-transcription rule -- the model cannot fabricate rows from detail
        it never received. Measured on the PCN's page-5 table (20 rows, 9
        columns) with an explicit "list the row headings verbatim, or say
        you cannot read them" probe, it did exactly that instead: `"low"`
        invented `Voltage`, `Wafer Base Supplier`, `Wafer Fab Location`,
        `Package Fab (OSAT)`, `Package Type`, `Mold Compound Lot Number`, and
        `Mold Compound Location`, and missed real rows -- confident
        fabrication, not a safe blank. At `"high"` the same probe returned
        19 of 20 row headings verbatim correct (the one miss: `Die
        Composition` for `Bond Wire Composition`), including both supplier
        rows the row-labels-first prompt exists to surface. Cost, read back
        from `usage` on live responses rather than estimated: 120 input
        tokens per image at `"low"`, 1074 at `"high"` -- about 9x, or roughly
        2.4k to 21.5k input tokens per document at the default cap of 20. It
        is paid once per document and then cached on disk by the existing
        artifact reuse.

        Note that ``image_url`` is an **object** here. On the Responses API it
        is a plain string; the two forms are not interchangeable, and the wrong
        one type-checks and then fails at the gateway.
        """
        response = self._chat_api.create(
            model=self._vision_model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=VISION_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

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


def _vision_model_env() -> str | None:
    """The model figure captioning should use, when a deployment names one.

    ``None`` -- the default -- means vision follows ``model``, which is exactly
    the behaviour before this knob existed.

    Env rather than a hardcoded name because the cheapest capable vision model
    on a gateway is a property of *that deployment*, not of this library. The
    one that motivated the knob, ``qwen3.6-27b``, is a self-hosted alias that
    exists on one internal gateway: it is absent from that gateway's own
    staging tier (which serves ``qwen3.5-27b``) and means nothing to an outside
    user pointing ``LITELLM_BASE_URL`` at some other endpoint. Baking it in
    would break both of them to save one line of configuration.
    """
    value = os.environ.get("DATASHEETINDEX_VISION_MODEL")
    if value is None:
        return None
    value = value.strip()
    return value or None


def create_llm_client(model: str = "gpt-4.1") -> LlmCallable:
    """Create a sync LLM callable backed by the OpenAI-compatible gateway.

    Reads ``LITELLM_BASE_URL`` and ``LITELLM_MASTER_KEY`` from the
    environment (loading ``.env`` via python-dotenv if available).
    TLS verification is disabled by default for compatibility with internal
    endpoints and can be enabled with ``LITELLM_TLS_VERIFY=true``.
    Request timeout and retry policy can be tuned with
    ``LITELLM_TIMEOUT_SECONDS`` and ``LITELLM_MAX_RETRIES``.

    Text calls use the Responses API; figure captioning uses Chat Completions
    and, when ``DATASHEETINDEX_VISION_MODEL`` is set, a different model. See
    ``_ManagedLlmClient.describe_image`` for why the transports differ.

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
        # The SDK's Responses.create spells out every request field as a named
        # parameter, so it cannot structurally satisfy a protocol that forwards
        # **kwargs -- even though it accepts every field we pass. Cast at the
        # SDK boundary rather than weaken _ResponsesApi for our own callers.
        responses_api=cast("_ResponsesApi", client.responses),
        chat_api=cast("_ChatCompletionsApi", client.chat.completions),
        http_client=http_client,
        model=model,
        vision_model=_vision_model_env(),
    )


def close_llm_client(llm_callable: object | None) -> None:
    """Close a managed LLM callable if it exposes a ``close()`` method."""
    _close_resource(llm_callable)


def get_structured_output_client(
    llm_callable: object | None,
) -> StructuredLlmCallable | None:
    """Return the structured-output interface when the callable exposes one."""

    structured_json = getattr(llm_callable, "structured_json", None)
    if callable(structured_json):
        return cast(StructuredLlmCallable, llm_callable)
    return None


def get_vision_client(llm_callable: object | None) -> VisionLlmCallable | None:
    """Return the vision interface when the callable exposes one.

    Duck-typed rather than a change to ``LlmCallable``, following the
    ``structured_json`` precedent: a third-party ``(system, user) -> str`` a
    consumer injects today simply yields no captions instead of breaking.
    """
    describe_image = getattr(llm_callable, "describe_image", None)
    if callable(describe_image):
        return cast(VisionLlmCallable, llm_callable)
    return None
