"""LLM client factory for an OpenAI-compatible gateway.

Every call -- text, structured, and vision -- goes over **Chat Completions**.
There is deliberately no second transport: see ``_ManagedLlmClient.describe_image``
for the measurement that chose it, and ``_read_chat_reply`` for the guard that
makes the failure it fixes visible rather than silent.
"""

from __future__ import annotations

import importlib
import logging
import os
import ssl
import time
import weakref
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Literal, Protocol, cast

logger = logging.getLogger(__name__)


class LlmCallable(Protocol):
    """Callable interface used by the indexing pipeline."""

    def __call__(self, system: str, user: str) -> str:
        """Run a prompt pair and return text output."""


@dataclass(frozen=True)
class StructuredLlmResult:
    """Structured JSON payload plus completion metadata.

    The field names predate the move to Chat Completions and are kept
    deliberately -- ``toc_fallback`` reads them, and a consumer may inject its
    own ``structured_json`` callable. ``status`` is ``None`` when the gateway
    reports no completion signal, which says nothing about the payload.
    """

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


class _ChatMessage(Protocol):
    @property
    def content(self) -> str | None:
        """The assistant's reply, or None when the model produced no text."""


class _ChatChoice(Protocol):
    @property
    def message(self) -> _ChatMessage:
        """The message of this choice."""

    @property
    def finish_reason(self) -> str | None:
        """Why generation stopped; ``"length"`` means the cap bound."""


class _ChatCompletion(Protocol):
    @property
    def choices(self) -> Sequence[_ChatChoice]:
        """The completion's choices; we request one and read the first."""


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


class LlmTlsVerificationError(RuntimeError):
    """The LLM gateway's TLS certificate could not be verified.

    Raised in place of the transport's own error, which describes this failure
    uselessly: openai reports ``APIConnectionError("Connection error.")`` and
    the ``ssl.SSLCertVerificationError`` naming the real cause sits three links
    down the exception chain (openai -> httpx2 -> httpcore2 -> ssl, measured
    against a self-signed local server). It is **not** three links of
    ``__cause__``: httpcore2 re-raises ``from None``, so the last hop is
    reachable only through ``__context__``. Do not shorten this to "the cause
    chain" -- that misreading is what produced a guard that detected nothing,
    and ``_tls_verification_failure`` carries the measurement.

    A *distinguishable type* is the point, not the nicer message. Every caller
    of this client wraps its calls in a blanket ``except Exception`` that logs a
    warning and degrades -- ``llm/toc_fallback.py`` returns ``[]`` when the
    first chunk fails, which reaches the caller as an index with **no ToC
    entries**, indistinguishable from a document that genuinely has no outline.
    So the one LLM failure an operator can actually fix was also the least
    visible one. Two degrading sites now let this type through -- the ToC
    fallback's chunk loop and ``index.build``'s fallback handler. Figure
    captioning deliberately does not: it runs before the artifacts are
    written, so raising there destroys an otherwise-complete index.

    Scoped deliberately to *certificate verification*, not to connection errors
    at large. An unreachable host, a DNS failure or a refused connection are
    transient and worth degrading over; a certificate the local trust store will
    not accept is a permanent fact about the deployment that no retry and no
    later build will change. That is also why it is not retryable below.
    """


class LlmIncompleteResponseError(RuntimeError):
    """Plain-text generation hit its output limit; partial text is not a summary.

    Structured callers receive ``StructuredLlmResult.status`` instead. Caption
    prefixes are intentionally useful even when clipped and remain available.
    """


class LlmCredentialsMissingError(ValueError):
    """``LITELLM_BASE_URL`` or ``LITELLM_MASTER_KEY`` is not set.

    The one ``create_llm_client`` failure that means "no client here": ``[llm]``
    is optional and a keyless install is the default, so the self-creating
    callers catch this type and build without an LLM. A malformed setting beside
    valid credentials raises a plain ``ValueError`` instead and must reach the
    operator -- caught as "no client", a one-letter typo silently disabled the
    ToC fallback and captions. A ``ValueError`` subclass so callers that caught
    the old type keep working.
    """


def _tls_verification_failure(
    exc: BaseException,
) -> ssl.SSLCertVerificationError | None:
    """Find a certificate-verification failure anywhere in ``exc``'s causes.

    Walks ``__cause__`` first and falls back to ``__context__``: a bare
    ``raise`` inside an ``except`` block links implicitly through the latter.
    Cycle-guarded because ``raise X from Y`` can build one, and this runs on the
    failure path of every gateway call.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return current
        # ``__context__`` is followed even when ``__suppress_context__`` is set,
        # and that is load-bearing rather than sloppy. ``httpcore2``'s connection
        # pool re-raises every exception as ``raise exc from None``, which clears
        # ``__cause__`` and sets ``__suppress_context__`` -- so on the real stack
        # the ``ssl.SSLCertVerificationError`` hangs off ``__context__`` of a
        # link that claims its context is suppressed:
        #
        #   httpx2.ConnectError     cause=ConnectError  suppress=True
        #   httpcore2.ConnectError  cause=None          suppress=True
        #     context=ssl.SSLCertVerificationError   <-- only reachable here
        #
        # Honouring the flag was tried and reverted: it stops the walk at
        # exactly that link, so nothing is ever detected and the whole feature
        # silently does nothing. The cost is that an unrelated error raised
        # ``from None`` while handling a certificate failure is still classified
        # as one -- a shape no library in this stack produces, and one where an
        # ``ssl.SSLCertVerificationError`` did genuinely occur anyway.
        # Identity, not truthiness: an exception type that subclasses an empty
        # collection is falsy, and `or` would skip a real __cause__ branch.
        current = (
            current.__cause__ if current.__cause__ is not None else current.__context__
        )
    return None


def _raise_if_tls_verification_failed(exc: Exception) -> None:
    """Re-raise ``exc`` as :class:`LlmTlsVerificationError` if that is what it is.

    A no-op for every other failure, so it is safe as the first statement of a
    blanket ``except`` block.
    """
    cause = _tls_verification_failure(exc)
    if cause is None:
        return
    target = os.environ.get("LITELLM_BASE_URL") or "the URL in LITELLM_BASE_URL"
    raise LlmTlsVerificationError(
        f"TLS certificate verification failed for the LLM gateway at {target}: "
        f"{cause}. Add the gateway's CA certificate to the trust store, or set "
        "LITELLM_TLS_VERIFY=false to skip verification. Prefer the trust store: "
        "LITELLM_MASTER_KEY travels on this channel."
    ) from exc


@contextmanager
def _naming_tls_failures() -> Generator[None]:
    """Wrap a gateway call so a certificate failure arrives as a named error.

    Every call shape -- text, structured and vision -- reaches the gateway
    inside this block, so the conversion cannot be wired into two of the three
    and missed on the last.

    A context manager rather than a ``create(**kwargs)`` wrapper on purpose:
    forwarding through ``**kwargs`` erases ``_ChatCompletionsApi.create``'s
    signature, and the type checker stops checking `model=`/`messages=` at all
    three call sites. Here the call keeps its own shape and stays checked.
    """
    try:
        yield
    except Exception as exc:
        _raise_if_tls_verification_failed(exc)
        raise


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


@dataclass(frozen=True)
class ChatRequestOptions:
    """Generation settings independent of model names and the SDK.

    Unset reasoning is omitted, not guessed from a model alias. ``store=None``
    omits the field for gateways that do not support it. Vision uses 300 tokens
    when no budget is named; text and structured calls have no default budget.
    """

    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    token_limit_parameter: Literal["max_completion_tokens", "max_tokens"] = (
        "max_completion_tokens"
    )
    store: bool | None = False

    def __post_init__(self) -> None:
        if self.reasoning_effort not in (
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            raise ValueError(
                "reasoning_effort must be none/minimal/low/medium/high/xhigh/max"
            )
        if self.max_output_tokens is not None:
            _validate_token_limit(self.max_output_tokens)
        if self.token_limit_parameter not in ("max_completion_tokens", "max_tokens"):
            raise ValueError(
                "token_limit_parameter must be max_completion_tokens or max_tokens"
            )
        if self.store is not None and not isinstance(self.store, bool):
            raise ValueError("store must be a boolean or None (omit)")

    def to_kwargs(self, *, max_output_tokens: int | None = None) -> dict[str, object]:
        """An explicit per-call budget overrides the configured budget."""
        limit = (
            self.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        result: dict[str, object] = {}
        if limit is not None:
            _validate_token_limit(limit)
            result[self.token_limit_parameter] = limit
        if self.reasoning_effort is not None:
            result["reasoning_effort"] = self.reasoning_effort
        if self.store is not None:
            result["store"] = self.store
        return result


def _validate_token_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_output_tokens must be an integer > 0")


@dataclass(frozen=True)
class LlmCallMetadata:
    """Payload-free metadata for one SDK invocation that returned a response.

    Includes empty replies and refusals. SDK-internal retries are included in
    elapsed time, not separate events. No prompt, answer, image, credential, or
    refusal text is retained. Compatible gateways may omit token counts/IDs.
    """

    operation: Literal["text", "structured", "vision"]
    requested_model: str
    response_model: str | None
    finish_reason: str | None
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    request_id: str | None
    empty: bool
    refused: bool


#: Statuses that mean the deployment is misconfigured rather than unlucky.
#:
#: 404 is deliberately absent. It reads like "the model does not exist", but a
#: gateway also returns it for a route that is temporarily unregistered -- and
#: the wrong-model case already repairs itself, because
#: ``DATASHEETINDEX_VISION_MODEL`` is part of the artifact-reuse key, so
#: correcting the name invalidates the artifact without any help from here.
_PERMANENT_STATUS_CODES = frozenset({401, 403})


def is_permanent_llm_failure(exc: BaseException) -> bool:
    """Is this a misconfiguration no retry and no later build will fix?

    The point is to tell "your certificate is wrong" apart from "the gateway
    had a bad afternoon", so a caller can say the first out loud and stay quiet
    about the second. Being wrong in the loud direction is the expensive one:
    it sends an operator to fix a gateway that is already fine, so anything
    unrecognised is treated as transient.

    Duck-typed on ``status_code`` -- the openai SDK sets it on its status
    errors -- so this stays importable with no ``[llm]`` extra installed.
    An unrecognised shape stays transient rather than being guessed permanent.
    The SDK alone owns HTTP retries for every request shape.
    """
    # Both forms, and both are needed. The type alone misses a raw transport
    # chain from any future path that skipped the conversion; the chain walk
    # alone misses a directly-constructed LlmTlsVerificationError, which has no
    # __cause__ -- and since that type is exported from the package root,
    # raising one bare is something a caller may legitimately do.
    if isinstance(exc, LlmTlsVerificationError):
        return True
    if _tls_verification_failure(exc) is not None:
        return True
    return getattr(exc, "status_code", None) in _PERMANENT_STATUS_CODES


def _read_chat_reply(
    response: _ChatCompletion, model: str, *, what: str
) -> tuple[str, str | None]:
    """Pull the text out of a chat completion, and never let a blank one be silent.

    Both failures handled here are well-formed 200s that carry no text, and both
    used to be invisible. A caller sees only ``""``, which every consumer in this
    package scores as "the model had nothing to say" -- indistinguishable from a
    transport that swallowed the answer.

    That distinction is not hypothetical. The bug that moved captioning onto this
    transport (a gateway bridge misfiling the answer as a reasoning item, leaving
    ``output_text`` empty with the text sitting in the payload) took a five-run,
    16-region campaign to see *precisely because* nothing logged it. The class of
    failure outlives its fix, so name the two things that separate its causes:
    the model, and whether the token cap bound.

    Returns the text and the ``finish_reason``; the latter is what
    ``structured_json`` turns into a completion status.
    """
    if not response.choices:
        # A well-formed 200 with no choices at all: a content filter or an error
        # envelope. Returning "" keeps every caller's contract instead of raising
        # an IndexError whose text says nothing about what was being asked for.
        logger.warning("%s returned no choices from model %s", what, model)
        return "", None

    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)

    # Checked before the blank-content branch below, and that order is the whole
    # point. A refusal arrives as ``content=None`` with the model's explanation
    # in ``refusal`` and ``finish_reason="stop"`` -- so without this it logs
    # "came back empty (finish_reason=stop)", which names the wrong cause in a
    # message whose only job is to name the right one. OpenAI's structured-output
    # guidance is to inspect ``refusal`` *before* parsing content, because a
    # refusal deliberately does not follow the supplied schema.
    #
    # Read with getattr rather than declared on ``_ChatMessage``: the field is
    # OpenAI's, and an OpenAI-compatible backend such as vLLM need not send it.
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        logger.warning("%s was refused by model %s: %s", what, model, refusal)
        return "", finish_reason

    if not content.strip():
        logger.warning(
            "%s came back empty from model %s (finish_reason=%s). A reasoning "
            "model can spend its whole token budget thinking and return nothing; "
            "name a non-reasoning model if that is what happened.",
            what,
            model,
            finish_reason,
        )
    return content, finish_reason


class _ManagedLlmClient:
    """Callable wrapper that owns and closes its underlying HTTP client."""

    def __init__(
        self,
        chat_api: _ChatCompletionsApi,
        http_client: object,
        model: str,
        *,
        vision_model: str | None = None,
        text_options: ChatRequestOptions | None = None,
        vision_options: ChatRequestOptions | None = None,
        on_completion: Callable[[LlmCallMetadata], None] | None = None,
    ) -> None:
        self._chat_api = chat_api
        self._model = model
        #: The text model unless a deployment names a different one. Vision is
        #: the one call whose model is worth separating: it is the only
        #: per-figure cost, and the cheapest capable model on a gateway is
        #: rarely the one you want writing summaries.
        self._vision_model = vision_model or model
        self._text_options = text_options or ChatRequestOptions()
        self._vision_options = vision_options or ChatRequestOptions(
            max_output_tokens=VISION_MAX_TOKENS
        )
        self._on_completion = on_completion
        self._finalizer = weakref.finalize(self, _close_resource, http_client)

    def __call__(self, system: str, user: str) -> str:
        started = time.perf_counter()
        with _naming_tls_failures():
            response = self._chat_api.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **self._text_options.to_kwargs(),
            )
        self._record_completion(response, "text", self._model, started)
        content, finish_reason = _read_chat_reply(
            response, self._model, what="Text completion"
        )
        if finish_reason == "length":
            raise LlmIncompleteResponseError(
                f"Text completion from model {self._model} was truncated "
                "(finish_reason=length); refusing to return incomplete text. "
                "Increase the text output budget or reduce reasoning effort."
            )
        return content

    def structured_json(
        self,
        system: str,
        user: str,
        *,
        name: str,
        schema: dict[str, object],
        max_output_tokens: int | None = None,
    ) -> StructuredLlmResult:
        optional = self._text_options.to_kwargs(max_output_tokens=max_output_tokens)
        started = time.perf_counter()
        with _naming_tls_failures():
            response = self._chat_api.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True, "schema": schema},
                },
                **optional,
            )
        self._record_completion(response, "structured", self._model, started)
        content, finish_reason = _read_chat_reply(
            response, self._model, what="Structured completion"
        )
        # Missing completion metadata is not evidence of truncation.
        if finish_reason is None:
            return StructuredLlmResult(output_text=content)
        completed = finish_reason == "stop"
        return StructuredLlmResult(
            output_text=content,
            status="completed" if completed else "incomplete",
            incomplete_details=None if completed else {"reason": finish_reason},
        )

    def describe_image(
        self, system: str, image_base64: str, *, media_type: str = "image/png"
    ) -> str:
        """Describe one image in a single line.

        This is the call whose measurement chose **Chat Completions** for the
        whole client, so the reasoning is recorded here rather than at the
        transport it now explains.

        The Responses API is a bridge for any gateway model that is not natively
        an OpenAI one, and on a LiteLLM gateway that bridge can misfile the
        model's answer as a *reasoning* item carrying ``reasoning_text``.
        ``output_text`` reads only ``output_text`` chunks, so the caption comes
        back as the empty string with the text sitting right there in the
        payload. Measured against the self-hosted ``qwen3.6-27b`` (vLLM) on the
        prod gateway over 16 real figure regions: **8 to 12 of 16 captions
        empty** over five runs, and a *different* subset each run -- it is
        per-call sampling, not a property of a figure. Chat Completions is not a
        workaround for that bridge; it bypasses it. Same model, same images,
        same prompt: 0 empty in 144 calls, and the raw message has no reasoning
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
        list of model names to maintain. The same argument is why the text and
        structured calls were later folded onto this transport too: one path for
        every model *and* every call shape, with no second protocol to keep
        working.

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
        was a plain string; the two forms are not interchangeable, and the wrong
        one type-checks and then fails at the gateway. Worth keeping in mind if
        this call is ever ported back.
        """
        started = time.perf_counter()
        with _naming_tls_failures():
            response = self._chat_api.create(
                model=self._vision_model,
                messages=[
                    # The prompt is sent twice, deliberately. It is parity with the
                    # Responses shape this replaced (``instructions=`` *and* an
                    # ``input_text`` part), the measured 1074/1084 input tokens
                    # quoted above already include it, and it keeps the instruction
                    # visible to a model that weights the system role weakly.
                    # Deleting either copy changes every number in this docstring.
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
                # Both token spellings count reasoning as well as visible output.
                # Explicit vision reasoning controls, not a larger universal cap,
                # fix a reasoning model exhausting the budget before it writes.
                **self._vision_options.to_kwargs(
                    max_output_tokens=(
                        self._vision_options.max_output_tokens or VISION_MAX_TOKENS
                    )
                ),
            )
        self._record_completion(response, "vision", self._vision_model, started)
        # An empty caption is worse here than the shared warning conveys, so the
        # vision-specific advice stays: ``caption_figures_in_place`` scores a
        # blank reply as ``failed``, which marks the artifact incomplete and
        # re-captions the document on every build.
        caption, finish_reason = _read_chat_reply(
            response, self._vision_model, what="Figure caption"
        )
        if not caption.strip() and finish_reason == "length":
            logger.warning(
                "The %d-token vision budget bound before any caption text was "
                "written; configure DATASHEETINDEX_VISION_REASONING_EFFORT=none "
                "if supported, or name a non-reasoning DATASHEETINDEX_VISION_MODEL.",
                self._vision_options.max_output_tokens or VISION_MAX_TOKENS,
            )
        # Keep the public single-line contract even when a model wraps its
        # caption in paragraphs or a list. Do not rewrite or truncate its text.
        return " ".join(caption.split())

    def _record_completion(
        self,
        response: _ChatCompletion,
        operation: Literal["text", "structured", "vision"],
        model: str,
        started: float,
    ) -> None:
        callback = self._on_completion
        if callback is None:
            return
        choice = response.choices[0] if response.choices else None
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        metadata = LlmCallMetadata(
            operation=operation,
            requested_model=model,
            response_model=_optional_str(response, "model"),
            finish_reason=_optional_str(choice, "finish_reason"),
            elapsed_seconds=time.perf_counter() - started,
            prompt_tokens=_optional_int(usage, "prompt_tokens"),
            completion_tokens=_optional_int(usage, "completion_tokens"),
            reasoning_tokens=_optional_int(details, "reasoning_tokens"),
            request_id=_optional_str(response, "_request_id"),
            empty=not choice or not (choice.message.content or "").strip(),
            refused=bool(choice and getattr(choice.message, "refusal", None)),
        )
        try:
            callback(metadata)
        except Exception:
            # Telemetry must not fail a successful build or leak callback payloads
            # through its exception message. Callbacks may run on caption threads.
            logger.warning("LLM completion callback failed; ignoring it")

    def close(self) -> None:
        """Release the underlying HTTP client once the callable is no longer needed."""
        if self._finalizer.alive:
            self._finalizer()


DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2


def _optional_str(value: object, name: str) -> str | None:
    field = getattr(value, name, None)
    return field if isinstance(field, str) else None


def _optional_int(value: object, name: str) -> int | None:
    field = getattr(value, name, None)
    return field if isinstance(field, int) and not isinstance(field, bool) else None


def _parse_tls_verify_env(value: str | None) -> bool:
    """Resolve ``LITELLM_TLS_VERIFY`` into an ``httpx2`` ``verify`` argument.

    Unset means **verify**. It used to mean the opposite, which made every
    default install send ``LITELLM_MASTER_KEY`` to ``LITELLM_BASE_URL`` over a
    channel an interposer could read -- and silently, since an unverified
    connection succeeds exactly like a verified one. Nobody opted into that;
    it was simply what an absent variable happened to select.

    Disabling verification is now an explicit act: ``LITELLM_TLS_VERIFY`` set
    to ``0``/``false``/``no``/``off`` (case-insensitive). Any other value,
    including the empty string, verifies.
    """
    if value is None:
        return True
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


#: The text model when neither the caller nor the deployment names one.
#:
#: A last resort, not a recommendation: it is a name this library cannot know
#: any given gateway serves. ``DATASHEETINDEX_MODEL`` is how a deployment that
#: serves something else says so.
DEFAULT_TEXT_MODEL = "qwen3.8-27b"


def ensure_dotenv_loaded() -> None:
    """Fold ``.env`` into the environment, if python-dotenv is installed.

    Called by **both** model-name readers and by the factory, and that is
    load-bearing rather than defensive. It used to live only inside
    ``create_llm_client``, which made the answer to "what does this deployment
    name?" depend on whether a client had been constructed yet -- and
    ``_BuildOptions``, the artifact cache key, asks *before* any client exists.
    A ``.env``-configured model therefore keyed as ``None`` while the build that
    followed ran on the ``.env`` value, which both rebuilt every document on the
    second call of a process and served the previous model's output after the
    knob moved: precisely the silent staleness the key exists to prevent.

    Not cached. ``load_dotenv`` does not override variables already set, so
    repeating it is a no-op on everything the process was launched with, and a
    cache would freeze the answer for a long-lived server across a ``.env`` edit.

    Public, and called from ``tools/bound.py`` as well, because ``.env`` also
    carries non-LLM settings (``DATASHEETINDEX_OUTPUT_DIR``): folding it in once
    at the top of ``build_datasheet`` makes the load ordering-independent for
    every ``DATASHEETINDEX_*`` variable rather than for the two the model
    readers happen to touch. It lives here because python-dotenv is an ``[llm]``
    extra, so the optional-import guard belongs beside the others.

    Failures are swallowed. ``load_dotenv`` is not total -- a ``.env`` saved as
    UTF-16 by a Windows editor raises ``UnicodeDecodeError``, an unreadable one
    raises ``PermissionError`` -- and every ``create_llm_client`` caller already
    catches those, so before this was hoisted the worst case was "no LLM". It
    now runs on the plain ``build_datasheet`` path, which has no such guard, and
    a build that wanted no LLM at all must not die on a file it never needed.

    ``tests/conftest.py``'s ``_hermetic_llm_env`` neutralises this, which is why
    the suite cannot see the bug above on its own -- do not "simplify" the
    fixture's ``load_dotenv`` patch away.
    """
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError:
        return
    try:
        dotenv.load_dotenv()
    except (OSError, ValueError):
        logger.debug(
            "Could not read .env; continuing with the ambient environment",
            exc_info=True,
        )


def text_model_env() -> str | None:
    """The default model for the ToC fallback and summaries, when one is named.

    ``None`` means ``DEFAULT_TEXT_MODEL``, which is the behaviour before this
    knob existed.

    The counterpart to ``vision_model_env`` at the other end of the same
    question, and it exists because the two were asymmetric in a way that left
    a hole rather than merely an inconsistency. Captioning had a deployment-level
    override; text had only ``build_datasheet``'s ``model`` argument -- which the
    agent is told to omit unless summaries are requested or the ToC is poor. So
    the model that actually ran the *automatic* ToC fallback, the common path,
    was a hardcoded name with no override anywhere, and a gateway not serving it
    had no way to make the fallback work at all.

    It reaches summaries only through the Python API. ``build_datasheet`` and
    the CLI both reject ``include_summaries`` without an explicit ``model``, and
    an explicit model outranks this, so on those two surfaces the knob decides
    the ToC fallback and nothing else. Worth knowing before reading a summary
    and wondering which model wrote it.

    Deployment-level rather than per-call because which models a gateway serves
    is a property of that deployment, not of this library or of any one
    document -- the same argument ``vision_model_env`` makes. What *is* per-call
    is whether this document is worth spending a better model on, and that stays
    with the ``model`` argument, which wins over this.
    """
    ensure_dotenv_loaded()
    value = os.environ.get("DATASHEETINDEX_MODEL")
    if value is None:
        return None
    value = value.strip()
    return value or None


def vision_model_env() -> str | None:
    """The model figure captioning should use, when a deployment names one.

    ``None`` -- the default -- means vision follows the resolved text model,
    which is exactly the behaviour before this knob existed.

    Env rather than a hardcoded name because the cheapest capable vision model
    on a gateway is a property of *that deployment*, not of this library. The
    one that motivated the knob, ``qwen3.6-27b``, is a self-hosted alias that
    exists on one internal gateway: it is absent from that gateway's own
    staging tier (which serves ``qwen3.5-27b``) and means nothing to an outside
    user pointing ``LITELLM_BASE_URL`` at some other endpoint. Baking it in
    would break both of them to save one line of configuration.

    A reasoning model may exhaust ``VISION_MAX_TOKENS`` before writing a caption.
    Set ``DATASHEETINDEX_VISION_REASONING_EFFORT=none`` when supported, or name a
    non-reasoning model. Reasoning settings are independent of the model alias.
    """
    ensure_dotenv_loaded()
    value = os.environ.get("DATASHEETINDEX_VISION_MODEL")
    if value is None:
        return None
    value = value.strip()
    return value or None


def _request_options_env(
    reasoning_variable: str, budget_variable: str, default_budget: int | None
) -> ChatRequestOptions:
    ensure_dotenv_loaded()
    reasoning = os.environ.get(reasoning_variable, "").strip().lower() or None
    budget_value = os.environ.get(budget_variable, "").strip()
    budget = default_budget
    if budget_value:
        try:
            budget = int(budget_value)
            _validate_token_limit(budget)
        except ValueError as exc:
            raise ValueError(f"{budget_variable} must be an integer > 0") from exc
    parameter = (
        os.environ.get("DATASHEETINDEX_TOKEN_LIMIT_PARAMETER", "").strip().lower()
        or "max_completion_tokens"
    )
    storage = os.environ.get("DATASHEETINDEX_LLM_STORE", "").strip().lower()
    if storage in {"", "0", "false", "no", "off"}:
        store = False
    elif storage in {"1", "true", "yes", "on"}:
        store = True
    elif storage == "omit":
        store = None
    else:
        raise ValueError("DATASHEETINDEX_LLM_STORE must be true, false, or omit")
    try:
        return ChatRequestOptions(
            reasoning_effort=reasoning,
            max_output_tokens=budget,
            token_limit_parameter=cast(
                'Literal["max_completion_tokens", "max_tokens"]', parameter
            ),
            store=store,
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid {reasoning_variable} or "
            f"DATASHEETINDEX_TOKEN_LIMIT_PARAMETER: {exc}"
        ) from exc


def text_request_options_env() -> ChatRequestOptions:
    """Resolved text/structured options; also read before artifact cache lookup."""
    return _request_options_env(
        "DATASHEETINDEX_REASONING_EFFORT", "DATASHEETINDEX_MAX_OUTPUT_TOKENS", None
    )


def vision_request_options_env() -> ChatRequestOptions:
    """Resolved vision options, independent of the text model's reasoning effort."""
    return _request_options_env(
        "DATASHEETINDEX_VISION_REASONING_EFFORT",
        "DATASHEETINDEX_VISION_MAX_OUTPUT_TOKENS",
        VISION_MAX_TOKENS,
    )


def create_llm_client(
    model: str | None = None,
    *,
    text_options: ChatRequestOptions | None = None,
    vision_options: ChatRequestOptions | None = None,
    on_completion: Callable[[LlmCallMetadata], None] | None = None,
) -> LlmCallable:
    """Create a sync LLM callable backed by the OpenAI-compatible gateway.

    Reads ``LITELLM_BASE_URL`` and ``LITELLM_MASTER_KEY`` from the
    environment (loading ``.env`` via python-dotenv if available).
    TLS verification is **on** unless ``LITELLM_TLS_VERIFY`` explicitly turns
    it off (``0``/``false``/``no``/``off``). A gateway presenting a certificate
    the local trust store does not accept -- a self-signed cert on a proxy you
    run yourself is the usual case -- is the reason that opt-out exists; prefer
    adding its CA to the trust store, because the key travels on this channel.
    Request timeout and retry policy can be tuned with
    ``LITELLM_TIMEOUT_SECONDS`` and ``LITELLM_MAX_RETRIES``.

    The text model resolves as ``model`` > ``DATASHEETINDEX_MODEL`` >
    ``DEFAULT_TEXT_MODEL``: an explicit argument is a per-call decision and
    outranks the deployment's default. ``model=None`` is therefore "resolve it",
    not "use the built-in default" -- a caller with nothing to say must pass
    nothing, or it silently overrides the deployment.

    Every call goes over Chat Completions. Figure captioning uses a different
    *model* when ``DATASHEETINDEX_VISION_MODEL`` is set, but never a different
    transport -- see ``_ManagedLlmClient.describe_image`` for the measurement
    behind that.

    Returns a ``(system: str, user: str) -> str`` callable. The returned object
    also exposes ``close()`` for explicit cleanup when the caller owns it.

    Explicit options replace environment options for that role. Reasoning is
    omitted by default; output limits use ``max_completion_tokens`` by default,
    with ``max_tokens`` available for older gateways. Requests send ``store=False``
    unless configured otherwise. Only the SDK retries requests: ``max_retries``
    means that many retries after the initial attempt, for all three call shapes.

    ``on_completion`` receives immutable metadata without document payloads for
    each returned response, including empty/refused ones, but not terminal API
    errors. It may be called concurrently by caption workers; the callback must
    be thread-safe. Callback exceptions are logged without their text and ignored.
    """
    ensure_dotenv_loaded()

    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM_MASTER_KEY")

    if not base_url or not api_key:
        raise LlmCredentialsMissingError(
            "LITELLM_BASE_URL and LITELLM_MASTER_KEY must be set "
            "in the environment or .env file"
        )

    tls_verify = _parse_tls_verify_env(os.environ.get("LITELLM_TLS_VERIFY"))
    timeout_seconds = _parse_timeout_seconds_env(
        os.environ.get("LITELLM_TIMEOUT_SECONDS")
    )
    max_retries = _parse_max_retries_env(os.environ.get("LITELLM_MAX_RETRIES"))
    resolved_text_options = (
        text_options if text_options is not None else text_request_options_env()
    )
    resolved_vision_options = (
        vision_options if vision_options is not None else vision_request_options_env()
    )
    resolved_model = (model or "").strip() or text_model_env() or DEFAULT_TEXT_MODEL
    resolved_vision_model = vision_model_env()
    # `httpx2` is httpx 2.x under its new distribution name, and it is openai
    # 3.x's own transport -- the injected `http_client` comes from the same
    # library the SDK is built on rather than a second HTTP stack installed
    # only for this line. Imported lazily, like `openai`, so `[llm]` stays
    # optional; the extra pins the two together.
    httpx2 = importlib.import_module("httpx2")
    openai = importlib.import_module("openai")
    http_client = httpx2.Client(verify=tls_verify, timeout=timeout_seconds)
    try:
        client = openai.OpenAI(
            base_url=base_url.rstrip("/") + "/v1",
            api_key=api_key,
            http_client=http_client,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
    except BaseException:
        # Ownership has not reached the managed client/finalizer yet. Preserve
        # the construction error even if best-effort cleanup itself fails.
        with suppress(Exception):
            _close_resource(http_client)
        raise
    return _ManagedLlmClient(
        # The SDK's Completions.create spells out every request field as a named
        # parameter, so it cannot structurally satisfy a protocol that forwards
        # **kwargs -- even though it accepts every field we pass. Cast at the
        # SDK boundary rather than weaken _ChatCompletionsApi for our own callers.
        chat_api=cast("_ChatCompletionsApi", client.chat.completions),
        http_client=http_client,
        # Stripped: a name with stray whitespace is always a mistake, and
        # sending it verbatim turns it into the gateway's error rather than
        # ours. Matches what both env readers already do with their values.
        model=resolved_model,
        vision_model=resolved_vision_model,
        text_options=resolved_text_options,
        vision_options=resolved_vision_options,
        on_completion=on_completion,
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
