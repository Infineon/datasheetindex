"""Chamber-grounded benchmark engine.

Two extraction paths against the same `ClaimSpec -> ClaimResult` contract:

* `extract_chamber_baseline` -- single-pass PDF -> LLM -> JSON, no tools.
  This is the non-agentic baseline committed in the methodology doc.
* `extract_chamber_agentic` -- an explicit manual loop over the Anthropic
  Messages API with per-step trace recording. It replaces `tool_runner`
  precisely so that per-call instrumentation is observable, which is what the
  dispatch-level detector rules are predicates over.

In the private repository this module was deliberately kept separate from
the production `extract_lite.py`; here that shared half is `datasheet_tools.py`,
and this module reuses its client setup, PDF block builder, schema dereferencer
and JSON-text parser via direct imports.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from chamberbench.claims import (
    ClaimResult,
    ClaimSpec,
    Engine,
    ReasoningKind,
    TraceStep,
)
from chamberbench.claimsio import corpus_dir
from chamberbench.harness.chamber_tools import _make_chamber_tools
from chamberbench.harness.datasheet_tools import (
    _PARAMETER_RESULT_DEFAULTS,
    InspectDetail,
    _build_pdf_content_block,
    _create_client,
    _make_large_pdf_tools,
    _parse_json_from_text,
)

logger = logging.getLogger(__name__)


def _dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all $ref/$defs in a JSON schema for SDK compatibility.

    The SDK's StructuredOutput validator doesn't support $defs/$ref.
    This resolves all references into inline definitions.
    """
    schema = dict(schema)  # don't mutate the caller's dict
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]  # e.g. "#/$defs/NumericalValue"
                ref_name = ref_path.rsplit("/", 1)[-1]
                if ref_name in defs:
                    # Return a resolved copy of the referenced definition
                    return _resolve(dict(defs[ref_name]))
                return node
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


def _make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a JSON schema to OpenAI / Azure structured-output strict mode.

    Strict mode imposes two rules per object node that the LiteLLM Anthropic-
    shape passthrough does *not* enforce on Anthropic models but does on
    Azure-OpenAI ones:

    * ``additionalProperties`` must be present and ``False``.
    * ``required`` must list every key in ``properties`` -- optionality is
      expressed via nullable types, not via omission from ``required``. Fields
      with Pydantic defaults remain semantically optional because the default
      round-trips through schema serialization, but the strict validator wants
      them named in ``required`` regardless.

    For Pydantic models declared with ``extra: forbid`` and no optional fields
    this is a no-op; for everything else it's a defensive upgrade. Sibling to
    ``_dereference_schema`` and idempotent, so callers can compose the two in
    either order. Apply this to schemas the chamber agent loop dispatches
    across providers via the LiteLLM Anthropic-shape passthrough.

    Object detection is robust to three Pydantic emission shapes: scalar
    ``"type": "object"``, union form ``"type": ["object", "null"]`` (from
    ``Optional[NestedModel]``), and the bare-``properties`` form (no explicit
    ``type``) that some hand-written schemas use.
    """

    def _is_object_node(node: dict[str, Any]) -> bool:
        node_type = node.get("type")
        if node_type == "object":
            return True
        if isinstance(node_type, list) and "object" in node_type:
            return True
        # Bare-properties form: dict-shaped `properties` and no explicit type.
        return node_type is None and isinstance(node.get("properties"), dict)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            updated = {k: _walk(v) for k, v in node.items()}
            if _is_object_node(updated):
                if "additionalProperties" not in updated:
                    updated["additionalProperties"] = False
                props = updated.get("properties")
                if isinstance(props, dict) and props:
                    # Strict mode requires every property to be listed in
                    # `required` -- defaults still round-trip via the schema
                    # but the validator wants the explicit declaration.
                    updated["required"] = list(props.keys())
            return updated
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    return _walk(schema)


TraceSink = Callable[[TraceStep], None]


# Exceptions that warrant a bounded retry inside both the baseline and the
# per-turn agentic stream. ValueError is *not* in this set for the agentic
# path (it carries the terminal `max_tokens` signal), but the baseline adds
# it explicitly to its own decorator to cover degenerate-response cases.
_RETRYABLE = (
    ConnectionError,
    OSError,
    TimeoutError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


def _null_trace_sink(_: TraceStep) -> None:
    pass


def _is_openai_model(model: str) -> bool:
    """True for OpenAI models, which route through the Responses API.

    OpenAI reasoning models return no reasoning content through the
    LiteLLM Anthropic-shape passthrough (it uses Chat Completions); the
    Responses-API fork in ``openai_path.py`` handles them instead.
    Anthropic and vLLM (qwen) models stay on the Anthropic path.
    """
    m = model.lower()
    return m.startswith(("gpt-", "o3", "o4"))


def _is_claude_model(model: str) -> bool:
    """True for Anthropic Claude models.

    Claude Sonnet/Opus 4.6+ use adaptive thinking driven by the ``effort``
    knob; ``budget_tokens`` is deprecated for them. vLLM models (qwen)
    reached through the same Anthropic-shape passthrough only understand
    ``budget_tokens``, so the two need different per-turn thinking configs.
    """
    return model.lower().startswith("claude")


def _is_pdf_native(model: str) -> bool:
    """True when the model's backend can ingest a PDF ``document`` block.

    Claude (the Anthropic backend) accepts a native ``document`` content
    block, so the non-agentic baseline attaches the datasheet PDF
    directly. vLLM-hosted models (qwen) cannot: LiteLLM's Anthropic-shape
    passthrough tries to image-decode the raw PDF bytes and fails
    ("cannot identify image file"). VL-capable vLLM models do accept
    images, so for those the baseline renders every page to an image
    block instead -- see ``_build_datasheet_image_blocks``.

    Only Claude and vLLM models reach this check; OpenAI routes to
    ``openai_path.py``, whose Responses-API baseline sends a native
    ``input_file``.
    """
    return _is_claude_model(model)


# ---------------------------------------------------------------------------
# Submit-tool finalization (hybrid output_format replacement)
# ---------------------------------------------------------------------------


class SubmitToolNotCalledError(Exception):
    """The agentic loop ended without calling ``submit_claim_result``.

    Retryable: a fresh run usually recovers because the prompt strongly
    instructs the call. Distinct from ``ValueError`` so the retry policy
    does not also catch ``pydantic.ValidationError`` (a subclass), which
    is *not* retryable -- the same payload would fail the same way.
    """


def _make_capture_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    ack: str,
) -> tuple[Any, Callable[[], dict[str, Any] | None]]:
    """Build a ``beta_async_tool`` that captures its kwargs as a payload.

    Generalises the old ``_make_submit_claim_tool``: the model finalises a
    phase by calling this tool instead of emitting JSON in a text block.
    The submit-tool channel (rather than ``extra_body.output_format``)
    composes with tool use uniformly across backends -- the LiteLLM
    Anthropic-shape passthrough otherwise translates ``output_format`` into
    vLLM guided decoding, whose token mask suppresses ``tool_use`` emission
    from turn 0 (the qwen tool-bypass wedge of the worked example).

    Returns ``(tool, get_captured)``; ``get_captured()`` returns the captured
    kwargs dict (or None if the model never called the tool).
    """
    from anthropic.lib.tools import beta_async_tool

    captured: dict[str, dict[str, Any]] = {}

    @beta_async_tool(name=name, description=description, input_schema=input_schema)
    async def _capture(**kwargs: Any) -> str:
        captured["payload"] = kwargs
        return ack

    def get_captured() -> dict[str, Any] | None:
        return captured.get("payload")

    return _capture, get_captured


def _make_submit_extraction_tool(
    output_schema: dict[str, Any],
) -> tuple[Any, Callable[[], dict[str, Any] | None]]:
    """Phase-1 finalization: freezes the datasheet-side extraction.

    ``output_schema`` is the ClaimResult schema *minus*
    ``expected_chamber_outcome`` (see :func:`_build_extraction_schema`), so
    this tool cannot carry a chamber prediction. Once captured, the
    extracted value is immutable -- chamber tools are only exposed to the
    model after this call, so chamber data cannot have influenced it.
    """
    return _make_capture_tool(
        name="submit_extraction",
        description=(
            "Submit your FINAL datasheet-side extraction. Call this tool "
            "exactly once, after you have located the claim, extracted its "
            "value and conditions, and cross-checked at least one secondary "
            "location. Pass each field as a top-level argument (claim_id, "
            "found, confidence, reason, extracted, extracted_conditions). "
            "Submitting FREEZES the extraction -- it cannot be changed -- and "
            "only then are chamber tools made available to you."
        ),
        input_schema=output_schema,
        ack=(
            "Extraction recorded and frozen. You now have chamber tools to "
            "predict the measurement; finish by calling submit_chamber_outcome."
        ),
    )


def _make_submit_chamber_outcome_tool(
    output_schema: dict[str, Any],
) -> tuple[Any, Callable[[], dict[str, Any] | None]]:
    """Phase-2 finalization: records the (ungraded) chamber prediction.

    ``output_schema`` carries only ``{claim_id, expected_chamber_outcome}``
    (see :func:`_build_chamber_outcome_schema`); it has no ``extracted`` or
    ``found`` property, so phase 2 structurally cannot rewrite the frozen
    extraction.
    """
    return _make_capture_tool(
        name="submit_chamber_outcome",
        description=(
            "Submit your predicted chamber measurement under the frozen "
            "claim's stated conditions, as expected_chamber_outcome (plain "
            "language). Submit an empty string if no chamber experiment "
            "applies. This prediction is a sanity signal, not graded. "
            "Calling this tool ends the task."
        ),
        input_schema=output_schema,
        ack="Chamber prediction recorded. The task is complete -- do not call any more tools.",
    )


# ---------------------------------------------------------------------------
# PDF resolution
# ---------------------------------------------------------------------------


# Default location for cached chamber datasheets: `benchmark/corpus/`, which
# the repository-root .gitignore excludes as a directory (the repo-wide *.pdf
# rule alone would not cover the `.pdf.part` temp file written below). The
# committed claims.yaml therefore stays URL-referenced and no third-party
# datasheet is ever accidentally committed.
_CHAMBER_PDF_CACHE = Path(
    os.environ.get(
        "CHAMBER_PDF_CACHE",
        str(corpus_dir()),
    )
)


def _resolve_pdf_to_local(pdf_source: str) -> str:
    """Resolve a benchmark `pdf_source` to a local file path.

    Reasons we don't use `DatasheetIndex._resolve_pdf_source()` directly: it
    rejects PDFs served with `application/octet-stream`, which is exactly
    what GitHub raw URLs return for PDF blobs. The corpus datasheets are
    mirrored in Juan Gamella's public Causal Chambers repository and are
    fetched from GitHub raw, so we download with `requests` (lenient on
    content type) and cache under `_CHAMBER_PDF_CACHE`.

    If `pdf_source` is already a local file, returns it unchanged.
    """
    if not pdf_source.lower().startswith(("http://", "https://")):
        if not Path(pdf_source).exists():
            raise FileNotFoundError(f"Local PDF not found: {pdf_source}")
        return pdf_source

    _CHAMBER_PDF_CACHE.mkdir(parents=True, exist_ok=True)
    # Use a hash of the URL plus the URL's last path segment for readability.
    digest = hashlib.sha256(pdf_source.encode("utf-8")).hexdigest()[:12]
    suffix = Path(pdf_source.split("?", 1)[0]).name or "file.pdf"
    cache_path = _CHAMBER_PDF_CACHE / f"{digest}_{suffix}"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return str(cache_path)

    import requests

    logger.info("Downloading datasheet to cache: %s -> %s", pdf_source, cache_path)
    # Atomic write: download to a sibling .part file, then rename. Avoids
    # corrupted blobs if multiple parallel claim runs race on the
    # same URL. The URL hash in the filename makes collisions impossible
    # across distinct sources, so we only need to guard the same-URL case.
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
    resp = requests.get(pdf_source, stream=True, timeout=60)
    resp.raise_for_status()
    with open(tmp_path, "wb") as fh:
        fh.writelines(resp.iter_content(chunk_size=65536))
    # If a sibling task got here first, prefer the existing complete cache.
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return str(cache_path)
    tmp_path.replace(cache_path)
    return str(cache_path)


def _build_datasheet_image_blocks(
    pdf_path: str, *, detail: Literal["low", "medium", "high"] = "low"
) -> list[dict[str, Any]]:
    """Render every datasheet page to an Anthropic image content block.

    Used by the non-agentic baseline for VL-capable models whose backend
    cannot ingest a PDF but does accept images (vLLM-hosted qwen). This
    mirrors what the Anthropic and OpenAI backends do internally with a
    native PDF -- rasterise every page -- so the baseline input stays a
    rendered document across all three providers rather than degrading
    qwen to a text-only transcription that drops tables and figures.

    ``detail`` sets the render dpi (low/medium/high, via datasheetindex
    ``inspect_page``). The default is "low", matching the chamber agentic
    config for qwen: a chamber datasheet runs up to 65 pages, and at
    "medium" that many page images pushes a single request past the
    client timeout. "low" keeps the call tractable (~725 vision tokens
    per page); raise it per run if OCR fidelity proves insufficient.

    Blocking (PDF render) -- call via ``asyncio.to_thread``.
    """
    from datasheetindex import DatasheetTools

    blocks: list[dict[str, Any]] = []
    with DatasheetTools(pdf_path) as tools:
        n_pages = len(tools.doc)
        for page in range(1, n_pages + 1):
            for item in tools.inspect_page(page, detail=detail):
                if item.get("type") == "image":
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": item.get("mime_type", "image/png"),
                                "data": item["data"],
                            },
                        }
                    )
    if not blocks:
        raise ValueError(f"no page images rendered from {pdf_path}")
    return blocks


# ---------------------------------------------------------------------------
# System / user prompts
# ---------------------------------------------------------------------------


_AGENTIC_SYSTEM = """\
You are a meticulous datasheet-extraction assistant for a chamber-grounded
benchmark. You work in TWO PHASES, and your tools change between them.

PHASE 1 -- Datasheet extraction. You have datasheet-navigation tools only:
    build_datasheet, get_section_text, search_text,
    extract_table_markdown, inspect_page.
  1. Always call build_datasheet first to load the document and get the
     enriched ToC, then navigate to the relevant section.
  2. Extract the numeric value(s), unit, and every operating condition.
     Missing a condition is the most common failure mode.
  3. Cross-check at least one secondary location (a table, an electrical-
     characteristics section, or a figure caption).
  4. Finalize PHASE 1 by calling `submit_extraction` exactly once with the
     structured result as its arguments.
  Submitting FREEZES your datasheet extraction: it cannot be changed
  afterward, and only then are chamber tools made available. You cannot
  see chamber data before you submit, so it cannot influence the value
  you extract. If the claim is genuinely not in the datasheet, call
  submit_extraction with found=false, an honest confidence, and a reason.

PHASE 2 -- Chamber prediction (only when found=true). You now have the
chamber tools instead:
    list_experiments, get_experiment_metadata, query_dataset,
    cross_sensor_check, run_simulator, get_ground_truth_graph.
  Use them to predict what a calibrated independent measurement would
  yield under the frozen claim's stated conditions, and submit it via
  `submit_chamber_outcome`. If no chamber experiment applies, submit an
  empty prediction. This prediction is a sanity signal, not graded.

Do NOT emit results as JSON in a text message -- each phase advances only
when you call that phase's submit tool. Do not call further tools after
submit_chamber_outcome.
"""


_BASELINE_SYSTEM = """\
You are a meticulous datasheet-extraction assistant for a chamber-grounded benchmark.
You will be given exactly one claim to verify against a single attached PDF datasheet.

For the given claim:
1. Locate the claim in the datasheet.
2. Extract the numeric value(s) and unit exactly as stated.
3. Extract every operating condition the datasheet attaches to the claim
   (temperature, supply voltage, sample rate, mode, range, etc.). Missing a
   condition is the most common failure mode -- be thorough.
4. Cross-check at least one secondary location in the datasheet (a table, a
   figure caption, or the electrical-characteristics section) to confirm the
   value before finalizing.
5. Return strictly valid JSON conforming to the ClaimResult schema.

If the claim is genuinely not in the datasheet, return found=false with
confidence reflecting your certainty and a clear reason.
"""


def _render_claim_spec_for_prompt(claim: ClaimSpec) -> str:
    """Single chokepoint for serialising a ClaimSpec into prompt JSON.

    All agent-facing prompt builders MUST route through this helper to
    guarantee they cannot diverge on what the model sees. The projection
    drops oracle fields (source_page, source_text, value_contains,
    claimed_*, tolerance_*, description, ...) that the fidelity scorer
    grades against -- see ``ClaimSpec.agent_visible_dump`` for the
    allowlist rationale.
    """
    return json.dumps(claim.agent_visible_dump(), indent=2)


def _build_baseline_user_prompt(claim: ClaimSpec) -> str:
    """User prompt for the non-agentic baseline."""
    spec_json = _render_claim_spec_for_prompt(claim)
    return (
        "Verify the following claim against the attached PDF datasheet. "
        "Return a single ClaimResult JSON object as your final structured output.\n\n"
        f"Claim spec:\n```json\n{spec_json}\n```\n\n"
        "Notes:\n"
        f"- Set claim_id to '{claim.id}'.\n"
        "- Populate `extracted` (a ParameterResult) with the value(s) you "
        "find, with conditions strings recording all stated operating "
        "conditions verbatim.\n"
        "- Populate `extracted_conditions` with the structured form of every "
        "operating condition you can identify, mapping to the names in "
        "operating_conditions when possible.\n"
        "- `expected_chamber_outcome` is your prediction (in plain language) "
        "of what a calibrated independent measurement under the stated "
        "conditions should produce.\n"
        "- Set `confidence` honestly; this is calibration data."
    )


# ---------------------------------------------------------------------------
# Schema + sanitization
# ---------------------------------------------------------------------------


def _build_claim_result_schema() -> dict[str, Any]:
    # Strict mode (`additionalProperties: false` everywhere) is required by
    # OpenAI / Azure structured-output validators -- exercised by Slice B's
    # cross-provider runs through the LiteLLM passthrough. No-op for Anthropic.
    return _make_strict_schema(_dereference_schema(ClaimResult.model_json_schema()))


def _build_extraction_schema() -> dict[str, Any]:
    """Phase-1 ``submit_extraction`` schema: ClaimResult minus the chamber field.

    Derived from :func:`_build_claim_result_schema` so the nested
    ``ParameterResult`` / ``OperatingCondition`` shapes stay byte-identical to
    the single-pass tool. ``expected_chamber_outcome`` is dropped from both
    ``properties`` and ``required`` -- the freeze tool cannot carry a chamber
    prediction.
    """
    schema = copy.deepcopy(_build_claim_result_schema())
    schema.get("properties", {}).pop("expected_chamber_outcome", None)
    if isinstance(schema.get("required"), list):
        schema["required"] = [
            r for r in schema["required"] if r != "expected_chamber_outcome"
        ]
    return schema


def _build_chamber_outcome_schema() -> dict[str, Any]:
    """Phase-2 ``submit_chamber_outcome`` schema: ``{claim_id, expected_chamber_outcome}``.

    Deliberately has no ``extracted`` / ``found`` / ``confidence`` property and
    ``additionalProperties: false``, so phase 2 structurally cannot mutate the
    frozen extraction.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claim_id": {"type": "string"},
            "expected_chamber_outcome": {"type": "string"},
        },
        "required": ["claim_id", "expected_chamber_outcome"],
    }


def _sanitize_claim_result_data(data: dict[str, Any]) -> None:
    """Clean up model output to match ClaimResult schema expectations.

    Normalises text-channel model output for the ClaimResult shape:
    top-level fields, nested ``extracted`` ParameterResult, and nested
    ``extracted_conditions`` OperatingCondition list.

    Defensive against malformed model output: qwen via vLLM does not
    strictly schema-validate tool-call inputs, so it can emit a string
    where a nested object / list-of-objects is expected. Non-dict /
    non-list values are left untouched here so the shape mismatch surfaces
    as a clean ``pydantic.ValidationError`` at ``model_validate`` -- rather
    than crashing this sanitizer with an ``AttributeError``.
    """
    if data.get("reason") is None:
        data["reason"] = ""
    if data.get("expected_chamber_outcome") is None:
        data["expected_chamber_outcome"] = ""

    extracted = data.get("extracted")
    if extracted is None:
        extracted = {}
    if isinstance(extracted, dict):
        for field, default in _PARAMETER_RESULT_DEFAULTS.items():
            if extracted.get(field) is None:
                extracted[field] = default
        values = extracted.get("values")
        if isinstance(values, list):
            for cv in values:
                if not isinstance(cv, dict):
                    continue
                for field in ("conditions", "unit", "source_text"):
                    if cv.get(field) is None:
                        cv[field] = ""
                for field in ("min_value", "max_value", "typical_value"):
                    if cv.get(field) is None:
                        cv[field] = -999.0
        data["extracted"] = extracted

    conditions = data.get("extracted_conditions")
    if isinstance(conditions, list):
        for oc in conditions:
            if not isinstance(oc, dict):
                continue
            for field in ("unit", "chamber_variable"):
                if oc.get(field) is None:
                    oc[field] = ""
            for field in ("value", "min_value", "max_value"):
                if oc.get(field) is None:
                    oc[field] = -999.0
            if oc.get("load_bearing") is None:
                oc["load_bearing"] = True


# ---------------------------------------------------------------------------
# Non-agentic baseline
# ---------------------------------------------------------------------------


@retry(
    # Same retryable set as `_RETRYABLE` plus ValueError to cover the
    # "Model returned no text" path the baseline hits on degenerate
    # gateway responses; the agentic loop does not retry ValueError
    # since that includes its terminal `max_tokens` signal.
    retry=retry_if_exception_type(_RETRYABLE + (ValueError,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    reraise=True,
)
async def _extract_chamber_baseline_anthropic(
    claim: ClaimSpec,
    *,
    model: str = "claudesonnet4.6",
    max_tokens: int = 32768,
    trace_sink: TraceSink | None = None,
) -> ClaimResult:
    """Single-pass non-agentic baseline.

    The PDF is sent as a native document content block. No tools, no loop.
    Returns a `ClaimResult` for the supplied claim.

    If a `trace_sink` is supplied, one `final_output` `TraceStep` is emitted
    on success carrying the API call's input/output/cache token counts. Cost
    analyses downstream (e.g. agentic-vs-baseline comparison) read these
    from the same trace-event surface the agentic loop already populates.
    """
    pdf_path = _resolve_pdf_to_local(claim.pdf_source)
    client, http_client = _create_client()
    output_schema = _build_claim_result_schema()

    # PDF-native backends (Claude) take the datasheet as one `document`
    # block. vLLM-hosted models (qwen) cannot ingest a PDF but can read
    # images, so they get every page rendered to an image block -- the
    # same rendered-document input, not a text-only downgrade. See
    # `_is_pdf_native`.
    prompt_block: dict[str, Any] = {
        "type": "text",
        "text": _build_baseline_user_prompt(claim),
    }
    user_content: list[dict[str, Any]]
    if _is_pdf_native(model):
        user_content = [_build_pdf_content_block(pdf_path, model), prompt_block]
    else:
        page_images = await asyncio.to_thread(_build_datasheet_image_blocks, pdf_path)
        user_content = [*page_images, prompt_block]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    run_id = new_run_id(model)
    t0 = time.monotonic()
    try:
        async with client.beta.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=_BASELINE_SYSTEM,
            messages=messages,
            extra_body={
                "output_format": {
                    "type": "json_schema",
                    "schema": output_schema,
                },
            },
        ) as stream:
            response = await stream.get_final_message()
    finally:
        try:
            await client.close()
        except Exception:
            logger.debug("Error closing Anthropic client", exc_info=True)
        if http_client is not None:
            try:
                await http_client.aclose()
            except Exception:
                logger.debug("Error closing HTTP client", exc_info=True)

    latency_ms = int((time.monotonic() - t0) * 1000)

    text = _extract_response_text(response)
    if not text:
        stop_reason = getattr(response, "stop_reason", "?")
        raise ValueError(f"Baseline returned no text (stop_reason={stop_reason})")

    data = _parse_json_from_text(text)
    data.setdefault("claim_id", claim.id)
    _sanitize_claim_result_data(data)
    claim_result = ClaimResult.model_validate(data)

    # Emit a single final_output TraceStep so cost-side roll-ups can read
    # baseline usage from the same surface as agentic events. `step` and
    # `turn_idx` are 0 -- the baseline is by definition single-turn.
    if trace_sink is not None:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        trace_sink(
            TraceStep(
                run_id=run_id,
                claim_id=claim.id,
                engine="baseline",
                step=0,
                turn_idx=0,
                kind="final_output",
                agent_reasoning=text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
            )
        )

    return claim_result


def _extract_response_text(message: Any) -> str:
    parts = []
    for block in message.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Manual agentic loop
# ---------------------------------------------------------------------------


def _build_agentic_user_prompt(claim: ClaimSpec) -> str:
    spec_json = _render_claim_spec_for_prompt(claim)
    return (
        "PHASE 1 -- extract the following claim from the datasheet, using the "
        "datasheet tools as needed, then call submit_extraction. You will "
        "receive chamber tools only after you submit.\n\n"
        f"Claim spec:\n```json\n{spec_json}\n```\n\n"
        "Notes:\n"
        f"- Set claim_id to '{claim.id}'.\n"
        "- Use build_datasheet first to load the PDF.\n"
        "- Cross-check the claim against at least two locations in the "
        "datasheet before submitting (e.g. the Features section AND the "
        "electrical-characteristics table).\n"
        "- Set `confidence` honestly; this is calibration data."
    )


def _build_chamber_handoff_prompt(claim: ClaimSpec, extraction: dict[str, Any]) -> str:
    """Phase-2 hand-off injected at the freeze.

    Echoes only the agent's *own* frozen output (no oracle field), so it
    opens no leak surface beyond what the agent already produced.
    """
    frozen = json.dumps(
        {
            "claim_id": claim.id,
            "found": extraction.get("found"),
            "extracted": extraction.get("extracted"),
            "extracted_conditions": extraction.get("extracted_conditions"),
        },
        indent=2,
        default=str,
    )
    return (
        "PHASE 2 -- chamber prediction. Your datasheet extraction is now FROZEN "
        "and cannot be changed:\n"
        f"```json\n{frozen}\n```\n\n"
        "Using the chamber tools, predict what a calibrated independent "
        "measurement would yield under the stated operating conditions, and "
        f"submit it with submit_chamber_outcome (set claim_id to '{claim.id}'). "
        "If no chamber experiment applies, submit expected_chamber_outcome as an "
        "empty string. This prediction is a sanity signal and is not graded."
    )


def _split_response(message: Any) -> tuple[str, ReasoningKind, list[Any]]:
    """Split an Anthropic response into (reasoning_text, reasoning_kind, tool_uses).

    `reasoning_text` is the agent's extended-thinking content when the
    response carries `thinking` blocks (extended thinking is enabled on
    this engine); otherwise it is the visible text preamble.
    `reasoning_kind` records which -- "thinking" or "text". Thinking
    blocks expose `.thinking`, not `.text`, so they need an explicit
    branch; before this they were silently dropped and only the preamble
    reached `agent_reasoning`.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_uses: list[Any] = []
    for block in message.content:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            tool_uses.append(block)
        elif btype == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
        elif hasattr(block, "text"):
            text_parts.append(block.text)
    if thinking_parts:
        return "\n".join(thinking_parts).strip(), "thinking", tool_uses
    return "\n".join(text_parts).strip(), "text", tool_uses


def _is_image_payload(value: Any) -> bool:
    """True when `value` is a list of Anthropic-style image content blocks.

    Used by the trace summarizer (collapse to a marker) and the tool_result
    builder (pass through verbatim so the model sees the image). Single
    discriminator so both paths stay in lock-step.
    """
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], dict)
        and value[0].get("type") == "image"
    )


def _summarize(value: Any, max_chars: int = 1500) -> str:
    """Compact, log-friendly representation of arbitrary tool input/output.

    Image/base64-bearing outputs (e.g. `inspect_page`) are aggressively
    truncated so the JSONL trace stays human-readable.
    """
    if _is_image_payload(value):
        sources = [v.get("source", {}).get("media_type", "?") for v in value]
        return f"<image content x{len(value)}: {','.join(sources)}>"
    try:
        if not isinstance(value, str):
            text = json.dumps(value, default=str)
        else:
            text = value
    except Exception:  # noqa: BLE001
        text = repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 16] + "...<truncated>"
    return text


def _format_tool_error(exc: BaseException) -> str:
    """Sanitized error string passed back to the agent.

    Avoids leaking `repr(exc)` (which can include file paths, tracebacks,
    and dataset internals) to the model. Tracebacks are still logged via
    `logger.exception`.
    """
    return f"ERROR: {type(exc).__name__}: {exc}"


async def _dispatch_tool(tool_use: Any, by_name: dict[str, Any]) -> Any:
    """Invoke the named beta_async_tool with the provided input dict."""
    tool = by_name.get(tool_use.name)
    if tool is None:
        raise KeyError(f"unknown tool: {tool_use.name}")
    return await tool.call(dict(tool_use.input))


def _tool_result_block(tool_use_id: str, output: Any, is_error: bool) -> dict:
    """Build a `tool_result` content block for the next user turn."""
    if _is_image_payload(output):
        # Image content (from inspect_page) passes through directly so the
        # model can see it on the next turn.
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output,
            "is_error": is_error,
        }
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": output if isinstance(output, str) else str(output),
        "is_error": is_error,
    }


# ---------------------------------------------------------------------------
# Per-turn execution with bounded retry
# ---------------------------------------------------------------------------

# Cap on per-turn `max_tokens` for the non-streaming vLLM path. The Anthropic
# SDK refuses a non-streaming `.create()` whose `max_tokens` could imply a
# >10-minute request -- it raises when `3600 * max_tokens / 128000 > 600`,
# i.e. `max_tokens > 21333`. A single agent turn never needs that many output
# tokens (thinking + one tool call), so qwen turns are capped below the guard.
_VLLM_NONSTREAMING_MAX_TOKENS = 20_000


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    reraise=True,
)
async def _run_one_turn(
    *,
    client: Any,
    model: str,
    max_tokens: int,
    system: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    reasoning_effort: str = "medium",
) -> Any:
    """Run one agent turn and return the final message.

    Retries are bounded per turn (not per whole run) so the messages list
    stays a consistent prefix on transient gateway errors. Non-retryable
    failures (e.g. ValueError, schema problems) bubble immediately.

    Transport and the reasoning knob are model-conditional:

    - **Claude** (Sonnet/Opus 4.6+) uses native adaptive thinking driven
      by the ``effort`` knob, over a *streaming* turn -- the SDK's stream
      accumulator reconstructs Anthropic-native thinking blocks correctly.

    - **vLLM-hosted models (qwen)** on the LiteLLM Anthropic-shape
      passthrough use a *non-streaming* turn. Two gateway facts force this
      (both probed 2026-05-21 against a LiteLLM gateway fronting a
      vLLM-hosted qwen3.6-27b):
        * Reasoning is enabled only by the vLLM chat-template kwarg
          ``enable_thinking``, passed through ``extra_body``. The
          Anthropic-native ``thinking`` param does NOT work here -- LiteLLM
          mistranslates it into a vLLM ``reasoning_effort`` field, which
          never triggers the thinking turn (and 400s on small budgets).
        * The gateway *does* stream thinking, but the Anthropic SDK's
          stream accumulator drops the thinking blocks LiteLLM emits;
          ``.create()`` preserves them.
      ``.create()`` rejects a `max_tokens` that could imply a >10-minute
      request, so qwen turns are capped (see `_VLLM_NONSTREAMING_MAX_TOKENS`).

    Structured output is delivered via the
    ``submit_claim_result`` tool, not ``extra_body.output_format`` -- the
    latter was translated into vLLM ``response_format`` (guided decoding)
    and masked out tool-use tokens from turn 0. See the
    ``_make_submit_claim_tool`` docstring.
    """
    if _is_claude_model(model):
        async with client.beta.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": reasoning_effort},
        ) as stream:
            return await stream.get_final_message()

    # Probe knob: CHAMBER_QWEN_ENABLE_THINKING=false disables reasoning to
    # work around QwenLM/Qwen3 issue #1817 (thinking-mode tool-call drop).
    _think = (
        os.environ.get("CHAMBER_QWEN_ENABLE_THINKING", "true").strip().lower()
        != "false"
    )
    _qwen_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": min(max_tokens, _VLLM_NONSTREAMING_MAX_TOKENS),
        "system": system,
        "tools": tools,
        "messages": messages,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": _think}},
    }
    # Probe knob: CHAMBER_QWEN_TOOL_CHOICE=any forces structured tool decoding.
    # After a gateway-side tool-parser change (2026-06-05) the default
    # `auto` decoding returns tool calls as markdown/XML *text* that the
    # gateway no longer extracts into a tool_use block -> the loop sees a
    # terminal turn with no tool. Forcing `any` makes the gateway emit a
    # proper tool_use block again. Default unset keeps prior behavior.
    _tc = os.environ.get("CHAMBER_QWEN_TOOL_CHOICE", "").strip().lower()
    if _tc == "any":
        _qwen_kwargs["tool_choice"] = {"type": "any"}
    elif _tc == "auto":
        _qwen_kwargs["tool_choice"] = {"type": "auto"}
    return await client.beta.messages.create(**_qwen_kwargs)


# Finalization tools are never fault-injectable: without them a phase has no
# structured-output channel and every run would be a loud engine error rather
# than the silent failure the experiment plants. (`submit_claim_result` is the
# pre-two-pass name, retained so archived callers still resolve.)
_SUBMIT_TOOL_NAMES: frozenset[str] = frozenset(
    {"submit_extraction", "submit_chamber_outcome", "submit_claim_result"}
)

# Soft per-phase turn cap for the ungraded chamber-prediction phase, so a
# pass-2 stall cannot consume the whole turn budget (and the qwen reasoning-mode
# tool-call-drop degrades to an empty prediction rather than an engine error).
_PASS2_TURN_CAP = 8


def _apply_tool_exclusions(
    tools: list[Any], excluded_tools: frozenset[str]
) -> list[Any]:
    """Drop tools whose name is in `excluded_tools`; always keep submit tools.

    Used by the fault-injection experiment
    (`scripts/fault_injection.py`): running the agent with a
    deliberately reduced tool set emulates a model that cannot call those
    tools -- the synthetic, backend-agnostic stand-in for the Section-5
    guided-decoding tool-bypass. The finalization tools
    (`submit_extraction` / `submit_chamber_outcome`) are never excluded.
    With an empty `excluded_tools` this is the identity function, so the
    normal benchmark path is unaffected.
    """
    if not excluded_tools:
        return tools
    return [
        t for t in tools if t.name in _SUBMIT_TOOL_NAMES or t.name not in excluded_tools
    ]


@dataclass
class _TwoPassState:
    """The tool partitioning and capture getters for the two-pass loop.

    Built once per claim run by :func:`_build_two_pass_state` and shared by
    both provider loops so the security-critical split (chamber tools never in
    phase 1) has a single source of truth. The provider loop serialises
    ``pass1_tools`` / ``pass2_tools`` in its own wire format and flips between
    them on the freeze.
    """

    pass1_tools: list[Any]
    pass2_tools: list[Any]
    get_extraction: Callable[[], dict[str, Any] | None]
    get_outcome: Callable[[], dict[str, Any] | None]


def _build_two_pass_state(
    pdf_tools: list[Any],
    ch_tools: list[Any],
    excluded_tools: frozenset[str],
) -> _TwoPassState:
    """Partition tools into the two phases and wire up the freeze/merge captures.

    Phase 1 offers only datasheet tools + ``submit_extraction``; phase 2 offers
    only chamber tools + ``submit_chamber_outcome``. Fault-injection exclusions
    apply to both lists so a withheld tool is withheld everywhere.
    """
    submit_extraction_tool, get_extraction = _make_submit_extraction_tool(
        _build_extraction_schema()
    )
    submit_outcome_tool, get_outcome = _make_submit_chamber_outcome_tool(
        _build_chamber_outcome_schema()
    )
    pass1 = _apply_tool_exclusions([*pdf_tools, submit_extraction_tool], excluded_tools)
    pass2 = _apply_tool_exclusions([*ch_tools, submit_outcome_tool], excluded_tools)
    return _TwoPassState(
        pass1_tools=pass1,
        pass2_tools=pass2,
        get_extraction=get_extraction,
        get_outcome=get_outcome,
    )


def _merge_two_pass_payload(
    claim: ClaimSpec,
    frozen_extraction: dict[str, Any],
    outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the final ClaimResult payload from frozen pass-1 + pass-2 output.

    ``extracted`` / ``found`` / ``confidence`` are read *only* from the frozen
    pass-1 capture; ``expected_chamber_outcome`` is the sole field taken from
    pass 2. A malformed pass-2 payload can at most contribute stray keys the
    merge ignores -- it cannot rewrite the frozen extraction.
    """
    payload = dict(frozen_extraction)
    payload["claim_id"] = claim.id
    payload["expected_chamber_outcome"] = (outcome or {}).get(
        "expected_chamber_outcome", ""
    ) or ""
    _sanitize_claim_result_data(payload)
    return payload


async def _extract_chamber_agentic_anthropic(
    claim: ClaimSpec,
    *,
    model: str = "claudesonnet4.6",
    max_turns: int = 30,
    trace_sink: TraceSink = _null_trace_sink,
    chamber_dataset: str | None = None,
    chamber_config: str = "standard",
    chamber_name: Literal["wt", "lt"] = "wt",
    max_tokens: int = 32768,
    inspect_page_detail: InspectDetail = "high",
    reasoning_effort: str = "medium",
    excluded_tools: frozenset[str] = frozenset(),
) -> ClaimResult:
    """Manual agentic loop with per-step trace recording.

    Replaces `client.beta.messages.tool_runner` (which the non-instrumented
    extraction engine in the private repository uses) with an explicit
    while-loop. Each tool call, agent reasoning text, and
    final structured output is emitted as a `TraceStep` to `trace_sink`,
    enabling the failure-attribution rubric in `chamberbench/classifier.py`.

    Tools wired:
      - datasheetindex (`_make_large_pdf_tools` from datasheet_tools.py)
      - causal chamber (`_make_chamber_tools` from chamber_tools.py)
    """
    if not claim.pdf_source:
        raise ValueError(f"claim {claim.id!r} has empty pdf_source")
    ch_dataset = chamber_dataset or claim.chamber_dataset
    if not ch_dataset:
        raise ValueError(
            f"claim {claim.id!r} has no chamber_dataset; supply one via "
            "ClaimSpec.chamber_dataset or the chamber_dataset arg"
        )

    pdf_path = _resolve_pdf_to_local(claim.pdf_source)
    pdf_tools, pdf_cleanup, _pdf_get_tools = _make_large_pdf_tools(
        pdf_path, inspect_page_detail=inspect_page_detail
    )
    ch_tools, ch_cleanup = _make_chamber_tools(
        chamber=chamber_name,
        config=chamber_config,
        dataset_name=ch_dataset,
    )

    # Two-pass partitioning: phase 1 offers datasheet tools + submit_extraction;
    # the chamber tools and submit_chamber_outcome are withheld until the
    # extracted value is frozen, so chamber data cannot enter context first.
    state = _build_two_pass_state(pdf_tools, ch_tools, excluded_tools)
    phase: Literal["extraction", "chamber"] = "extraction"
    active_tools = state.pass1_tools
    by_name = {t.name: t for t in active_tools}
    tool_param_dicts = [t.to_dict() for t in active_tools]

    run_id = new_run_id(model)
    engine: Engine = "agentic"

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": _build_agentic_user_prompt(claim)},
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    client, http_client = _create_client()
    final_message: Any = None
    step_idx = 0
    turn_idx = 0
    pass2_deadline: int | None = None

    try:
        while turn_idx < max_turns:
            t0 = time.monotonic()
            response = await _run_one_turn(
                client=client,
                model=model,
                max_tokens=max_tokens,
                system=_AGENTIC_SYSTEM,
                tools=tool_param_dicts,
                messages=messages,
                reasoning_effort=reasoning_effort,
            )
            turn_latency_ms = int((time.monotonic() - t0) * 1000)

            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

            reasoning_text, reasoning_kind, tool_uses = _split_response(response)
            stop_reason = getattr(response, "stop_reason", "")

            # Terminal-with-no-tool-uses: the agent ended a turn without
            # calling the phase's submit tool. In phase 1 this is the soft
            # SubmitToolNotCalledError raised below (no frozen extraction); in
            # phase 2 the (ungraded) chamber prediction is simply absent and we
            # finalize with the frozen extraction. Either way emit a
            # final_output marker so downstream rubric consumers see the stop.
            terminal_stop = stop_reason in ("end_turn", "stop_sequence")
            if terminal_stop and not tool_uses:
                # In phase 2 the graded extraction is already frozen, so ending
                # without a chamber prediction is a SUCCESS -- emit a clean
                # terminal marker (no error) so the rubric classifier does not
                # mislabel the cell as a failure. In phase 1 it is the genuine
                # SubmitToolNotCalledError raised after the loop.
                trace_sink(
                    TraceStep(
                        run_id=run_id,
                        claim_id=claim.id,
                        engine=engine,
                        phase=phase,
                        step=step_idx,
                        turn_idx=turn_idx,
                        kind="final_output",
                        agent_reasoning=reasoning_text,
                        reasoning_kind=reasoning_kind,
                        error=("" if phase == "chamber" else "terminal_without_submit"),
                        latency_ms=turn_latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read,
                        cache_creation_tokens=cache_creation,
                    )
                )
                final_message = response
                break

            # max_tokens is treated as terminal regardless of tool_uses. In
            # phase 1 (graded) fail loudly with the partial trace; in phase 2
            # the chamber prediction is ungraded, so keep the frozen extraction
            # and finalize rather than discard a good pass-1 result.
            if stop_reason == "max_tokens":
                trace_sink(
                    TraceStep(
                        run_id=run_id,
                        claim_id=claim.id,
                        engine=engine,
                        phase=phase,
                        step=step_idx,
                        turn_idx=turn_idx,
                        kind="final_output",
                        agent_reasoning=reasoning_text,
                        reasoning_kind=reasoning_kind,
                        # phase 1 raises below; phase 2 succeeds with the frozen
                        # extraction, so its terminal marker carries no error.
                        error=("" if phase == "chamber" else "stop_reason=max_tokens"),
                        latency_ms=turn_latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read,
                        cache_creation_tokens=cache_creation,
                    )
                )
                if phase == "extraction":
                    raise ValueError(
                        f"agent stopped at max_tokens (turn {turn_idx}); raise max_tokens or simplify the claim"
                    )
                final_message = response
                break

            # Dispatch every tool_use sequentially with per-call timing.
            # Reasoning + per-turn usage are duplicated to every TraceStep
            # in the turn so the rubric classifier can group by turn_idx
            # without losing context on the second call.
            tool_result_blocks: list[dict] = []
            for tu in tool_uses:
                # A tool outside the active phase set is not dispatched and
                # leaves no trace step: the chamber tools before the freeze
                # (the structural enforcement of fidelity independence) and
                # fault-injected exclusions both land here. Feed back an error
                # tool_result so the message list stays valid; the dispatch
                # record must show the tool was never called and no tool output
                # entered the context. A normal phase-appropriate call passes.
                if tu.name not in by_name:
                    tool_result_blocks.append(
                        _tool_result_block(
                            getattr(tu, "id", ""),
                            "ERROR: tool not available",
                            True,
                        )
                    )
                    continue
                t1 = time.monotonic()
                err: str | None = None
                output: Any
                try:
                    output = await _dispatch_tool(tu, by_name)
                except Exception as exc:
                    err = _format_tool_error(exc)
                    output = err
                    logger.exception("tool %s failed", tu.name)
                tool_result_blocks.append(
                    _tool_result_block(getattr(tu, "id", ""), output, err is not None)
                )
                trace_sink(
                    TraceStep(
                        run_id=run_id,
                        claim_id=claim.id,
                        engine=engine,
                        phase=phase,
                        step=step_idx,
                        turn_idx=turn_idx,
                        kind="tool_call",
                        tool_name=tu.name,
                        tool_input_summary=_summarize(dict(tu.input)),
                        tool_output_summary=_summarize(output),
                        error=err or "",
                        agent_reasoning=reasoning_text,
                        reasoning_kind=reasoning_kind,
                        latency_ms=int((time.monotonic() - t1) * 1000),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read,
                        cache_creation_tokens=cache_creation,
                    )
                )
                step_idx += 1

            messages.append({"role": "assistant", "content": response.content})

            # Phase-1 freeze: submit_extraction was captured this turn. Its own
            # tool_call event is already recorded above; here we either end the
            # run (found=false) or hand off to phase 2.
            froze_now = phase == "extraction" and state.get_extraction() is not None
            enter_pass2 = False
            if froze_now:
                frozen = state.get_extraction() or {}
                enter_pass2 = bool(frozen.get("found", False))
                if enter_pass2:
                    # Hand off to phase 2 in the SAME user turn as the tool
                    # results -- the Messages API requires alternating roles, so
                    # a separate user message would be rejected.
                    tool_result_blocks.append(
                        {
                            "type": "text",
                            "text": _build_chamber_handoff_prompt(claim, frozen),
                        }
                    )

            messages.append({"role": "user", "content": tool_result_blocks})

            if froze_now and not enter_pass2:
                # found=false: no chamber phase. This is the terminal event.
                trace_sink(
                    TraceStep(
                        run_id=run_id,
                        claim_id=claim.id,
                        engine=engine,
                        phase=phase,
                        step=step_idx,
                        turn_idx=turn_idx,
                        kind="final_output",
                        agent_reasoning="submit_extraction invoked (found=false; no chamber phase)",
                        reasoning_kind=reasoning_kind,
                        latency_ms=turn_latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read,
                        cache_creation_tokens=cache_creation,
                    )
                )
                final_message = response
                break

            if froze_now and enter_pass2:
                # Flip to phase 2: swap the offered tools to chamber-only.
                phase = "chamber"
                active_tools = state.pass2_tools
                by_name = {t.name: t for t in active_tools}
                tool_param_dicts = [t.to_dict() for t in active_tools]
                pass2_deadline = turn_idx + 1 + _PASS2_TURN_CAP
                turn_idx += 1
                continue

            # Phase-2 finalization: submit_chamber_outcome captured.
            if phase == "chamber" and state.get_outcome() is not None:
                trace_sink(
                    TraceStep(
                        run_id=run_id,
                        claim_id=claim.id,
                        engine=engine,
                        phase=phase,
                        step=step_idx,
                        turn_idx=turn_idx,
                        kind="final_output",
                        agent_reasoning="submit_chamber_outcome invoked",
                        reasoning_kind=reasoning_kind,
                        latency_ms=turn_latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read,
                        cache_creation_tokens=cache_creation,
                    )
                )
                final_message = response
                break

            # Phase-2 soft budget: the ungraded chamber prediction never
            # arrived. Finalize with the frozen extraction and an empty outcome
            # rather than spend the whole turn budget (this also absorbs the
            # qwen reasoning-mode tool-call drop on the second submit tool).
            if (
                phase == "chamber"
                and pass2_deadline is not None
                and turn_idx + 1 >= pass2_deadline
            ):
                # Success: the graded extraction is frozen. No error -- the
                # missing chamber prediction is ungraded (the marker lives in
                # agent_reasoning so the classifier does not see a failure).
                trace_sink(
                    TraceStep(
                        run_id=run_id,
                        claim_id=claim.id,
                        engine=engine,
                        phase=phase,
                        step=step_idx,
                        turn_idx=turn_idx,
                        kind="final_output",
                        agent_reasoning="phase-2 budget exhausted; finalized with frozen extraction",
                        reasoning_kind=reasoning_kind,
                        error="",
                        latency_ms=turn_latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read,
                        cache_creation_tokens=cache_creation,
                    )
                )
                final_message = response
                break

            turn_idx += 1

    finally:
        try:
            await client.close()
        except Exception:
            logger.debug("error closing Anthropic client", exc_info=True)
        if http_client is not None:
            try:
                await http_client.aclose()
            except Exception:
                logger.debug("error closing HTTP client", exc_info=True)
        try:
            pdf_cleanup()
        except Exception:
            logger.debug("error in pdf_cleanup", exc_info=True)
        try:
            ch_cleanup()
        except Exception:
            logger.debug("error in ch_cleanup", exc_info=True)

    extraction_payload = state.get_extraction()
    if extraction_payload is None:
        # Phase 1 never finalized: the loop ran to max_turns or the model ended
        # a turn without calling submit_extraction. Both are retryable
        # behavioural glitches; SubmitToolNotCalledError signals the retry
        # policy in the runner -- the pattern the private repository's
        # extract_lite.py used, carried over.
        stop = getattr(final_message, "stop_reason", "no_final_message")
        if final_message is None:
            raise SubmitToolNotCalledError(
                f"agent exhausted max_turns={max_turns} without calling submit_extraction (claim {claim.id})"
            )
        raise SubmitToolNotCalledError(
            f"agent ended (stop_reason={stop}) without calling submit_extraction (claim {claim.id})"
        )

    # Phase 1 froze but the loop exited via the max_turns guard without a
    # terminal break (phase 2 ran out of turns before submit_chamber_outcome or
    # the pass-2 budget check). Emit the missing terminal final_output -- with
    # zero usage, since each ran turn was already counted via its tool_call --
    # so the "exactly one terminal final_output per run" invariant that
    # rollup_cell_usage and the classifier rely on still holds.
    if final_message is None:
        trace_sink(
            TraceStep(
                run_id=run_id,
                claim_id=claim.id,
                engine=engine,
                phase=phase,
                step=step_idx,
                turn_idx=turn_idx,
                kind="final_output",
                agent_reasoning="reached max_turns in phase 2; finalized with frozen extraction",
                reasoning_kind="text",
                error="",
            )
        )

    # Merge the frozen pass-1 extraction with the optional, ungraded pass-2
    # chamber prediction. The merge reads extracted/found only from the frozen
    # capture, so phase 2 cannot have altered the fidelity result.
    payload = _merge_two_pass_payload(claim, extraction_payload, state.get_outcome())
    return ClaimResult.model_validate(payload)


# ---------------------------------------------------------------------------
# Provider routers (public entry points)
# ---------------------------------------------------------------------------


async def extract_chamber_baseline(
    claim: ClaimSpec,
    *,
    model: str = "claudesonnet4.6",
    max_tokens: int = 32768,
    trace_sink: TraceSink | None = None,
    reasoning_effort: str = "medium",
) -> ClaimResult:
    """Provider router for the non-agentic baseline.

    OpenAI models route to the Responses-API engine in ``openai_path.py``
    (which captures reasoning summaries); every other model uses the
    Anthropic-shape engine. ``reasoning_effort`` applies only to the OpenAI path -- it is ignored for Anthropic/vLLM.
    The OpenAI module is imported lazily here to avoid an import cycle.
    """
    if _is_openai_model(model):
        from chamberbench.harness.openai_path import (
            extract_chamber_baseline_openai,
        )

        return await extract_chamber_baseline_openai(
            claim,
            model=model,
            max_tokens=max_tokens,
            trace_sink=trace_sink,
            reasoning_effort=reasoning_effort,
        )
    return await _extract_chamber_baseline_anthropic(
        claim, model=model, max_tokens=max_tokens, trace_sink=trace_sink
    )


async def extract_chamber_agentic(
    claim: ClaimSpec,
    *,
    model: str = "claudesonnet4.6",
    max_turns: int = 30,
    trace_sink: TraceSink = _null_trace_sink,
    chamber_dataset: str | None = None,
    chamber_config: str = "standard",
    chamber_name: Literal["wt", "lt"] = "wt",
    max_tokens: int = 32768,
    inspect_page_detail: InspectDetail = "high",
    reasoning_effort: str = "medium",
    excluded_tools: frozenset[str] = frozenset(),
) -> ClaimResult:
    """Provider router for the agentic engine.

    OpenAI models route to the Responses-API engine in ``openai_path.py``
    (which captures reasoning summaries); every other model uses the
    Anthropic-shape engine. ``reasoning_effort`` is
    honoured by the OpenAI (Responses API) and Claude (the ``effort``
    knob) paths; it has no effect for vLLM models (qwen). The OpenAI
    module is imported lazily here to avoid an import cycle.

    `excluded_tools` (fault injection) is honoured on both the Anthropic
    and the OpenAI (Responses-API) paths; the empty default is a no-op.
    """
    if _is_openai_model(model):
        from chamberbench.harness.openai_path import (
            extract_chamber_agentic_openai,
        )

        return await extract_chamber_agentic_openai(
            claim,
            model=model,
            max_turns=max_turns,
            trace_sink=trace_sink,
            chamber_dataset=chamber_dataset,
            chamber_config=chamber_config,
            chamber_name=chamber_name,
            max_tokens=max_tokens,
            inspect_page_detail=inspect_page_detail,
            reasoning_effort=reasoning_effort,
            excluded_tools=excluded_tools,
        )
    return await _extract_chamber_agentic_anthropic(
        claim,
        model=model,
        max_turns=max_turns,
        trace_sink=trace_sink,
        chamber_dataset=chamber_dataset,
        chamber_config=chamber_config,
        chamber_name=chamber_name,
        max_tokens=max_tokens,
        inspect_page_detail=inspect_page_detail,
        excluded_tools=excluded_tools,
    )


# ---------------------------------------------------------------------------
# Trace utilities (used by the agentic path; safe to import early)
# ---------------------------------------------------------------------------


def jsonl_trace_sink(path: str) -> TraceSink:
    """Return a trace sink that appends one JSON line per TraceStep."""

    def _sink(step: TraceStep) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(step.model_dump_json() + "\n")

    return _sink


def new_run_id(model: str) -> str:
    """Generate a stable run identifier for trace correlation."""
    return f"{int(time.time())}-{model}-{uuid.uuid4().hex[:8]}"
