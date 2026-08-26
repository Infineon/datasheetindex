"""Chamber benchmark engine -- OpenAI Responses-API path.

The Anthropic-shape engine in ``anthropic_path.py`` reaches OpenAI models
through the LiteLLM ``/v1/messages`` passthrough, which routes them to the
Chat Completions API and drops all reasoning content. This module is the
fork that calls OpenAI models through the **Responses API**
(``/v1/responses`` on the same gateway), which returns reasoning summaries.

Two paths against the same ``ClaimSpec -> ClaimResult`` contract, mirroring
``anthropic_path.py``:

* ``extract_chamber_agentic_openai`` -- manual agentic loop, stateless
  multi-turn threading (``store=False`` + ``include=["reasoning.encrypted_content"]``,
  the full ``input`` list re-sent each turn).
* ``extract_chamber_baseline_openai`` -- single-pass PDF -> JSON.

Every provider-agnostic helper (tool factories, claim schema, sanitization,
prompts, trace emission) is imported from ``anthropic_path.py`` /
``datasheet_tools.py`` -- only the request/response shape is new. The public
routers in ``anthropic_path.py`` (``extract_chamber_baseline`` /
``extract_chamber_agentic``) dispatch here for OpenAI models.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import httpx2
import openai
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from chamberbench.claims import ClaimResult, ClaimSpec, Engine, TraceStep
from chamberbench.harness.anthropic_path import (
    _AGENTIC_SYSTEM,
    _BASELINE_SYSTEM,
    _PASS2_TURN_CAP,
    SubmitToolNotCalledError,
    TraceSink,
    _build_agentic_user_prompt,
    _build_baseline_user_prompt,
    _build_chamber_handoff_prompt,
    _build_claim_result_schema,
    _build_two_pass_state,
    _format_tool_error,
    _is_image_payload,
    _merge_two_pass_payload,
    _null_trace_sink,
    _resolve_pdf_to_local,
    _sanitize_claim_result_data,
    _summarize,
    new_run_id,
)
from chamberbench.harness.chamber_tools import _make_chamber_tools
from chamberbench.harness.datasheet_tools import (
    InspectDetail,
    _make_large_pdf_tools,
    _parse_json_from_text,
)

logger = logging.getLogger(__name__)

# Transient gateway errors that warrant a bounded retry. ValueError is added
# to the baseline's decorator (to cover degenerate "no text" responses) but
# kept out of the agentic per-turn retry, mirroring the Anthropic engine.
_OPENAI_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# Per-request timeout (seconds). The chamber test wraps each claim in a
# 360 s asyncio.wait_for; a single Responses turn stays well under this.
_REQUEST_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _create_openai_client() -> tuple[OpenAI, httpx2.Client | None]:
    """Create a sync OpenAI client pointed at the LiteLLM gateway's
    Responses API.

    Returns ``(client, http_client)``; ``http_client`` is the custom httpx2
    client when TLS verification is disabled (a gateway presenting a
    self-signed certificate),
    else None. Callers must close both.

    Built on ``httpx2``, not plain ``httpx``, for consistency with the
    Anthropic client sites (``datasheet_tools._create_client``,
    ``classifier``'s LLM-assist path): anthropic==1.0.0 rejects a plain
    ``httpx.Client``/``AsyncClient`` at construction, while openai==3.3.1
    accepts either -- so httpx2 is used everywhere rather than splitting
    the two SDKs across two client libraries.
    """
    from chamberbench.credentials import setup_credentials, tls_verify_disabled

    setup_credentials()

    api_key = os.environ.get("LITELLM_MASTER_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    if not api_key:
        raise ValueError(
            "No API key found. Set LITELLM_MASTER_KEY or ANTHROPIC_API_KEY."
        )
    base_url = os.environ.get("LITELLM_BASE_URL") or os.environ.get(
        "ANTHROPIC_BASE_URL"
    )
    if not base_url:
        raise ValueError(
            "No gateway base URL. Set LITELLM_BASE_URL or ANTHROPIC_BASE_URL."
        )
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    http_client: httpx2.Client | None = None
    # One predicate, shared with every other path. A hand-rolled copy here
    # accepted a narrower set of spellings than credentials.tls_verify_disabled
    # and still honoured NODE_TLS_REJECT_UNAUTHORIZED, which that helper
    # deliberately dropped -- so DISABLE_TLS_VERIFY=on skipped verification on
    # one path and not the other.
    if tls_verify_disabled():
        http_client = httpx2.Client(verify=False, timeout=_REQUEST_TIMEOUT_S)

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "timeout": _REQUEST_TIMEOUT_S,
    }
    if http_client is not None:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs), http_client


def _close_client(client: OpenAI, http_client: httpx2.Client | None) -> None:
    try:
        client.close()
    except Exception:
        logger.debug("error closing OpenAI client", exc_info=True)
    if http_client is not None:
        try:
            http_client.close()
        except Exception:
            logger.debug("error closing HTTP client", exc_info=True)


# ---------------------------------------------------------------------------
# Tool / payload translation
# ---------------------------------------------------------------------------


def _to_openai_tools(beta_tools: list[Any]) -> list[dict[str, Any]]:
    """Translate the chamber's ``beta_async_tool`` objects into Responses-API
    function-tool dicts.

    ``strict`` is False: the PDF/chamber tool schemas are not all strict-clean,
    and ``submit_claim_result`` is Pydantic-validated downstream regardless
    (gateways validate tool input inconsistently).
    """
    tools: list[dict[str, Any]] = []
    for t in beta_tools:
        d = t.to_dict()
        tools.append(
            {
                "type": "function",
                "name": d["name"],
                "description": d.get("description", ""),
                "parameters": d["input_schema"],
                "strict": False,
            }
        )
    return tools


def _build_pdf_input_file(pdf_path: str) -> dict[str, Any]:
    """Build a Responses-API ``input_file`` content part from a local PDF."""
    data = Path(pdf_path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return {
        "type": "input_file",
        "filename": Path(pdf_path).name,
        "file_data": f"data:application/pdf;base64,{b64}",
    }


def _function_call_output(call_id: str, output: Any) -> dict[str, Any]:
    """Build a ``function_call_output`` item for the next turn's input.

    Image payloads (from ``inspect_page``) are passed as ``input_image``
    content parts -- confirmed readable by gpt-5.1/gpt-5.2 on the gateway.
    Everything else (including tool errors) is a plain string -- the
    Responses API has no error flag on ``function_call_output``.
    """
    if _is_image_payload(output):
        parts: list[dict[str, Any]] = [
            {"type": "input_text", "text": "Inspected page image."}
        ]
        for block in output:
            source = block.get("source", {})
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            parts.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{data}",
                }
            )
        return {"type": "function_call_output", "call_id": call_id, "output": parts}
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output if isinstance(output, str) else str(output),
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_responses_output(
    response: Any,
) -> tuple[str, str, list[Any], list[dict[str, Any]]]:
    """Split a Responses-API result into
    ``(reasoning_summary, assistant_text, function_calls, raw_items)``.

    ``raw_items`` is every output item as a dict, re-sent verbatim on the
    next turn so reasoning items (and their ``encrypted_content``) carry
    forward -- the stateless-threading mechanism.
    """
    reasoning_parts: list[str] = []
    text_parts: list[str] = []
    function_calls: list[Any] = []
    output_items = list(getattr(response, "output", None) or [])
    for item in output_items:
        itype = getattr(item, "type", None)
        if itype == "reasoning":
            for summary in getattr(item, "summary", None) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_parts.append(text)
        elif itype == "message":
            for part in getattr(item, "content", None) or []:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
        elif itype == "function_call":
            function_calls.append(item)
    raw_items = [item.model_dump() for item in output_items]
    return (
        "\n".join(reasoning_parts).strip(),
        "\n".join(text_parts).strip(),
        function_calls,
        raw_items,
    )


def _usage_from_response(response: Any) -> dict[str, int]:
    """Map a Responses-API ``usage`` object to the TraceStep token fields.

    The gateway's Responses usage carries no cache fields, so cache_* are 0.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
    }


@retry(
    retry=retry_if_exception_type(_OPENAI_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    reraise=True,
)
def _responses_create_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    """Bounded-retry wrapper around the (sync) Responses API call.

    Per-turn retry only -- non-retryable failures bubble immediately so the
    agentic loop's ``input`` list stays a consistent prefix.
    """
    return client.responses.create(**kwargs)


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


async def extract_chamber_agentic_openai(
    claim: ClaimSpec,
    *,
    model: str,
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
    """Manual agentic loop over the OpenAI Responses API.

    Stateless multi-turn: every output item (reasoning items included) is
    re-sent verbatim each turn so the model keeps reasoning context across
    tool calls. Emits the same ``TraceStep`` shape as the Anthropic engine
    (one ``tool_call`` per dispatched call carrying per-turn usage, plus a
    terminal ``final_output``), so the test-runner's usage roll-up is
    unchanged.
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

    # Two-pass partitioning (shared with the Anthropic engine): phase 1 offers
    # datasheet tools + submit_extraction; chamber tools + submit_chamber_outcome
    # are withheld until the extracted value is frozen. Fault-injection
    # exclusions apply to both phases inside _build_two_pass_state.
    state = _build_two_pass_state(pdf_tools, ch_tools, excluded_tools)
    phase: Literal["extraction", "chamber"] = "extraction"
    active_tools = state.pass1_tools
    by_name = {t.name: t for t in active_tools}
    openai_tools = _to_openai_tools(active_tools)

    run_id = new_run_id(model)
    engine: Engine = "agentic"

    input_items: list[dict[str, Any]] = [
        {"role": "user", "content": _build_agentic_user_prompt(claim)},
    ]

    client, http_client = _create_openai_client()
    final_response: Any = None
    step_idx = 0
    turn_idx = 0
    pass2_deadline: int | None = None

    try:
        while turn_idx < max_turns:
            t0 = time.monotonic()
            response = await asyncio.to_thread(
                _responses_create_with_retry,
                client,
                model=model,
                input=input_items,
                instructions=_AGENTIC_SYSTEM,
                tools=openai_tools,
                reasoning={"effort": reasoning_effort, "summary": "auto"},
                include=["reasoning.encrypted_content"],
                store=False,
                max_output_tokens=max_tokens,
            )
            turn_latency_ms = int((time.monotonic() - t0) * 1000)

            # `_assistant_text` (GPT's visible message text) is intentionally
            # not traced -- agent_reasoning carries the reasoning summary,
            # the load-bearing field for failure attribution.
            reasoning_text, _assistant_text, function_calls, raw_items = (
                _parse_responses_output(response)
            )
            usage = _usage_from_response(response)

            # Carry every output item forward -- reasoning items + their
            # encrypted_content must persist for cross-tool continuity.
            input_items.extend(raw_items)

            # Terminal: response truncated at the output-token cap. In phase 1
            # this raises (mirroring the Anthropic max_tokens path); in phase 2
            # it sets final_response and breaks into the post-loop merge, since
            # the graded extraction is already frozen.
            if getattr(response, "status", None) == "incomplete":
                reason = getattr(
                    getattr(response, "incomplete_details", None), "reason", ""
                )
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
                        reasoning_kind="summary",
                        # phase 1 raises below; phase 2 succeeds with the frozen
                        # extraction, so its terminal marker carries no error.
                        error=(
                            ""
                            if phase == "chamber"
                            else f"stop_reason={reason or 'incomplete'}"
                        ),
                        latency_ms=turn_latency_ms,
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        reasoning_tokens=usage["reasoning_tokens"],
                    )
                )
                # Phase 1 (graded) fails loud; phase 2 keeps the frozen
                # extraction and finalizes with an empty (ungraded) prediction.
                if phase == "extraction":
                    raise ValueError(
                        f"agent stopped incomplete (reason={reason!r}; turn "
                        f"{turn_idx}); raise max_tokens or simplify the claim"
                    )
                final_response = response
                break

            # Terminal: the model ended a turn with no tool call. In phase 1 it
            # never called submit_extraction (post-loop raises
            # SubmitToolNotCalledError); in phase 2 the ungraded chamber
            # prediction is simply absent and we finalize with the frozen value
            # -- a success, so its terminal marker carries no error.
            if not function_calls:
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
                        reasoning_kind="summary",
                        error=("" if phase == "chamber" else "terminal_without_submit"),
                        latency_ms=turn_latency_ms,
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        reasoning_tokens=usage["reasoning_tokens"],
                    )
                )
                final_response = response
                break

            # Dispatch every function call. Per-turn usage is duplicated to
            # each tool_call event so the test-runner roll-up (which sums
            # one tool_call per turn_idx) matches the Anthropic engine.
            for fc in function_calls:
                # A tool outside the active phase set (chamber tools before the
                # freeze -- the structural enforcement of fidelity independence
                # -- or fault-injected exclusions) is not dispatched and leaves
                # no trace step: the dispatch record must show the tool was never
                # called and no tool output entered the context.
                if fc.name not in by_name:
                    input_items.append(
                        _function_call_output(fc.call_id, "ERROR: tool not available")
                    )
                    continue
                t1 = time.monotonic()
                err: str | None = None
                output: Any = None
                args: Any = None
                try:
                    args = json.loads(fc.arguments)
                except json.JSONDecodeError as exc:
                    err = _format_tool_error(exc)
                    output = err
                if err is None:
                    try:
                        output = await by_name[fc.name].call(args)
                    except Exception as exc:
                        err = _format_tool_error(exc)
                        output = err
                        logger.exception("tool %s failed", fc.name)
                input_items.append(_function_call_output(fc.call_id, output))
                trace_sink(
                    TraceStep(
                        run_id=run_id,
                        claim_id=claim.id,
                        engine=engine,
                        phase=phase,
                        step=step_idx,
                        turn_idx=turn_idx,
                        kind="tool_call",
                        tool_name=fc.name,
                        tool_input_summary=_summarize(
                            args if args is not None else fc.arguments
                        ),
                        tool_output_summary=_summarize(output),
                        error=err or "",
                        agent_reasoning=reasoning_text,
                        reasoning_kind="summary",
                        latency_ms=int((time.monotonic() - t1) * 1000),
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        reasoning_tokens=usage["reasoning_tokens"],
                    )
                )
                step_idx += 1

            # Phase-1 freeze: submit_extraction captured this turn. Its own
            # tool_call event is already recorded; here we either end the run
            # (found=false) or hand off to phase 2.
            froze_now = phase == "extraction" and state.get_extraction() is not None
            if froze_now:
                frozen = state.get_extraction() or {}
                if not frozen.get("found", False):
                    # found=false: no chamber phase. Terminal event.
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
                            reasoning_kind="summary",
                            latency_ms=turn_latency_ms,
                            input_tokens=usage["input_tokens"],
                            output_tokens=usage["output_tokens"],
                            reasoning_tokens=usage["reasoning_tokens"],
                        )
                    )
                    final_response = response
                    break
                # Flip to phase 2: swap the offered tools to chamber-only and
                # hand off (Responses-API input has no role-alternation rule, so
                # a fresh user item after the function_call_outputs is valid).
                phase = "chamber"
                active_tools = state.pass2_tools
                by_name = {t.name: t for t in active_tools}
                openai_tools = _to_openai_tools(active_tools)
                input_items.append(
                    {
                        "role": "user",
                        "content": _build_chamber_handoff_prompt(claim, frozen),
                    }
                )
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
                        reasoning_kind="summary",
                        latency_ms=turn_latency_ms,
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        reasoning_tokens=usage["reasoning_tokens"],
                    )
                )
                final_response = response
                break

            # Phase-2 soft budget: the ungraded chamber prediction never
            # arrived. Finalize with the frozen extraction and an empty outcome
            # -- a success (no error; the marker lives in agent_reasoning).
            if (
                phase == "chamber"
                and pass2_deadline is not None
                and turn_idx + 1 >= pass2_deadline
            ):
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
                        reasoning_kind="summary",
                        error="",
                        latency_ms=turn_latency_ms,
                        input_tokens=usage["input_tokens"],
                        output_tokens=usage["output_tokens"],
                        reasoning_tokens=usage["reasoning_tokens"],
                    )
                )
                final_response = response
                break

            turn_idx += 1
    finally:
        _close_client(client, http_client)
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
        if final_response is None:
            raise SubmitToolNotCalledError(
                f"agent exhausted max_turns={max_turns} without calling submit_extraction (claim {claim.id})"
            )
        status = getattr(final_response, "status", "?")
        raise SubmitToolNotCalledError(
            f"agent ended (status={status}) without calling submit_extraction (claim {claim.id})"
        )

    # Phase 1 froze but the loop exited via the max_turns guard without a
    # terminal break (phase 2 ran out of turns). Emit the missing terminal
    # final_output with zero usage (each ran turn was already counted via its
    # tool_call) so the single-terminal-final_output invariant holds.
    if final_response is None:
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
                reasoning_kind="summary",
                error="",
            )
        )

    # Merge the frozen pass-1 extraction with the optional, ungraded pass-2
    # chamber prediction. extracted/found come only from the frozen capture.
    payload = _merge_two_pass_payload(claim, extraction_payload, state.get_outcome())
    return ClaimResult.model_validate(payload)


# ---------------------------------------------------------------------------
# Non-agentic baseline
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type(_OPENAI_RETRYABLE + (ValueError,)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    reraise=True,
)
async def extract_chamber_baseline_openai(
    claim: ClaimSpec,
    *,
    model: str,
    max_tokens: int = 32768,
    trace_sink: TraceSink | None = None,
    reasoning_effort: str = "medium",
) -> ClaimResult:
    """Single-pass non-agentic baseline over the OpenAI Responses API.

    The PDF is sent as an ``input_file`` content part; the structured
    ``ClaimResult`` is delivered via the Responses ``text.format``
    json-schema channel. Emits one ``final_output`` TraceStep.
    """
    pdf_path = _resolve_pdf_to_local(claim.pdf_source)
    output_schema = _build_claim_result_schema()
    run_id = new_run_id(model)

    input_items: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                _build_pdf_input_file(pdf_path),
                {"type": "input_text", "text": _build_baseline_user_prompt(claim)},
            ],
        }
    ]

    client, http_client = _create_openai_client()
    t0 = time.monotonic()
    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model=model,
            input=input_items,
            instructions=_BASELINE_SYSTEM,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            max_output_tokens=max_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ClaimResult",
                    "schema": output_schema,
                    "strict": True,
                }
            },
        )
    finally:
        _close_client(client, http_client)
    latency_ms = int((time.monotonic() - t0) * 1000)

    reasoning_text, assistant_text, _fcs, _raw = _parse_responses_output(response)
    text = getattr(response, "output_text", None) or assistant_text
    if not text:
        status = getattr(response, "status", "?")
        raise ValueError(f"Baseline returned no text (status={status})")

    data = _parse_json_from_text(text)
    data.setdefault("claim_id", claim.id)
    _sanitize_claim_result_data(data)
    claim_result = ClaimResult.model_validate(data)

    if trace_sink is not None:
        usage = _usage_from_response(response)
        trace_sink(
            TraceStep(
                run_id=run_id,
                claim_id=claim.id,
                engine="baseline",
                step=0,
                turn_idx=0,
                kind="final_output",
                agent_reasoning=reasoning_text,
                reasoning_kind="summary",
                latency_ms=latency_ms,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
            )
        )

    return claim_result
