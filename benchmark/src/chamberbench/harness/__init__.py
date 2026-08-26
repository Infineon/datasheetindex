"""Shared chamber-run infrastructure.

Importable by the runner (`chamberbench.harness.run`, the `chamber-run` entry
point) and by the standalone experiment scripts under `scripts/` -- `variance.py`,
`fault_injection.py`, `null_tool.py` and their siblings. Holds the pieces those
callers must agree on byte-for-byte: the per-cell token roll-up, the per-model
run configuration, and gateway credential setup.

In the private repository this was extracted from, the same module also backed
a pytest-driven harness (`tests/eval/conftest_chamber.py`,
`tests/eval/test_chamber.py`); `run.py` merges those two into the program that
ships here. Comments naming them describe that origin, not a path in this
repository.
"""

from __future__ import annotations

from collections.abc import Iterable

from chamberbench.claims import TraceStep


def rollup_cell_usage(steps: Iterable[TraceStep]) -> dict[str, int]:
    """Sum per-cell token usage across an agent loop's trace steps.

    Agentic events carry usage per turn, duplicated onto each tool_call
    within the turn -- summing every tool_call would double-count. Take
    the per-turn total from the final_output event and one representative
    tool_call per turn_idx. Extracted verbatim from the inline roll-up in the
    private repository's `tests/eval/test_chamber.py`, so that the runner and
    the experiment scripts compute identical numbers.
    """
    seen_turns: set[int] = set()
    usage_in = 0
    usage_out = 0
    usage_cache_r = 0
    usage_cache_w = 0
    for s in steps:
        if s.kind == "final_output":
            usage_in += s.input_tokens or 0
            usage_out += s.output_tokens or 0
            usage_cache_r += s.cache_read_tokens or 0
            usage_cache_w += s.cache_creation_tokens or 0
        elif s.kind == "tool_call" and s.turn_idx not in seen_turns:
            seen_turns.add(s.turn_idx)
            usage_in += s.input_tokens or 0
            usage_out += s.output_tokens or 0
            usage_cache_r += s.cache_read_tokens or 0
            usage_cache_w += s.cache_creation_tokens or 0
    return {
        "input_tokens": usage_in,
        "output_tokens": usage_out,
        "cache_read_tokens": usage_cache_r,
        "cache_creation_tokens": usage_cache_w,
    }


# ---------------------------------------------------------------------------
# Per-model run configuration (pinned for reproducibility)
# ---------------------------------------------------------------------------
# Per-model defaults. All three models run a uniform 30-turn ceiling.
# GPT-5.1 was previously capped at 15 turns -- a legacy setting from the
# old LiteLLM gateway-passthrough routing. After GPT-5.1 moved to the
# native Responses API, measurement (2026-05-22) showed the chamber loop
# accumulates only ~800-1100 tokens/turn for GPT-5.1 (peak ~12-14k at
# turn 13, ~30k projected at turn 30 -- far under its context window):
# the chamber agent leans on the cheap text tools and rarely the image
# `inspect_page`. The 15-turn cap was therefore a stale artifact, not a
# real constraint, and it confounded the cross-model comparison (it
# truncated GPT-5.1 runs that needed >15 turns). Raised to 30 so the
# turn budget is uniform.
#
# `inspect_page_detail` (datasheetindex 0.12.0+):
# vision-token cost tier for the agent's `inspect_page` tool, closure-
# captured at session start. The agent does not see this knob -- it
# can't observe its own context budget, so the system picks based on
# the gateway's published `max_input_tokens`:
#
#   - "high" (~2580 tokens / page) when the model has >=200K input
#     context (Sonnet 4.6 at 1M, gpt-5.1 at 272K).
#   - "low" (~650 tokens / page) when the model is on a small-context
#     deployment (qwen3.5-27b at 32K -- a 9-turn loop with "high"
#     overflows the cap; with "low" it stays under 30 %).
#
# When this paper's hybrid-output_format engineering follow-up lands,
# the qwen detail tier can be re-evaluated; for now "low" is the
# pragmatic budget-fit for the qwen small-context wedge described in
# docs/reproducing.md.
# `reasoning_effort`: the cross-provider reasoning-depth knob. GPT rows
# pass it to the Responses API; the claudesonnet4.6 row passes it to
# Claude's adaptive-thinking `effort` knob (low/medium/high). qwen omits
# it -- vLLM via the Anthropic passthrough enables reasoning with the
# `enable_thinking` chat-template kwarg, which has no depth setting.
# Env override: CHAMBER_REASONING_EFFORT.
CHAMBER_MODEL_CONFIG: dict[str, dict[str, object]] = {
    "claudesonnet4.6": {
        "max_turns": 30,
        "inspect_page_detail": "high",
        "reasoning_effort": "medium",
    },
    "gpt-5.1": {
        "max_turns": 30,
        "inspect_page_detail": "high",
        "reasoning_effort": "medium",
    },
    "gpt-5": {
        "max_turns": 30,
        "inspect_page_detail": "high",
        "reasoning_effort": "medium",
    },
    "qwen3.5-27b": {"max_turns": 30, "inspect_page_detail": "low"},
    # qwen3.5-27b is retired on the gateway the published runs used;
    # qwen3.6-27b replaces it. Same config -- vLLM via the Anthropic passthrough enables
    # reasoning with the `enable_thinking` chat-template kwarg, not the
    # effort knob, and the agentic loop's accumulated inspect_page
    # history still wants the "low" tier.
    "qwen3.6-27b": {"max_turns": 30, "inspect_page_detail": "low"},
}
# Default Sonnet ceilings if the model isn't in CHAMBER_MODEL_CONFIG --
# safer than guessing low and tripping spurious max_turns exhaustion.
_DEFAULT_MAX_TURNS = 30
_DEFAULT_INSPECT_PAGE_DETAIL = "high"

# Valid Responses-API reasoning-effort levels (none/minimal/low/medium/high).
# `run._resolve_reasoning_effort` validates the CHAMBER_REASONING_EFFORT env
# override against this set, so a typo fails loudly rather than as an opaque
# gateway 400; the variance harness uses the per-model defaults above.
_VALID_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high"})


def model_config(model: str) -> dict[str, object]:
    """Run configuration for a model alias.

    An unrecognised alias gets the Sonnet-shaped ceiling rather than a
    guess: too low a ``max_turns`` produces spurious exhaustion that looks
    like a model failure.
    """
    cfg = dict(CHAMBER_MODEL_CONFIG.get(model, {}))
    cfg.setdefault("max_turns", _DEFAULT_MAX_TURNS)
    cfg.setdefault("inspect_page_detail", _DEFAULT_INSPECT_PAGE_DETAIL)
    return cfg


# ---------------------------------------------------------------------------
# Gateway credentials
# ---------------------------------------------------------------------------


def setup_gateway_credentials() -> None:
    """Load gateway credentials. Delegates to the shared Tier 1 helper."""
    from chamberbench.credentials import setup_credentials

    setup_credentials()
