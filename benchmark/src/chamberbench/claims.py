"""Pydantic models for the chamber-grounded benchmark.

These wrap (rather than replace) the production extraction models in models.py.
A ClaimResult.extracted is a ParameterResult, so the fidelity helpers in
`chamberbench.grading` work unchanged on chamber results.

NOT_SPECIFIED sentinel pattern reused from models.py because the same SDK
schema-validation constraint applies (no anyOf/null in structured output).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer

from chamberbench.grading import DEFAULT_CONFIDENCE_FLOOR
from chamberbench.models import NOT_SPECIFIED, ParameterResult

ClaimKind = Literal["dc_accuracy", "range", "typical", "max", "min", "linearity"]
ToleranceKind = Literal["absolute", "relative", "spec_derived"]
Verdict = Literal["pass", "fail", "inconclusive"]
TraceKind = Literal["tool_call", "reasoning_only", "final_output"]
ReasoningKind = Literal["text", "thinking", "summary"]
Engine = Literal["agentic", "baseline"]
Attribution = Literal[
    "tool_output",
    "tool_selection",
    "condition_omission",
    "verification_skipped",
    # `engine_error` covers retry-exhausted gateway failures, timeouts, and
    # any model-side crashes that aren't agent reasoning errors. None of the
    # four rubric categories applies, so this is a fifth slot for honesty.
    "engine_error",
    "ok",
    "unclassified",
]


class OperatingCondition(BaseModel):
    """One stated condition required for a claim to apply."""

    model_config = {"extra": "forbid"}

    name: str
    value: float = Field(default=NOT_SPECIFIED)
    min_value: float = Field(default=NOT_SPECIFIED)
    max_value: float = Field(default=NOT_SPECIFIED)
    unit: str = ""
    # Empty string => the chamber does not directly control or expose this
    # variable; the verdict logic flags it as unmatched.
    chamber_variable: str = ""
    # Load-bearing conditions force "inconclusive" when unmatched. Non-load-bearing
    # ones are reported but do not force the verdict.
    load_bearing: bool = True

    def has_value(self) -> bool:
        return self.value != NOT_SPECIFIED

    def has_min(self) -> bool:
        return self.min_value != NOT_SPECIFIED

    def has_max(self) -> bool:
        return self.max_value != NOT_SPECIFIED

    @model_serializer
    def _serialize(self) -> dict:
        data: dict = {"name": self.name}
        if self.has_value():
            data["value"] = self.value
        if self.has_min():
            data["min_value"] = self.min_value
        if self.has_max():
            data["max_value"] = self.max_value
        if self.unit:
            data["unit"] = self.unit
        if self.chamber_variable:
            data["chamber_variable"] = self.chamber_variable
        data["load_bearing"] = self.load_bearing
        return data


# Allowlist of ClaimSpec fields safe to expose to the extraction agent.
# Everything else -- claimed_*, tolerance_*, source_page, source_text,
# value_contains, confidence_min, realizable_subset, verified_*, plus the
# raw pdf_source URL (the PDF itself is attached/loaded separately) -- is
# oracle ground truth for the fidelity scorer or downstream scorer/protocol
# config. Embedding it in the prompt turns the benchmark into an open-book
# exam (the agent sees the page number, verbatim text, and answer).
#
# This is an ALLOWLIST rather than a blocklist so that any future ClaimSpec
# field is opt-in visible -- a new oracle-bearing field can't silently leak
# through by being added without a matching exclusion.
#
# Note: `description` is intentionally NOT here. Curators in claims.yaml
# write the paraphrased answer (and sometimes the source row verbatim) into
# the description field, so exposing it to the agent is a second oracle
# leak channel. The agent identifies each claim from `parameter +
# expected_unit + operating_conditions`, which is sufficient. Sufficiency
# is pinned by tests/test_chamber_prompts.py::TestClaimsYamlHygiene::
# test_claim_kinds_are_disambiguable_from_visible_projection; the
# description leak backlog is tracked by
# test_descriptions_do_not_leak_numeric_answers and
# test_descriptions_do_not_leak_source_page_references.
AGENT_VISIBLE_SPEC_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "parameter",
        "expected_unit",
        "claim_kind",
        "operating_conditions",
        # Chamber-side fields the agent legitimately needs to populate
        # expected_chamber_outcome and pick the right experiment.
        #
        # `chamber_experiment_hint` is intentionally NOT here: it is read
        # server-side by chamberbench/protocols/* (e.g. light_sensor.py
        # references claim.chamber_experiment_hint directly) and is a
        # future-leak surface if a curator names a hint after the answer
        # ("validate_460ns_rise"). The agent must discover the right
        # experiment via list_experiments instead.
        "chamber_dataset",
        "chamber_protocol",
        "primary_chamber_variable",
        "cross_check_variables",
    }
)


class ClaimSpec(BaseModel):
    """A single claim to be verified. Loaded from data/claims.yaml."""

    model_config = {"extra": "forbid"}

    id: str
    pdf_source: str
    parameter: str
    description: str = ""
    expected_unit: str
    claim_kind: ClaimKind
    claimed_min: float = Field(default=NOT_SPECIFIED)
    claimed_max: float = Field(default=NOT_SPECIFIED)
    claimed_typical: float = Field(default=NOT_SPECIFIED)
    tolerance_value: float = Field(default=NOT_SPECIFIED)
    tolerance_kind: ToleranceKind = "spec_derived"
    operating_conditions: list[OperatingCondition] = Field(default_factory=list)
    # Source-side ground truth for the fidelity scorer.
    source_page: list[int] = Field(default_factory=list)
    source_text: str = ""
    value_contains: list[str] = Field(default_factory=list)
    confidence_min: float = DEFAULT_CONFIDENCE_FLOOR
    # Reproducibility-side hooks.
    chamber_dataset: str = "wt_validate_v1"
    chamber_experiment_hint: str = ""
    chamber_protocol: str
    primary_chamber_variable: str
    cross_check_variables: list[str] = Field(default_factory=list)
    # List of (lo, hi) windows of the claimed range that the chamber can exercise.
    realizable_subset: list[list[float]] = Field(default_factory=list)
    # Audit.
    verified_by: str = "manual"
    verified_date: str = ""

    def has_typical(self) -> bool:
        return self.claimed_typical != NOT_SPECIFIED

    def has_min(self) -> bool:
        return self.claimed_min != NOT_SPECIFIED

    def has_max(self) -> bool:
        return self.claimed_max != NOT_SPECIFIED

    def has_explicit_tolerance(self) -> bool:
        return self.tolerance_value != NOT_SPECIFIED

    def agent_visible_dump(self) -> dict[str, Any]:
        """Projection of this spec safe to embed in the extraction prompt.

        Called from `harness.anthropic_path._render_claim_spec_for_prompt`, which
        is the only chokepoint that serialises a `ClaimSpec` into prompt
        text. Do NOT call `model_dump_json` directly on a `ClaimSpec` for
        any agent-facing surface.

        Strips ground-truth oracle fields (page numbers, verbatim source
        text, claimed value, tolerance, required substrings) and scorer/
        audit config the agent has no business seeing. See
        AGENT_VISIBLE_SPEC_FIELDS for the allowlist rationale.
        """
        return self.model_dump(include=set(AGENT_VISIBLE_SPEC_FIELDS))


class ClaimResult(BaseModel):
    """Strict structured-output target for the agent on the fidelity side."""

    model_config = {"extra": "forbid"}

    claim_id: str
    found: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    extracted: ParameterResult
    extracted_conditions: list[OperatingCondition] = Field(default_factory=list)
    # Agent's own hypothesis for what the chamber should measure under the
    # claim's stated conditions. Reported but not gated; useful as a sanity
    # signal and for confidence calibration analysis.
    expected_chamber_outcome: str = ""


class ChamberMeasurement(BaseModel):
    """Reproducibility-side payload, populated by chamberbench/protocols/*."""

    model_config = {"extra": "forbid"}

    claim_id: str
    experiment_ids: list[str] = Field(default_factory=list)
    measured_value: float
    measured_unit: str
    measured_sigma: float
    # How `measured_sigma` was estimated. "cross_sensor" means the cross-sensor
    # pairwise spread; honest about the correlated-noise lower-bound limitation.
    # "stub" means the protocol short-circuited because a load-bearing
    # condition was unmatched -- measured_value/measured_sigma are NaN by
    # design, and quality_gates.H5 keys on this marker to know that NaN is
    # expected (not a protocol bug). Adding a new sigma_basis is forward-
    # compatible because Pydantic validates against the union, not order.
    measured_sigma_basis: Literal[
        "cross_sensor", "single_channel_std", "literature", "stub"
    ] = "cross_sensor"
    matched_conditions: list[str] = Field(default_factory=list)
    unmatched_conditions: list[str] = Field(default_factory=list)
    sample_n: int = 0
    cross_sensor_spread: float = 0.0
    notes: str = ""


class ReproducibilityVerdict(BaseModel):
    model_config = {"extra": "forbid"}

    claim_id: str
    verdict: Verdict
    rationale: str
    # Use None (not the sentinel) for fields that don't apply to a given
    # verdict path -- e.g. range claims have no central-value delta. Optional
    # is acceptable here because this model is consumed by JSON tooling, not
    # by the Anthropic structured-output validator.
    delta: float | None = None
    spec_tolerance: float | None = None
    combined_uncertainty: float | None = None
    matched_subset: list[list[float]] = Field(default_factory=list)

    @model_serializer
    def _serialize(self) -> dict:
        data: dict = {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "rationale": self.rationale,
        }
        if self.delta is not None:
            data["delta"] = self.delta
        if self.spec_tolerance is not None:
            data["spec_tolerance"] = self.spec_tolerance
        if self.combined_uncertainty is not None:
            data["combined_uncertainty"] = self.combined_uncertainty
        if self.matched_subset:
            data["matched_subset"] = self.matched_subset
        return data


class TraceStep(BaseModel):
    """One step in the agentic loop, written one-per-line as JSONL.

    Forward-compatibility: `schema_version` lets older consumers
    (e.g. a pinned classifier) read newer traces without breaking, and
    `extra="ignore"` lets newer consumers tolerate unknown fields from
    older binaries. Bump `schema_version` whenever a field's *meaning*
    changes; additions alone do not require a bump.
    """

    model_config = {"extra": "ignore"}

    # Bumped 1 -> 2 when the two-pass freeze landed: a single claim run now
    # has two structurally-distinct phases, and `phase` carries that meaning.
    schema_version: int = 2
    run_id: str
    claim_id: str
    engine: Engine
    # Which pass of the two-pass agentic loop emitted this step. "extraction"
    # is the datasheet-only phase (the extracted value is frozen at
    # submit_extraction); "chamber" is the post-freeze chamber-prediction
    # phase. Baseline traces and pre-two-pass (schema v1) traces default to
    # "extraction", which is correct: they had no separate chamber phase.
    phase: Literal["extraction", "chamber"] = "extraction"
    # Monotonic step counter across the whole claim run; one row per dispatched
    # tool_call plus a final_output row (and optionally reasoning_only rows).
    step: int
    # Monotonic turn counter (one per round-trip with the model). Multiple
    # tool calls in one turn share the same turn_idx; the rubric classifier
    # uses this to tie all calls in a turn to a single reasoning block.
    turn_idx: int = 0
    kind: TraceKind
    tool_name: str = ""
    tool_input_summary: str = ""
    tool_output_summary: str = ""
    error: str = ""
    agent_reasoning: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # Provenance of `agent_reasoning`: "text" = visible assistant preamble
    # (any model, when no thinking block is present), "thinking" =
    # extended-thinking block (Claude adaptive thinking, or qwen via the
    # enable_thinking chat-template kwarg), "summary" = GPT Responses-API
    # reasoning summary.
    reasoning_kind: ReasoningKind = "text"
    # Reasoning-token count from the Responses API usage (GPT); 0 otherwise.
    reasoning_tokens: int = 0
    # Filled post-hoc by classifier.py; defaults to "unclassified" at write time.
    attribution: Attribution = "unclassified"
    attribution_note: str = ""
