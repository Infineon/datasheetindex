"""Engagement diagnostic protocol -- fidelity-only, no chamber-side verdict.

Used for the qwen-engagement experiment (see docs/datasheetindex_chamber_benchmark.md
postmortem). Claims marked with `chamber_protocol: datasheet_agent.
chamber_eval.protocols.engagement_diagnostic` are checked on the *agent*
side only -- the chamber has no measurement that grounds them.

Returns a stub `ChamberMeasurement` so `verdict()` short-circuits to
`inconclusive` with a clear rationale. The point of these cells is not
the reproducibility verdict; it is the agent's per-tool dispatch
behaviour, which is recorded in `n_tool_calls_by_tool` and in the
trace JSONL by the test runner regardless of verdict.
"""

from __future__ import annotations

from chamberbench.claims import ChamberMeasurement, ClaimSpec
from chamberbench.protocols._common import make_stub_measurement


def run(claim: ClaimSpec) -> ChamberMeasurement:
    return make_stub_measurement(
        claim,
        reason=(
            "engagement-diagnostic claim -- the chamber has no measurement "
            "for this quantity. Fidelity-side instrumentation (extracted "
            "value vs. gold, plus per-tool dispatch counts) is what this "
            "cell exists to measure"
        ),
    )
