"""Silent-failure detector for the chamber benchmark.

A "silent failure" is an agentic cell that passes fidelity -- the extracted
value matches the datasheet -- but reached that answer through a defective
process that the per-tool dispatch record exposes. Fidelity-only scoring is
structurally blind to these: it grades the value, not the process. This
detector reads the dispatch record and flags them.

Two pre-registered rules, fixed in advance, no tuning:

  R1 tool_bypass          -- a fidelity-pass cell with zero datasheet-navigation
                             calls. The agent never opened the document; the
                             answer was recalled, not extracted. This is the
                             Section-5 silent tool-bypass.
  R2 verification_skipped -- a fidelity-pass cell that navigated the document
                             but called no datasheet-verification tool
                             (search_text / extract_table_markdown), so the
                             extracted value was never cross-checked.

Both rules count datasheet tools only. Under the two-pass loop the chamber
tools run exclusively in the post-freeze chamber phase, so a fidelity answer
reached with only chamber calls is still a tool-bypass; counting them as
"navigation" would mask R1.

The two rules partition the failure space: R1 is "zero navigation"; R2 is
"some navigation but no verification". A clean cell trips neither.

Consumers:
  * scripts/fault_injection_experiment.py -- recall on planted failures and
    the false-positive rate on the real post-audit run.
  * chamber_eval/quality_gates.py soft gate S9 -- a regression guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Datasheet-navigation tools: exercising one of these is the agent reading the
# document. Chamber tools and the finalization tools are deliberately excluded
# -- under the two-pass loop the chamber tools run only in the post-freeze
# chamber phase, so a fidelity answer reached with only chamber (or submit)
# calls is still a tool-bypass.
_DATASHEET_NAV_TOOLS = frozenset(
    {
        "build_datasheet",
        "get_section_text",
        "search_text",
        "extract_table_markdown",
        "inspect_page",
    }
)

# Datasheet-side cross-check tools: using one is the verification the system
# prompt asks for. cross_sensor_check is a chamber-phase tool, not a datasheet
# cross-check, so it does not count here.
_DATASHEET_VERIFICATION_TOOLS = frozenset(
    {
        "search_text",
        "extract_table_markdown",
    }
)

__all__ = ["SilentFailureReport", "detect_silent_failure"]


@dataclass
class SilentFailureReport:
    """Outcome of running the detector on one cell.

    `flagged` is True when at least one rule fired; `rules` lists the rule
    names that fired ("tool_bypass", "verification_skipped"). A cell that is
    not a fidelity pass, or that hit an engine error, is never flagged -- a
    silent failure is by definition one that fidelity-only scoring blessed,
    and an engine error is a loud failure, not a silent one.
    """

    flagged: bool = False
    rules: list[str] = field(default_factory=list)


def _navigation_calls(n_tool_calls_by_tool: dict[str, Any]) -> int:
    """Total datasheet-navigation tool calls (chamber / submit tools excluded)."""
    return sum(
        int(count)
        for name, count in n_tool_calls_by_tool.items()
        if name in _DATASHEET_NAV_TOOLS
    )


def detect_silent_failure(cell: dict[str, Any]) -> SilentFailureReport:
    """Flag a fidelity-passing agentic cell whose dispatch record is defective.

    `cell` is one per-cell record from `baseline_chamber.json`,
    `latest_chamber.json`, or `fault_injection.json`: a dict carrying a
    `fidelity` sub-dict (with `overall_pass`) and `n_tool_calls_by_tool`.
    Only fidelity-passing, non-engine-error cells are considered.
    """
    report = SilentFailureReport()

    fidelity = cell.get("fidelity") or {}
    if not fidelity.get("overall_pass"):
        return report
    if fidelity.get("engine_error") or cell.get("engine_error"):
        return report

    by_tool = cell.get("n_tool_calls_by_tool") or {}
    nav = _navigation_calls(by_tool)

    if nav == 0:
        # R1: the agent never exercised a datasheet-navigation tool.
        report.rules.append("tool_bypass")
    elif not any(by_tool.get(name, 0) for name in _DATASHEET_VERIFICATION_TOOLS):
        # R2: navigated, but never cross-checked the datasheet.
        report.rules.append("verification_skipped")

    report.flagged = bool(report.rules)
    return report
