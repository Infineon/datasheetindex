"""Per-arm token cost of one full reproduction, from the archive.

Offline, no credentials, no network. A reader deciding whether to spend
money on a reproduction should have the number before they start, not
after -- see docs/regenerating.md's "What a reproduction costs" section.

Deliberately reports tokens, not dollars: provider pricing varies and goes
stale the day after publication. A reader applies their own rates.

Each `archive/latest_chamber.<arm>.json` holds 50 result entries: 25 claims
x 2 engines (agentic + baseline), keyed "<claim_id>|<engine>". The totals
below sum BOTH engines -- a reader reproducing "the full experiment" runs
both, and this script says so in its own output rather than leaving an
unlabelled number to be misread later. Each entry's "usage" field is
already the per-cell roll-up the live harness computed at run time
(`chamberbench.harness.rollup_cell_usage` applied to that cell's trace
steps) -- this script only sums it across cells, it does not re-derive it
from raw trace steps.

Usage:
    uv run python scripts/reproduction_cost.py
"""

from __future__ import annotations

from chamberbench.claimsio import load_archive

ARMS = ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b")

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens")

COVERAGE_NOTE = "both engines (agentic + baseline), 25 claims each = 50 cells per arm"


def summarise() -> dict[str, dict[str, int]]:
    """Per-arm token totals across every cell in that arm's archived run.

    Returns a mapping of arm name to a dict with "input_tokens",
    "output_tokens", and "cache_read_tokens" totals. An arm whose archive
    file is not present (e.g. a filename/alias variant not shipped) is
    silently omitted rather than raising, so this stays usable against a
    partial or CHAMBERBENCH_ARCHIVE_DIR-overridden archive.
    """
    out: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        try:
            payload = load_archive("latest_chamber." + arm + ".json")
        except FileNotFoundError:
            continue
        totals = dict.fromkeys(USAGE_KEYS, 0)
        # `results` is a dict keyed "<claim_id>|<engine>", not a list --
        # iterate .values(). Both engines' cells are included; see
        # COVERAGE_NOTE above.
        for entry in payload["results"].values():
            usage = entry.get("usage") or {}
            for key in USAGE_KEYS:
                totals[key] += int(usage.get(key) or 0)
        out[arm] = totals
    return out


def main() -> int:
    print("Reproduction cost per arm -- " + COVERAGE_NOTE)
    print()
    for arm, usage in summarise().items():
        print(
            "{:<20s} in={:9d} out={:8d} cache_read={:9d}".format(
                arm,
                usage["input_tokens"],
                usage["output_tokens"],
                usage["cache_read_tokens"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
