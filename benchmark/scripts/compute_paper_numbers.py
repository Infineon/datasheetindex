"""Every rebuttal number that was previously computed by hand.

Written after the third adversarial review round, whose findings had a consistent
direction: ten of thirteen corrections ran in the authors' favour. That is not
noise -- noise has no sign. It came from a specific step: computing a number in
an ad-hoc shell one-liner, then typing a sentence about it. The sentence is where
the bias lives ("the record truncated it" rather than "the matcher caught a real
miss"; a mean printed to two decimals read back as "exactly 1.00 in all 207"; a
comparison of two quantities that were never the same quantity).

So this script computes those numbers and prints them next to the comparison they
are used in. The rebuttal's number-verification pass then refused to let a numeral
into the response unless it appeared in this output or was explicitly declared as
not-derived. That removes the typing step, which is the only fix with the right
shape -- another review round lowers the error rate but cannot change its sign.

No LLM calls: pure analysis of committed artifacts plus (where available) `git log`.

Run:
    uv run python scripts/compute_paper_numbers.py
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from chamberbench.claimsio import archive_dir, data_dir
from chamberbench.silent_failure import _DATASHEET_NAV_TOOLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = archive_dir()
PILOT = archive_dir() / "pilot_audit" / "pilot_dispatch_audit.json"
CLAIMS = data_dir() / "claims.yaml"
CLAIMS_A4988 = data_dir() / "claims_a4988.yaml"
PAPER_TEX = PROJECT_ROOT / "paper" / "acl_latex.tex"

MODELS = ("claudesonnet4.6", "gpt-5.1")
# Two anchors, because only one of them is the operative rule. The first commit
# authored the detector; the second revised the predicate (removing the chamber's
# cross-sensor check from the cross-check set), and it is the revised rule the
# paper reports. Quoting only the earlier date makes the "never tuned on" claim
# look stronger than the operative rule supports. Both are commits in the private
# development repository that produced the archive, not in this port's history --
# see `report_chronology` for how that is handled.
DETECTOR_RULES_COMMIT = "1f12fe9"
DETECTOR_PREDICATE_REVISION = "cd89e98"

# A "decline" is a fidelity failure whose reason begins with this: the model
# reported the parameter as not found. Anything else means it committed to a
# value, which is the only case that carries information about corrupt success.
_DECLINE_PREFIX = "Found mismatch"

_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _answered(stem: str, model: str) -> tuple[int, int]:
    """(answered, total) for one probe arm."""
    doc = _load(RESULTS / f"{stem}.{model}.json")
    cells = doc.get("cells") or []
    answered = sum(
        1
        for c in cells
        if not str(c.get("fidelity_failure_reason") or "").startswith(_DECLINE_PREFIX)
    )
    return answered, len(cells)


def _one_sided_upper(successes: int, trials: int) -> float:
    """Exact one-sided 95% upper bound for a zero-success binomial."""
    if successes or not trials:
        raise ValueError("this bound is only valid for 0 successes in n>0 trials")
    return 100.0 * (1 - 0.05 ** (1 / trials))


def report_answer_rates() -> None:
    print("=" * 78)
    print("ANSWER RATES PER MODEL (the 'collapse' claim, split rather than pooled)")
    print("=" * 78)
    print("A pooled before/after hid that only one model changed. Print both.")
    print()
    print(f"{'model':<20}{'closed-book':>14}{'degraded':>12}{'null':>7}{'wrong':>7}")
    pooled_answered = pooled_total = 0
    for model in MODELS:
        cb_a, cb_n = _answered("closed_book", model)
        nu_a, nu_n = _answered("null_tool_injection", model)
        wc_a, wc_n = _answered("wrong_content", model)
        deg_a, deg_n = nu_a + wc_a, nu_n + wc_n
        pooled_answered += deg_a
        pooled_total += deg_n
        print(
            f"{model:<20}{f'{cb_a}/{cb_n} = {100 * cb_a / cb_n:.0f}%':>14}"
            f"{f'{deg_a}/{deg_n} = {100 * deg_a / deg_n:.0f}%':>12}{nu_a:>7}{wc_a:>7}"
        )
    print(f"{'POOLED degraded':<20}{'':>14}{f'{pooled_answered}/{pooled_total}':>12}")
    print()
    cb_a, cb_n = _answered("closed_book", "claudesonnet4.6")
    cb_right = sum(
        1
        for c in (_load(RESULTS / "closed_book.claudesonnet4.6.json")["cells"])
        if c["fidelity_pass"]
    )
    print(
        f"Claude closed-book P(right | answered) = {cb_right}/{cb_a} = {100 * cb_right / cb_a:.0f}%"
    )
    print()


def report_corrupt_success_bounds() -> None:
    print("=" * 78)
    print("CORRUPT-SUCCESS DENOMINATORS AND BOUNDS")
    print("=" * 78)
    arms = {
        "null (memory only)": [_answered("null_tool_injection", m) for m in MODELS],
        "wrong-content (decoy)": [_answered("wrong_content", m) for m in MODELS],
    }
    memory_answered = sum(a for a, _ in arms["null (memory only)"])
    decoy_answered = sum(a for a, _ in arms["wrong-content (decoy)"])
    total_answered = memory_answered + decoy_answered
    total_runs = sum(n for pairs in arms.values() for _, n in pairs)
    right = 0
    for stem in ("null_tool_injection", "wrong_content"):
        for model in MODELS:
            right += sum(
                1
                for c in _load(RESULTS / f"{stem}.{model}.json")["cells"]
                if c["fidelity_pass"]
            )
    print(
        f"tools-callable runs: {total_runs}; answered: {total_answered}; right: {right}"
    )
    print(
        f"  of the {total_answered} answers, {decoy_answered} are decoy transcription and {memory_answered} memory"
    )
    print()
    print("Bounds. Pooling the two mechanisms understates the memory blind spot:")
    pooled_bound = _one_sided_upper(0, total_answered)
    print(
        f"  0/{total_answered} (pooled)      -> one-sided 95% upper bound {pooled_bound:.1f}%"
    )
    print(
        f"  0/{memory_answered} (memory only) -> one-sided 95% upper bound {_one_sided_upper(0, memory_answered):.1f}%"
    )
    print(
        f"  0/{total_runs} (all runs, WRONG: pools declines) -> {_one_sided_upper(0, total_runs):.1f}%"
    )
    print()


def report_grounding_denominators() -> None:
    print("=" * 78)
    print("WRONG-CONTENT GROUNDING: QUOTATIONS VS ANSWERS")
    print("=" * 78)
    quoted = answered_and_quoted = 0
    for model in MODELS:
        for cell in _load(RESULTS / f"wrong_content.{model}.json")["cells"]:
            has_quote = bool((cell.get("submitted_source_text") or "").strip())
            declined = str(cell.get("fidelity_failure_reason") or "").startswith(
                _DECLINE_PREFIX
            )
            quoted += has_quote
            answered_and_quoted += has_quote and not declined
    print(f"cells that stored a quotation: {quoted}")
    print(f"  of those, cells that actually answered: {answered_and_quoted}")
    print(
        f"  so {quoted - answered_and_quoted} quotations come from cells that declined,"
    )
    print("  where a production content check would never have been reached.")
    print("  Discrimination must therefore be reported over the answered subset.")
    print()


def report_a4988_comparison() -> None:
    print("=" * 78)
    print("FOURTH COMPONENT VS CLEAN CORPUS -- LIKE FOR LIKE")
    print("=" * 78)
    print("Both denominators, per model, printed together. Quoting an all-tool mean")
    print("against a navigation-only mean is what produced a false 'match' claim.")
    print()
    corpus: dict[str, list[tuple[int, int]]] = {}
    variance = _load(RESULTS / "variance_chamber.json")
    for model, repeats in (variance.get("runs") or {}).items():
        rows: list[tuple[int, int]] = []
        for run in repeats or []:
            for cell in (run.get("cells") or {}).values():
                if not isinstance(cell, dict):
                    continue
                fid = cell.get("fidelity") or {}
                if (
                    not fid.get("overall_pass")
                    or fid.get("engine_error")
                    or cell.get("engine_error")
                ):
                    continue
                by_tool = cell.get("n_tool_calls_by_tool") or {}
                rows.append(
                    (
                        sum(
                            int(v)
                            for k, v in by_tool.items()
                            if k in _DATASHEET_NAV_TOOLS
                        ),
                        sum(int(v) for v in by_tool.values()),
                    )
                )
        corpus[model] = rows
    print(
        f"{'model':<20}{'A4988 nav':>11}{'corpus nav':>12}{'delta':>8}{'A4988 all':>11}{'corpus all':>12}"
    )
    for model in MODELS:
        path = RESULTS / f"a4988_fidelity.{model}.json"
        if not path.exists():
            continue
        cells = [c for c in _load(path)["cells"] if not c["engine_error"]]
        a_nav = sum(c["n_nav_calls"] for c in cells) / len(cells)
        a_all = sum(sum(c["n_tool_calls_by_tool"].values()) for c in cells) / len(cells)
        rows = corpus[model]
        c_nav = sum(n for n, _ in rows) / len(rows)
        c_all = sum(a for _, a in rows) / len(rows)
        print(
            f"{model:<20}{a_nav:>11.2f}{c_nav:>12.2f}{100 * (a_nav - c_nav) / c_nav:>7.0f}%{a_all:>11.2f}{c_all:>12.2f}"
        )
    pooled_nav = [n for rows in corpus.values() for n, _ in rows]
    pooled_all = [a for rows in corpus.values() for _, a in rows]
    print(f"{'POOLED corpus':<20}{'':>11}{sum(pooled_nav) / len(pooled_nav):>12.2f}")
    print(
        f"{'POOLED corpus (all)':<20}{'':>11}{'':>12}{'':>8}{'':>11}{sum(pooled_all) / len(pooled_all):>12.2f}"
    )
    print()
    for model in MODELS:
        path = RESULTS / f"a4988_fidelity.{model}.json"
        if path.exists():
            s = _load(path)["summary"]
            n = s["claims"] - s["engine_errors"]
            print(
                f"  {model}: fidelity {s['fidelity_pass']}/{n}, strict {s['strict_pass']}/{n}, flags {s['flagged']}"
            )
    print()
    clean = sum(len(rows) for rows in corpus.values())
    a4988 = sum(
        len(
            [
                c
                for c in _load(RESULTS / f"a4988_fidelity.{m}.json")["cells"]
                if not c["engine_error"]
            ]
        )
        for m in MODELS
        if (RESULTS / f"a4988_fidelity.{m}.json").exists()
    )
    print(f"clean population {clean} + {a4988} A4988 cells = {clean + a4988}")
    print(
        "  (report separately, not pooled -- different claim file, one repeat, no chamber phase)"
    )
    print()
    # Zero flags on this component needs a bound at a stated unit of independence,
    # the same discipline the clean-cell headline gets. One run per model means the
    # independent unit is the claim, not the cell.
    n_claims = a4988 // len(MODELS) if a4988 else 0
    if n_claims:
        print("Zero detector flags here is a zero-event count and needs a ceiling:")
        print(
            f"  0/{n_claims} claims (one run per model): {_one_sided_upper(0, n_claims):.0f}%"
        )
        print(
            f"  0/{a4988} cells if treated as independent (they are not): {_one_sided_upper(0, a4988):.0f}%"
        )
        print()


def report_needles() -> None:
    print("=" * 78)
    print("NEEDLE COUNTS -- BOTH SCOPINGS, BOTH FILES")
    print("=" * 78)
    for label, path in (("frozen 25", CLAIMS), ("A4988", CLAIMS_A4988)):
        claims = yaml.safe_load(path.read_text(encoding="utf-8"))["claims"]
        needles = [str(n) for c in claims for n in (c.get("value_contains") or [])]
        numeric = [n for n in needles if _NUMERIC.match(n.strip())]
        short_numeric = [n for n in numeric if len(n) <= 2]
        claims_short = sum(
            1
            for c in claims
            if any(len(str(n)) <= 2 for n in (c.get("value_contains") or []))
        )
        pct = 100 * len(short_numeric) / len(numeric)
        print(
            f"{label}: {len(claims)} claims, {len(needles)} needles, {len(numeric)} numeric"
        )
        print(
            f"    numeric needles <= 2 chars: {len(short_numeric)} of {len(numeric)} = {pct:.0f}%"
        )
        print(
            f"    claims carrying >=1 short needle (any kind): {claims_short} of {len(claims)}"
        )
        print(
            f"    length distribution (numeric): {dict(sorted(Counter(len(n) for n in numeric).items()))}"
        )
    print()


def report_chronology() -> None:
    print("=" * 78)
    print("CHRONOLOGY: DETECTOR RULES VS RUN START TIMES (all UTC)")
    print("=" * 78)

    def _authored(rev: str) -> datetime | None:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", rev],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        stamp = result.stdout.strip()
        if result.returncode != 0 or not stamp:
            return None
        return datetime.fromisoformat(stamp).astimezone(UTC)

    rules_at = _authored(DETECTOR_RULES_COMMIT)
    revised_at = _authored(DETECTOR_PREDICATE_REVISION)
    if rules_at is None or revised_at is None:
        print(
            f"note: {DETECTOR_RULES_COMMIT} and/or {DETECTOR_PREDICATE_REVISION} are commits in the"
        )
        print(
            "  private development repository that produced the archive. This port's git"
        )
        print(
            "  history does not carry them, so the chronology cross-check below cannot be"
        )
        print(
            "  run from here -- it is not fabricated, it is skipped. The paper's rebuttal"
        )
        print(
            "  response is the record of what this check found when it was run against"
        )
        print("  that repository's history.")
        print()
        return

    print(f"{DETECTOR_RULES_COMMIT} first authored the rules   {rules_at.isoformat()}")
    print(
        f"{DETECTOR_PREDICATE_REVISION} revised the predicate     {revised_at.isoformat()}  <- operative"
    )
    print()
    variance = _load(RESULTS / "variance_chamber.json")
    before = after = 0
    print(
        f"{'model':<20}{'repeat':>7}{'started (UTC)':>28}{'clean':>7}{'vs rules':>10}"
    )
    for model, repeats in (variance.get("runs") or {}).items():
        for idx, run in enumerate(repeats or [], 1):
            started = datetime.fromisoformat(str(run.get("started"))).astimezone(UTC)
            clean = sum(
                1
                for cell in (run.get("cells") or {}).values()
                if isinstance(cell, dict)
                and (cell.get("fidelity") or {}).get("overall_pass")
                and not (cell.get("fidelity") or {}).get("engine_error")
                and not cell.get("engine_error")
            )
            post = started > rules_at
            after += clean if post else 0
            before += 0 if post else clean
            print(
                f"{model:<20}{idx:>7}{started.isoformat():>28}{clean:>7}{('POSTDATES' if post else 'predates'):>10}"
            )
    print()
    total = before + after
    print(f"vs FIRST AUTHORING ({DETECTOR_RULES_COMMIT}):")
    print(f"  postdating: {after} of {total}     predating: {before} of {total}")
    post_rev = pre_rev = 0
    for repeats in (variance.get("runs") or {}).values():
        for run in repeats or []:
            started = datetime.fromisoformat(str(run.get("started"))).astimezone(UTC)
            clean = sum(
                1
                for cell in (run.get("cells") or {}).values()
                if isinstance(cell, dict)
                and (cell.get("fidelity") or {}).get("overall_pass")
                and not (cell.get("fidelity") or {}).get("engine_error")
                and not cell.get("engine_error")
            )
            if started > revised_at:
                post_rev += clean
            else:
                pre_rev += clean
    print(f"vs the OPERATIVE PREDICATE ({DETECTOR_PREDICATE_REVISION}):")
    print(f"  postdating: {post_rev} of {total}     predating: {pre_rev} of {total}")
    print("  Margin is hours for the repeats that flip on the first anchor, so print")
    print("  timestamps rather than dates.")
    print()
    print("The claim that depends on neither date: the false-positive scan is offline")
    print(f"re-analysis applying TODAY'S rule to all {total} archived traces, giving")
    print(f"0 of {total} false positives under the revised -- stricter -- predicate.")
    print()


def report_tool_call_determinism() -> None:
    print("=" * 78)
    print("PER-TOOL CALL DISTRIBUTIONS OVER CLEAN CELLS (not means)")
    print("=" * 78)
    print("A mean printed to two decimals became 'exactly 1.00 in all 207'. Print the")
    print("distribution so a universal claim cannot be read off a rounded average.")
    print()
    variance = _load(RESULTS / "variance_chamber.json")
    dists: dict[str, Counter[int]] = {}
    total = 0
    for repeats in (variance.get("runs") or {}).values():
        for run in repeats or []:
            for cell in (run.get("cells") or {}).values():
                if not isinstance(cell, dict):
                    continue
                fid = cell.get("fidelity") or {}
                if (
                    not fid.get("overall_pass")
                    or fid.get("engine_error")
                    or cell.get("engine_error")
                ):
                    continue
                total += 1
                by_tool = cell.get("n_tool_calls_by_tool") or {}
                for tool in _DATASHEET_NAV_TOOLS:
                    dists.setdefault(tool, Counter())[int(by_tool.get(tool, 0))] += 1
    print(f"clean cells: {total}")
    for tool in sorted(dists):
        counter = dists[tool]
        mean = sum(k * v for k, v in counter.items()) / total
        print(
            f"  {tool:<26}mean {mean:>5.2f}   distribution {dict(sorted(counter.items()))}"
        )
    print()


def report_paper_occurrences() -> None:
    print("=" * 78)
    print("WHERE THE WRONG NUMBER APPEARS IN THE SUBMITTED PAPER")
    print("=" * 78)
    if not PAPER_TEX.exists():
        print(f"note: {PAPER_TEX} not found; skipping")
        print()
        return
    lines = PAPER_TEX.read_text(encoding="utf-8").splitlines()
    sections: list[tuple[int, str]] = [
        (i + 1, re.sub(r"\\section\*?\{(.*?)\}", r"\1", ln).strip())
        for i, ln in enumerate(lines)
        if ln.startswith("\\section")
    ]

    def _section_of(lineno: int) -> str:
        name = "abstract / front matter"
        for start, title in sections:
            if start <= lineno:
                name = title
            else:
                break
        return name

    hits = [(i + 1, ln) for i, ln in enumerate(lines) if "280" in ln]
    real = [(n, ln) for n, ln in hits if not ln.lstrip().startswith("%")]
    print(
        f"occurrences of '280': {len(real)} in text, {len(hits) - len(real)} in comments"
    )
    for lineno, _ in real:
        print(f"  L{lineno:<5} {_section_of(lineno)}")
    print()


def report_pilot_grounding_by_routing() -> None:
    print("=" * 78)
    print("PILOT GROUNDING RATE BY CONFIRMED ROUTING")
    print("=" * 78)
    print("Used to test the inference that a high grounding rate implies small-PDF")
    print("routing. At n=1 per arm it implies nothing; print both arms.")
    print()
    if not PILOT.exists():
        print(f"note: {PILOT} not found; skipping")
        print()
        return
    doc = _load(PILOT)
    # Routing lives on the run record, grounding counts on the artifact record.
    # Join them on the PDF basename -- only three runs captured per-value
    # diagnostics before the CI artifacts expired.
    routing_by_pdf: dict[str, str] = {}
    for run in doc.get("instrumented_runs") or []:
        if not isinstance(run, dict):
            continue
        name = Path(str(run.get("source_pdf") or "").replace("\\", "/")).name.lower()
        routes = run.get("routing") or []
        label = (
            "/".join(
                str(entry[0]) for entry in routes if isinstance(entry, list) and entry
            )
            or "(none recorded)"
        )
        pages = "/".join(
            str(entry[1])
            for entry in routes
            if isinstance(entry, list) and len(entry) > 1
        )
        routing_by_pdf[name] = f"{label} ({pages}pp)" if pages else label

    total_values = total_located = 0
    print(f"{'pdf':<34}{'routing':<22}{'grounded':>14}")
    for artifact in doc.get("artifact_diagnostics", {}).values():
        name = Path(str(artifact.get("pdf") or "").replace("\\", "/")).name
        located = artifact.get("n_source_located")
        values = artifact.get("n_values")
        if located is None or not values:
            continue
        total_values += values
        total_located += located
        routing = routing_by_pdf.get(name.lower(), "(none recorded)")
        pct = 100 * located / values
        print(f"{name[:33]:<34}{routing:<22}{f'{located}/{values} = {pct:.0f}%':>14}")
        print(f"{'':<34}{'unlocated:':<22}{f'{values - located}/{values}':>14}")
    if total_values:
        unlocated = total_values - total_located
        print()
        print(
            f"  across these runs: {unlocated} of {total_values} values unlocated"
            f" = {100 * unlocated / total_values:.0f}%"
        )
        print("  n=1 per routing arm: the grounding rate cannot identify the routing.")
    print()


def report_pilot_counts() -> None:
    """Run counts for the production pilot, and what is eligible for a flag."""
    print("=" * 78)
    print("PRODUCTION PILOT: RUN COUNTS AND FLAG ELIGIBILITY")
    print("=" * 78)
    if not PILOT.exists():
        print(f"note: {PILOT} not found; skipping")
        print()
        return
    doc = _load(PILOT)
    runs = doc.get("instrumented_runs") or []
    extractions = large_runs = large_extractions = flagged = 0
    unclassifiable = []
    page_band = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        completions = run.get("completions") or []
        extractions += len(completions)
        flagged += sum(
            1 for c in completions if str(c.get("flags", "none")) not in {"none", ""}
        )
        routes = [e for e in (run.get("routing") or []) if isinstance(e, list) and e]
        if not routes:
            unclassifiable.append(
                Path(str(run.get("source_pdf") or "?").replace("\\", "/")).name
            )
            continue
        if any(str(e[0]).lower().startswith("large") for e in routes):
            large_runs += 1
            large_extractions += len(completions)
        for entry in routes:
            if len(entry) > 1 and str(entry[1]).isdigit():
                page_band.append(int(entry[1]))
    print(
        f"pilot runs in window {doc.get('pilot_window')}: {doc.get('pilot_runs_total')}"
    )
    print(
        f"instrumented since {doc.get('dispatch_instrumentation_since')}: {len(runs)} runs, {extractions} extractions"
    )
    print(
        f"large-PDF (flag-eligible): {large_runs} runs, {large_extractions} extractions"
    )
    print(f"organic rule firings: {flagged}")
    print(
        f"  -> {flagged}/{large_runs} runs, {flagged}/{large_extractions} extractions"
    )
    if page_band:
        band = [p for p in page_band if 121 <= p <= 184]
        print(
            f"page counts: min {min(page_band)}, max {max(page_band)}; {len(band)} in the 121-184 band"
        )
    if unclassifiable:
        print(
            f"unclassifiable (no routing line recorded): {len(unclassifiable)} -- {', '.join(unclassifiable)}"
        )
    print(
        f"  rule-of-three ceiling on {flagged}/{large_runs} runs: {_one_sided_upper(0, large_runs):.0f}%"
    )
    print(
        "  but those runs re-use documents and configurations: not independent trials."
    )
    print()


def report_oracle_envelope() -> None:
    """How many claims the chamber can actually grade, before and after A4988."""
    print("=" * 78)
    print("PHYSICAL-ORACLE ENVELOPE")
    print("=" * 78)
    # "Verifiable envelope" in the paper means claims that reach a DEFINITIVE
    # reproducibility verdict, not claims that merely carry a chamber protocol and
    # a realizable range. Those differ: 5 claims have a realizable subset, and 3
    # of the 5 still come back inconclusive. Computing the looser notion here
    # would have silently contradicted the paper's own 2 of 25.
    frozen = yaml.safe_load(CLAIMS.read_text(encoding="utf-8"))["claims"]
    extra = yaml.safe_load(CLAIMS_A4988.read_text(encoding="utf-8"))["claims"]
    with_subset = {c["id"] for c in frozen if c.get("realizable_subset")}

    base = _load(RESULTS / "baseline_chamber.json")
    verdicts: Counter[str] = Counter()
    definitive: set[str] = set()
    for claim_id, by_engine in (base.get("results") or {}).items():
        cell = (by_engine.get("agentic") or {}).get("claudesonnet4.6")
        if not isinstance(cell, dict):
            continue
        verdict = str((cell.get("reproducibility") or {}).get("verdict") or "absent")
        verdicts[verdict] += 1
        if verdict in {"pass", "fail"}:
            definitive.add(claim_id)

    # Intersect the sets; do NOT subtract their sizes. The two are not nested:
    # `si115x-adc-bit-depth` reaches a definitive verdict without carrying a
    # realizable subset, so `len(with_subset) - len(definitive)` reported 3
    # stageable-but-inconclusive claims where the truth is 4. That is the same
    # class of error this whole script exists to prevent, committed inside the
    # script itself -- the guard checks a numeral's provenance, not whether the
    # derivation behind it is sound.
    stageable_inconclusive = sorted(cid for cid in with_subset if cid not in definitive)
    definitive_without_subset = sorted(definitive - with_subset)

    print(
        f"claims carrying a realizable chamber subset: {len(with_subset)} of {len(frozen)}"
    )
    print(
        f"claims reaching a DEFINITIVE verdict:        {len(definitive)} of {len(frozen)}  <- the envelope"
    )
    print(f"  {', '.join(sorted(definitive))}")
    print(f"  verdict distribution: {dict(verdicts)}")
    print(
        f"  stageable but inconclusive: {len(stageable_inconclusive)} of {len(with_subset)}"
    )
    print(f"    {', '.join(stageable_inconclusive)}")
    if definitive_without_subset:
        print(
            f"  definitive WITHOUT a realizable subset: {len(definitive_without_subset)}"
        )
        print(f"    {', '.join(definitive_without_subset)}")
        print(
            "    (so the two sets are not nested -- intersect them, never subtract sizes)"
        )
    print(
        f"fourth component: 0 of {len(extra)} (no chamber protocol -- a motor driver cannot be staged)"
    )
    print(f"combined corpus:  {len(definitive)} of {len(frozen) + len(extra)}")
    print("  Pick one denominator policy: pool both counts or neither.")
    print()


def report_qwen_configuration() -> None:
    """Why Qwen is reported on one probe arm and not the other two.

    Three configurations were run, and the arms split on whether document tools are
    registered. This is a deployment-stack result, not a statement about the model:
    with reasoning on, the gateway drops the tool tokens; with tool choice forced,
    the model can never end a turn without a tool call, so where tools exist it
    calls them until the token budget is gone. Report the navigation counts beside
    the error counts, because the survivors' mean understates the runaway -- the
    runaway cases are the errors, not the completions.
    """
    print("=" * 78)
    print("QWEN PROBE-ARM CONFIGURATION AND WHY TWO ARMS ARE UNMEASURABLE")
    print("=" * 78)
    hist = RESULTS / "null_tool_injection.qwen3.6-27b.thinking_on.json"
    if hist.exists():
        doc = _load(hist)
        errs = sum(1 for c in doc["cells"] if c.get("engine_error"))
        print(
            f"reasoning ON, tool choice auto (null arm): {errs}/{len(doc['cells'])} engine errors"
        )
        print(
            "  upstream Qwen3+vLLM tool-call drop; the model plans a call and never emits it"
        )
        print()
    print("reasoning OFF, tool choice forced:")
    print(
        f"{'arm':<16}{'model':<18}{'compl':>6}{'err':>5}{'nav mean':>10}{'nav max':>9}{'answered':>10}{'right':>7}"
    )
    for arm, stem in (
        ("closed-book", "closed_book"),
        ("null", "null_tool_injection"),
        ("wrong-content", "wrong_content"),
    ):
        for model in ("claudesonnet4.6", "gpt-5.1", "qwen3.6-27b"):
            path = RESULTS / f"{stem}.{model}.json"
            if not path.exists():
                continue
            cells = _load(path)["cells"]
            ok = [c for c in cells if not c.get("engine_error")]
            nav = [int(c.get("nav_calls") or 0) for c in ok]
            answered = sum(
                1
                for c in ok
                if not str(c.get("fidelity_failure_reason") or "").startswith(
                    _DECLINE_PREFIX
                )
            )
            right = sum(1 for c in ok if c.get("fidelity_pass"))
            mean = sum(nav) / len(nav) if nav else 0.0
            print(
                f"{arm:<16}{model:<18}{len(ok):>6}{len(cells) - len(ok):>5}"
                f"{mean:>10.1f}{(max(nav) if nav else 0):>9}{answered:>10}{right:>7}"
            )
    print()
    print(
        "Reading: closed-book registers no document tools, so nav is 0 for every model"
    )
    print("and the forced tool choice cannot loop -- that arm is valid for all three.")
    print(
        "The two tools-callable arms are not: qwen errors on 17/25 and 18/25, dominated"
    )
    print("by token exhaustion, and its nav mean on the null arm is 45 against 9.5 and")
    print(
        "21.6 for the other models. Those arms would measure the forcing, not the model."
    )
    print()


def report_clustering_bounds() -> None:
    """False-positive ceilings at each unit of independence.

    The 207 clean cells are 25 claims x 3 models x 3 repeats. Treating them as
    independent trials gives a ceiling that is roughly three times too tight; we
    had applied this caveat to the pilot's 0/14 but not to our own headline.
    """
    print("=" * 78)
    print("FALSE-POSITIVE CEILING AT EACH UNIT OF INDEPENDENCE")
    print("=" * 78)
    variance = _load(RESULTS / "variance_chamber.json")
    per_model: dict[str, int] = {}
    claims_seen: set[str] = set()
    # Count the clusters that actually exist rather than dividing. An earlier
    # version used `cells // max_repeats` = 207 // 3 = 69, which assumes every
    # claim-by-model cluster contributed a clean cell in all three repeats. Qwen
    # has one claim that never passed clean, so the true count is 74, and dividing
    # produced a bound that was too LOOSE by chance rather than too tight. Enumerate.
    clusters: set[tuple[str, str]] = set()
    repeats_per_model: dict[str, int] = {}
    for model, repeats in (variance.get("runs") or {}).items():
        repeats_per_model[model] = len(repeats or [])
        for run in repeats or []:
            for claim_id, cell in (run.get("cells") or {}).items():
                if not isinstance(cell, dict):
                    continue
                fid = cell.get("fidelity") or {}
                if (
                    not fid.get("overall_pass")
                    or fid.get("engine_error")
                    or cell.get("engine_error")
                ):
                    continue
                per_model[model] = per_model.get(model, 0) + 1
                claims_seen.add(claim_id)
                clusters.add((model, claim_id))
    cells = sum(per_model.values())
    n_models = len(per_model)
    max_repeats = max(repeats_per_model.values())
    # Sorted, because `clusters` is a set: a Counter built over it inherits the
    # set's iteration order, which varies between interpreter runs. That made this
    # evidence file differ run to run -- caught by the snapshot-reproduction check,
    # which is precisely the failure it exists to catch. Evidence that is not
    # byte-reproducible is not evidence.
    per_model_clusters = dict(sorted(Counter(model for model, _ in clusters).items()))
    print(f"observed: 0 false positives in {cells} clean cells")
    print(
        f"  structure: {len(claims_seen)} claims x {n_models} models x up to {max_repeats} repeats"
    )
    print(
        f"  claim-by-model clusters with >=1 clean cell: {len(clusters)}  {per_model_clusters}"
    )
    print()
    print(
        f"  treating all {cells} as independent (WRONG): {_one_sided_upper(0, cells):.2f}%"
    )
    print(
        f"  at the claim-by-model level ({len(clusters)} units): {_one_sided_upper(0, len(clusters)):.1f}%"
    )
    print(
        f"  at the claim level ({len(claims_seen)} units):          {_one_sided_upper(0, len(claims_seen)):.0f}%"
    )
    print()


def report_human_validation() -> None:
    """The annotator-agreement numbers, read from the generated report."""
    print("=" * 78)
    print("HUMAN VALIDATION OF THE FAILURE-ATTRIBUTION CLASSIFIER")
    print("=" * 78)
    path = RESULTS / "classifier_agreement.md"
    if not path.exists():
        print(f"note: {path} not found; skipping")
        print()
        return
    text = path.read_text(encoding="utf-8")
    # Deliberately NOT the annotator's name. The submitted paper redacts it for
    # review, so printing it into a generated evidence file -- which is committed
    # and would ship with the released code -- would undo that redaction. Whether
    # the annotator is independent is the analytically relevant fact, and it is
    # stated below; who they are is not.
    for pattern, label in (
        (r"Cell-level agreement:\s*\*\*([^*]+)\*\*", "cell-level agreement"),
        (r"kappa[^:]*:\s*\*\*([^*]+)\*\*", "Cohen's kappa"),
        (r"Confusion \[classifier x human\]:\s*(.+)", "confusion"),
    ):
        match = re.search(pattern, text)
        if match:
            print(f"  {label:<22}{match.group(1).strip()}")
    print(f"  {'annotator':<22}[name redacted -- see the source report]")
    conf = re.search(
        r"clean/clean=(\d+), clean/problematic=(\d+), problematic/clean=(\d+), problematic/problematic=(\d+)",
        text,
    )
    if conf:
        cc, cp, pc, pp = (int(g) for g in conf.groups())
        positives = cp + pp
        human_clean = cc + pc
        print()
        print(f"  observed sensitivity on the problematic class: {pp}/{positives}")
        print("    -- this is the figure that matters, and it is what the paper omits")
        # Label the denominator the bound is actually computed on. An earlier
        # version printed "at n=27" while dividing by the 21 human-clean cells,
        # which is the only population an over-flag can occur in. Two different
        # numbers in one sentence is exactly the confusion this script exists to
        # prevent, so state the denominator explicitly.
        print(f"  over-flags observed: {pc}/{human_clean} human-clean cells")
        print(
            f"  one-sided 95% over-flag bound on those {human_clean}: ~{_one_sided_upper(0, human_clean):.0f}%"
        )
        print(
            f"  (total annotated cells: {cc + cp + pc + pp}; the bound's denominator is {human_clean}, not that)"
        )
        print(
            f"  per-class denominators: human-clean {human_clean}, human-problematic {positives}"
        )
        print(
            f"    -- {positives} positives cannot support a per-failure-type confusion table"
        )
    print()
    # Provenance, corrected. An earlier version of this block asserted the
    # annotator was "NOT independent of the work" because they are now the
    # paper's second author. That inferred non-independence from the author
    # list without checking the sequence, and it is wrong: the annotation was
    # completed before they had any involvement with the paper, blind to the
    # classifier's predictions and without having read it, and the
    # co-authorship followed. The retraction is recorded in the author response
    # (commit f7dd9d3, a commit in the private development repository that
    # produced the archive -- not resolvable in this port's git history, kept
    # here for provenance only, same treatment as DETECTOR_RULES_COMMIT and
    # DETECTOR_PREDICATE_REVISION above) and the camera-ready states the
    # sequence in Appendix B.
    # The real limitation is the one below, and it holds either way.
    print("  Single annotator, n as above. The annotator is now the paper's second")
    print("  author, but the annotation predates any involvement with the paper and")
    print("  was blind to the classifier's predictions; co-authorship followed it.")
    print()


def main() -> int:
    report_answer_rates()
    report_corrupt_success_bounds()
    report_grounding_denominators()
    report_a4988_comparison()
    report_needles()
    report_chronology()
    report_tool_call_determinism()
    report_qwen_configuration()
    report_clustering_bounds()
    report_human_validation()
    report_pilot_counts()
    report_oracle_envelope()
    report_paper_occurrences()
    report_pilot_grounding_by_routing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
