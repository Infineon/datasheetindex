"""Measure what an answer costs against what the document costs.

The claim this exists to test is the one the library is built on: to answer a
question about a datasheet you do not need the datasheet in context. You need
the map ``build_datasheet`` returns, and the one section you then read.

So, per document, it counts three things:

``full``
    Every token of the page-matched text file -- what you paste in to "read
    the datasheet". This is a *conservative* baseline: attaching the PDF
    itself costs more, since the pages arrive as images too.
``manifest``
    What ``get_artifact_manifest`` actually hands the agent -- the enriched
    ToC and the build's signals.
``section``
    What ``get_section_text`` returns for one leaf ToC section, summarised
    across every leaf as a median and a p90.

An *answer* is priced at ``manifest + median section``, and the headline
number is ``full / answer``.

There is no model in any of this and no network call. Nothing here measures
whether the answer is *right*; that is the benchmark's job, and it needs
ground truth this script deliberately does not have. See ``benchmark/``.

Usage::

    uv run --group bench python scripts/token_economy.py --corpus DIR \\
        --markdown docs/token-economy.md --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# The encoding, not a model name: ``o200k_base`` is what current GPT-family
# models tokenize with, and it is within a few percent of Claude's tokenizer
# on English technical prose. The ratio we publish is a ratio of two counts
# from the *same* encoder, so the residual error largely cancels.
ENCODING = "o200k_base"

Encoder = Callable[[str], int]


def get_encoder(encoding: str = ENCODING) -> Encoder:
    """Return a token counter, or explain how to get one.

    ``tiktoken`` lives in the ``bench`` dependency-group rather than ``dev``:
    a plain ``uv sync`` is the environment CI and the pre-commit hook run in,
    and nothing there needs a BPE table.
    """
    try:
        import tiktoken
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise SystemExit(
            "token_economy needs tiktoken; run it with "
            "`uv run --group bench python scripts/token_economy.py ...`"
        ) from exc

    enc = tiktoken.get_encoding(encoding)
    return lambda text: len(enc.encode(text, disallowed_special=()))


def iter_leaf_sections(toc: list[dict]) -> Iterator[dict]:
    """Yield, in document order, the ToC nodes an agent would actually read.

    Only leaves: reading a parent re-reads its children, so counting both
    would price the same text twice. Nodes whose range cannot be read back are
    skipped rather than counted as zero -- ``TocNode.end_page`` defaults to 0,
    so an un-enriched tree really does serialize leaves that
    ``get_section_text`` would reject.
    """
    for node in toc:
        children = node.get("nodes") or []
        if children:
            yield from iter_leaf_sections(children)
            continue
        start = node.get("start_page", 0)
        end = node.get("end_page", 0)
        if start >= 1 and end >= start:
            yield node


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile.

    Not ``statistics.quantiles``: it interpolates and needs at least two data
    points, and plenty of small datasheets have exactly one readable section.
    """
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered))))
    return ordered[rank - 1]


@dataclass
class DocumentMeasurement:
    """One document's numbers."""

    name: str
    pages: int
    full_tokens: int
    manifest_tokens: int
    section_tokens: list[int]
    cold_build_s: float
    warm_build_s: float
    warm_reused: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def section_median(self) -> int:
        if not self.section_tokens:
            return 0
        return int(statistics.median(self.section_tokens))

    @property
    def section_p90(self) -> int:
        return _percentile(self.section_tokens, 0.9)

    @property
    def answer_tokens(self) -> int | None:
        """Manifest plus one section, or ``None`` when there is no section.

        A document with no readable ToC is genuinely not priced by this
        script. Falling back to ``full / manifest`` there would flatter the
        result: without a section map the agent reaches for ``search_text``,
        whose cost this does not measure.
        """
        if not self.section_tokens:
            return None
        return self.manifest_tokens + self.section_median

    @property
    def followup_tokens(self) -> int | None:
        """What the *next* question about the same part costs.

        The manifest is paid once per document, not once per question: it is
        already in context, and the artifact it describes is already on disk.
        Pricing every question at ``manifest + section`` therefore understates
        the library on the workload it exists for, which is an agent asking
        several things about one part.
        """
        if not self.section_tokens:
            return None
        return self.section_median

    @property
    def followup_ratio(self) -> float | None:
        followup = self.followup_tokens
        if not followup:
            return None
        return self.full_tokens / followup

    @property
    def ratio(self) -> float | None:
        answer = self.answer_tokens
        if not answer:
            return None
        return self.full_tokens / answer

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pages": self.pages,
            "full_tokens": self.full_tokens,
            "manifest_tokens": self.manifest_tokens,
            "sections": len(self.section_tokens),
            "section_median": self.section_median,
            "section_p90": self.section_p90,
            "answer_tokens": self.answer_tokens,
            "ratio": self.ratio,
            "followup_tokens": self.followup_tokens,
            "followup_ratio": self.followup_ratio,
            "cold_build_s": round(self.cold_build_s, 2),
            "warm_build_s": round(self.warm_build_s, 3),
            "warm_reused": self.warm_reused,
            "notes": self.notes,
        }


def measure_document(
    pdf_path: Path | str,
    encode: Encoder,
    output_dir: str,
) -> DocumentMeasurement:
    """Build one datasheet and price it.

    The cold pass forces a rebuild so the timing is a real build rather than
    whatever happened to be on disk. The warm pass then runs through a
    **fresh** instance: ``build_datasheet`` short circuits in memory, so
    reusing the first one would time an attribute lookup and call it a cache
    hit. ``warm_reused`` records whether the artifact files were left
    untouched, which is what makes the warm number honest.
    """
    from datasheetindex import DatasheetTools

    pdf_path = Path(pdf_path)
    notes: list[str] = []

    with DatasheetTools(str(pdf_path)) as tools:
        started = time.perf_counter()
        artifacts = tools.build_datasheet(
            output_dir=output_dir,
            caption_figures=False,
            force_rebuild=True,
        )
        cold_build_s = time.perf_counter() - started

        manifest = tools.get_artifact_manifest()
        # ``json_path``/``text_path`` are absolute and machine-specific. They
        # are a handful of tokens and the agent really is sent them, so they
        # stay in the count rather than being trimmed to make the number look
        # tidier.
        manifest_tokens = encode(json.dumps(manifest, ensure_ascii=False))
        full_tokens = encode(artifacts.text_content)
        raw_pages = manifest.get("total_pages")
        pages = raw_pages if isinstance(raw_pages, int) else 0

        # The manifest is typed dict[str, object], so narrow rather than cast:
        # an artifact with no tree really does carry no list here.
        raw_toc = manifest.get("toc")
        toc: list[dict] = raw_toc if isinstance(raw_toc, list) else []
        section_tokens: list[int] = []
        for leaf in iter_leaf_sections(toc):
            text = tools.get_section_text(leaf["start_page"], leaf["end_page"])
            section_tokens.append(encode(text))

    if not section_tokens:
        notes.append("no readable ToC section; not priced")

    stamps_before = _artifact_stamps(output_dir)
    with DatasheetTools(str(pdf_path)) as tools:
        started = time.perf_counter()
        tools.build_datasheet(output_dir=output_dir, caption_figures=False)
        warm_build_s = time.perf_counter() - started
    warm_reused = bool(stamps_before) and _artifact_stamps(output_dir) == stamps_before
    if not warm_reused:
        notes.append("warm build rewrote the artifact; timing is not a cache hit")

    return DocumentMeasurement(
        name=pdf_path.name,
        pages=pages,
        full_tokens=full_tokens,
        manifest_tokens=manifest_tokens,
        section_tokens=section_tokens,
        cold_build_s=cold_build_s,
        warm_build_s=warm_build_s,
        warm_reused=warm_reused,
        notes=notes,
    )


def _artifact_stamps(output_dir: str) -> dict[str, int]:
    directory = Path(output_dir)
    if not directory.is_dir():
        return {}
    return {p.name: p.stat().st_mtime_ns for p in directory.iterdir() if p.is_file()}


def summarize(measurements: list[DocumentMeasurement]) -> dict:
    """Aggregate across documents.

    ``documents`` counts everything measured and ``priced`` only those a ratio
    could be computed for, so the gap between them is visible rather than
    hidden by averaging over a smaller denominator.
    """
    ratios = [m.ratio for m in measurements if m.ratio is not None]
    followups = [m.followup_ratio for m in measurements if m.followup_ratio is not None]
    warm_hits = [m.warm_build_s for m in measurements if m.warm_reused]
    return {
        "encoding": ENCODING,
        "documents": len(measurements),
        "priced": len(ratios),
        "median_ratio": statistics.median(ratios) if ratios else None,
        "min_ratio": min(ratios) if ratios else None,
        "max_ratio": max(ratios) if ratios else None,
        "median_followup_ratio": (statistics.median(followups) if followups else None),
        "median_followup_tokens": (
            int(
                statistics.median(
                    [
                        m.followup_tokens
                        for m in measurements
                        if m.followup_tokens is not None
                    ]
                )
            )
            if followups
            else None
        ),
        "median_full_tokens": (
            int(statistics.median([m.full_tokens for m in measurements]))
            if measurements
            else None
        ),
        "median_answer_tokens": (
            int(
                statistics.median(
                    [
                        m.answer_tokens
                        for m in measurements
                        if m.answer_tokens is not None
                    ]
                )
            )
            if ratios
            else None
        ),
        "median_cold_build_s": (
            round(statistics.median([m.cold_build_s for m in measurements]), 2)
            if measurements
            else None
        ),
        # Only over documents whose warm pass really was a cache hit. Averaging
        # a missed cache into this would advertise a rebuild as a reuse -- the
        # exact number the per-row column refuses to print.
        "warm_hits": len(warm_hits),
        "median_warm_build_s": (
            round(statistics.median(warm_hits), 3) if warm_hits else None
        ),
    }


def _fmt(value: float | int | None, spec: str = ",") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def render_markdown(
    measurements: list[DocumentMeasurement],
    summary: dict,
) -> str:
    """Render the committed table.

    Generated, so it carries the command that regenerates it rather than an
    invitation to hand-edit.
    """
    lines = [
        "<!-- Generated by scripts/token_economy.py. Do not edit by hand. -->",
        "# Token economy",
        "",
        f"Tokens counted with the `{summary['encoding']}` encoding. "
        f"{summary['documents']} documents, {summary['priced']} priced.",
        "",
        "| Document | Pages | Full doc | Manifest | Section (med) | "
        "Section (p90) | First answer | vs full | Further answer | vs full | "
        "Build (cold) | Build (warm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in sorted(measurements, key=lambda m: m.pages):
        ratio = "n/a" if m.ratio is None else f"{m.ratio:.1f}x"
        followup = "n/a" if m.followup_ratio is None else f"{m.followup_ratio:.0f}x"
        warm = "n/a" if not m.warm_reused else f"{m.warm_build_s:.2f}s"
        lines.append(
            f"| {m.name} | {m.pages} | {_fmt(m.full_tokens)} | "
            f"{_fmt(m.manifest_tokens)} | {_fmt(m.section_median)} | "
            f"{_fmt(m.section_p90)} | {_fmt(m.answer_tokens)} | {ratio} | "
            f"{_fmt(m.followup_tokens)} | {followup} | "
            f"{m.cold_build_s:.1f}s | {warm} |"
        )
    median = summary["median_ratio"]
    lines += [
        "",
        "## Headline",
        "",
        f"- Median full document: **{_fmt(summary['median_full_tokens'])}** tokens",
        f"- Median **first** answer (manifest + one section): "
        f"**{_fmt(summary['median_answer_tokens'])}** tokens, "
        f"**{'n/a' if median is None else f'{median:.1f}x'}** cheaper than the "
        "document"
        + (
            ""
            if summary["min_ratio"] is None
            else f" (range {summary['min_ratio']:.1f}x-{summary['max_ratio']:.1f}x)"
        ),
        f"- Median **further** answer about the same part (one section, "
        f"manifest already in context): "
        f"**{_fmt(summary['median_followup_tokens'])}** tokens, "
        + (
            "n/a"
            if summary["median_followup_ratio"] is None
            else f"**{summary['median_followup_ratio']:.0f}x** cheaper"
        ),
        f"- Median build: **{_fmt(summary['median_cold_build_s'], '.2f')}s** cold, "
        + (
            f"**{summary['median_warm_build_s']:.2f}s** warm "
            f"({summary['warm_hits']} of {summary['documents']} cache hits)"
            if summary["median_warm_build_s"] is not None
            else "warm **n/a** (no cache hit; see below)"
        ),
        "",
    ]
    if summary["warm_hits"] < summary["documents"]:
        lines += [
            f"> {summary['documents'] - summary['warm_hits']} of "
            f"{summary['documents']} warm builds were not cache hits. Artifact "
            "reuse is disabled on an editable install, so measuring from a "
            "checkout reports no warm timing at all; run from a non-editable "
            "install, or pass `--allow-editable-reuse` to measure the cache "
            "path an installed consumer takes.",
            "",
        ]
    lines += [
        "## How to read this",
        "",
        "**`Full doc` is a conservative baseline.** It is the page-matched "
        "text file -- what you paste in to read the document. Attaching the "
        "PDF itself costs more, because the pages arrive as images too.",
        "",
        "**The two ratios answer different questions.** The first answer pays "
        "for the map as well as the section, and on large documents the map "
        "is most of the cost: the enriched ToC of a 322-page datasheet is "
        "itself half that document's tokens. Every question after it about "
        "the same part pays for a section alone, which is where the ratio "
        "gets large. An agent that asks one thing and leaves sees the first "
        "ratio; an agent doing real extraction work sees the second.",
        "",
        "**Nothing here measures accuracy.** A cheap wrong answer is worth "
        "nothing, and this script has no ground truth to check against. That "
        "is the benchmark's job -- see `benchmark/`.",
        "",
        "**One section is assumed to be enough.** A question spanning "
        "sections costs more, and a document with no section map costs "
        "whatever `search_text` costs, which is not measured here. Documents "
        "in that state are listed as `n/a` rather than priced.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "uv run --group bench python scripts/token_economy.py \\",
        "    --corpus <dir> --markdown docs/token-economy.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="directory of PDF datasheets to measure",
    )
    parser.add_argument("--markdown", type=Path, help="write the rendered table here")
    parser.add_argument("--json", type=Path, help="write the raw measurements here")
    parser.add_argument(
        "--allow-editable-reuse",
        action="store_true",
        help=(
            "measure the warm build as an installed consumer would see it. "
            "Artifact reuse is disabled on an editable install by design, so "
            "without this a run from a checkout reports no warm timing at all"
        ),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(".token-economy-artifacts"),
        help="where builds are cached (one subdirectory per document)",
    )
    args = parser.parse_args(argv)

    pdfs = sorted(args.corpus.glob("*.pdf"))
    if not pdfs:
        parser.error(f"no PDFs found in {args.corpus}")

    if args.allow_editable_reuse:
        # Deliberately narrow: it disables the editable-install rule and
        # nothing else, so the warm pass exercises the same on-disk cache a
        # wheel install would. The cold pass is unaffected -- it forces a
        # rebuild regardless.
        import datasheetindex.tools.bound as bound

        def _not_editable() -> bool:
            return False

        # setattr, because rebinding a module-level function is exactly the
        # kind of thing a type checker should object to. It is deliberate here
        # and confined to this opt-in flag.
        setattr(bound, "is_editable_install", _not_editable)  # noqa: B010

    encode = get_encoder()
    measurements: list[DocumentMeasurement] = []
    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}", file=sys.stderr, flush=True)
        # One subdirectory per document: the cold pass forces a rebuild, and a
        # shared directory would let one document's artifacts age out another's.
        out = args.artifacts / pdf.stem
        out.mkdir(parents=True, exist_ok=True)
        try:
            measurements.append(measure_document(pdf, encode, str(out)))
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not end the run
            print(f"    failed: {exc}", file=sys.stderr, flush=True)

    summary = summarize(measurements)
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "documents": [m.to_dict() for m in measurements],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    rendered = render_markdown(measurements, summary)
    if args.markdown:
        args.markdown.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
