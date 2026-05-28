"""Command-line interface for datasheetindex."""

from __future__ import annotations

import argparse
import sys

from datasheetindex.index import DatasheetIndex
from datasheetindex.llm.client import close_llm_client


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datasheetindex")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Build artifacts from a datasheet path or URL",
    )
    build_parser.add_argument("source", help="Local PDF path or http(s) URL")
    build_parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write output artifacts",
    )
    build_parser.add_argument(
        "--include-summaries",
        action="store_true",
        help=(
            "Force section summaries with the explicit --model client "
            "(automatic fallback may still add recommended summaries when "
            "default LLM credentials are available)"
        ),
    )
    build_parser.add_argument(
        "--model",
        help=(
            "Explicit LLM model for ToC fallback and summaries; without it, "
            "low-quality ToCs may still use the default auto-fallback model "
            "when LLM credentials are available"
        ),
    )
    return parser


def _run_build(args: argparse.Namespace) -> int:
    llm_callable = None
    if args.include_summaries and not args.model:
        print("Error: --include-summaries requires --model", file=sys.stderr)
        return 2
    idx = DatasheetIndex(args.source)
    try:
        if args.model:
            from datasheetindex.llm.client import create_llm_client

            llm_callable = create_llm_client(model=args.model)

        artifacts = idx.build(
            output_dir=args.output_dir,
            include_summaries=args.include_summaries,
            llm_callable=llm_callable,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        idx.close()
        close_llm_client(llm_callable)

    print(f"JSON: {artifacts.json_path}")
    print(f"TEXT: {artifacts.text_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return an exit code (for programmatic use / tests)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return _run_build(args)
    parser.print_help()
    return 2


def main_cli() -> None:
    """Entry point for console_scripts (properly propagates exit codes)."""
    raise SystemExit(main())


if __name__ == "__main__":
    main_cli()
