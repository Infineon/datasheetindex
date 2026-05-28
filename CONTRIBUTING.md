# Contributing to datasheetindex

Thank you for your interest in contributing! This document explains how to
report issues and submit changes.

Please also read our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep this
community welcoming and respectful.

## Reporting issues

- Search the [existing issues](https://github.com/Infineon/datasheetindex/issues)
  first to avoid duplicates.
- If none matches, open a new issue. Include the datasheet/PDF characteristics
  (page count, layout), the command you ran, the output you got, and the output
  you expected. A minimal reproduction is the fastest path to a fix.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                    # install runtime + dev dependencies
uv run pre-commit install  # install the pre-commit hooks
```

Optional extras:

```bash
uv sync --extra llm   # LLM-backed ToC fallback and summaries
uv sync --extra mcp   # local MCP server testing
```

## Making changes

1. Fork the repository and create a topic branch from `main`.
2. Make your change. Keep it focused; unrelated changes belong in separate PRs.
3. Run the full local check suite before pushing:

   ```bash
   uv run ruff check src/ tests/   # lint
   uv run ruff format src/ tests/  # format
   uv run ty check                 # type check
   uv run pytest                   # full test suite
   ```

   The pre-commit hooks enforce lint, format, type checking, and the fast test
   subset on every commit. Do not bypass them with `--no-verify`.
4. Add or update tests for any behavior change.

## Commit messages

Follow the existing conventional-commit style used in the history, e.g.
`feat: ...`, `fix: ...`, `chore: ...`, `docs: ...`. Write the message to explain
*why* the change is needed, not just *what* changed.

## Pull requests

- Open the PR against `main` and describe the motivation and the change.
- Link any related issue.
- Ensure CI passes. A maintainer will review your proposal and may request
  changes before merging.

By contributing, you agree that your contributions will be licensed under the
[MIT License](./LICENSE) that covers this project.
