"""Tests for CLI interface."""

from __future__ import annotations

from pathlib import Path

from datasheetindex.models import DatasheetArtifacts


class _FakeIndex:
    def __init__(self, source: str) -> None:
        self.source = source
        self.closed = False

    def build(
        self,
        output_dir: str = "output",
        include_summaries: bool = False,
        llm_callable=None,
    ) -> DatasheetArtifacts:
        _ = include_summaries, llm_callable
        return DatasheetArtifacts(
            json_path=Path(output_dir) / "fake.json",
            text_path=Path(output_dir) / "fake.txt",
        )

    def close(self) -> None:
        self.closed = True


def test_cli_build_success(monkeypatch, capsys):
    from datasheetindex import cli

    monkeypatch.setattr("datasheetindex.cli.DatasheetIndex", _FakeIndex)
    exit_code = cli.main(
        ["build", "https://example.com/test.pdf", "--output-dir", "out"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "JSON: out" in captured.out
    assert "TEXT: out" in captured.out


def test_cli_include_summaries_requires_model(capsys):
    from datasheetindex import cli

    exit_code = cli.main(["build", "input.pdf", "--include-summaries"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--include-summaries requires --model" in captured.err


def test_cli_build_error_returns_nonzero(monkeypatch, capsys):
    from datasheetindex import cli

    class _RaisingIndex(_FakeIndex):
        def build(self, *args, **kwargs):
            raise ValueError("boom")

    monkeypatch.setattr("datasheetindex.cli.DatasheetIndex", _RaisingIndex)
    exit_code = cli.main(["build", "input.pdf"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: boom" in captured.err


def test_cli_default_output_dir_is_output(monkeypatch, capsys):
    """CLI must keep the interactive default of ./output/ -- not the resolver."""
    from datasheetindex import cli

    captured_output_dir: dict[str, str] = {}

    class _CapturingIndex(_FakeIndex):
        def build(
            self,
            output_dir: str = "output",
            include_summaries: bool = False,
            llm_callable=None,
        ) -> DatasheetArtifacts:
            captured_output_dir["value"] = output_dir
            return super().build(
                output_dir=output_dir,
                include_summaries=include_summaries,
                llm_callable=llm_callable,
            )

    monkeypatch.setattr("datasheetindex.cli.DatasheetIndex", _CapturingIndex)
    exit_code = cli.main(["build", "input.pdf"])
    capsys.readouterr()

    assert exit_code == 0
    assert captured_output_dir["value"] == "output"


def test_cli_build_with_model_uses_llm_client(monkeypatch):
    from datasheetindex import cli

    calls: list[str] = []
    closed = {"value": False}

    class _CloseableLlm:
        def __call__(self, _system: str, _user: str) -> str:
            return "ok"

        def close(self) -> None:
            closed["value"] = True

    def _fake_client(model: str):
        calls.append(model)
        return _CloseableLlm()

    monkeypatch.setattr("datasheetindex.cli.DatasheetIndex", _FakeIndex)
    monkeypatch.setattr("datasheetindex.llm.client.create_llm_client", _fake_client)

    exit_code = cli.main(["build", "input.pdf", "--model", "gpt-4.1"])

    assert exit_code == 0
    assert calls == ["gpt-4.1"]
    assert closed["value"] is True


def test_cli_mcp_defaults_to_stdio(monkeypatch):
    """`datasheetindex mcp` with no arguments must serve stdio.

    The registry entry invokes exactly this, with no arguments.
    """
    from datasheetindex import cli, mcp_server

    calls: list[tuple[str, str, int, str]] = []

    def _fake_run(transport, host, port, streamable_http_path):
        calls.append((transport, host, port, streamable_http_path))

    monkeypatch.setattr(mcp_server, "run_mcp_server", _fake_run)

    exit_code = cli.main(["mcp"])

    assert exit_code == 0
    assert calls == [("stdio", "127.0.0.1", 8000, "/mcp")]


def test_cli_mcp_passes_through_options(monkeypatch):
    from datasheetindex import cli, mcp_server

    calls: list[tuple[str, str, int, str]] = []

    def _fake_run(transport, host, port, streamable_http_path):
        calls.append((transport, host, port, streamable_http_path))

    monkeypatch.setattr(mcp_server, "run_mcp_server", _fake_run)

    exit_code = cli.main(
        [
            "mcp",
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--streamable-http-path",
            "/inspect",
        ]
    )

    assert exit_code == 0
    assert calls == [("streamable-http", "0.0.0.0", 9000, "/inspect")]


def test_cli_mcp_reports_error_without_extra(monkeypatch, capsys):
    """A missing [mcp] extra must be a clean message, not a traceback."""
    from datasheetindex import cli, mcp_server

    def _raise(**kwargs):
        _ = kwargs
        raise ImportError("mcp is required for local MCP server support.")

    monkeypatch.setattr(mcp_server, "run_mcp_server", _raise)

    exit_code = cli.main(["mcp"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "mcp is required" in captured.err
