"""The Anthropic-shape engine survives the port, prompt text included."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

# Tier 1 -- the grading surface and the archive -- does not install the harness
# extra, and this module's import chain reaches `anthropic`. Skip rather than
# raise a collection error, so a Tier-1-only install still runs the suite that
# `README.md` and `docs/reproducing.md` point readers at.
pytest.importorskip(
    "anthropic",
    reason="needs the harness extra: uv pip install -e '.[harness]'",
)

from chamberbench.harness import anthropic_path

SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "chamberbench"
    / "harness"
    / "anthropic_path.py"
)


def test_entry_points_present_with_expected_defaults():
    sig = inspect.signature(anthropic_path.extract_chamber_agentic)
    assert sig.parameters["model"].default == "claudesonnet4.6"
    assert sig.parameters["max_turns"].default == 30
    assert sig.parameters["inspect_page_detail"].default == "high"
    assert (
        inspect.signature(anthropic_path.extract_chamber_baseline)
        .parameters["max_tokens"]
        .default
        == 32768
    )


def test_two_pass_freeze_prompt_text_is_intact():
    """PHASE 1/PHASE 2 are methodology inside the prompt. Scrubbing them
    would change agent behaviour, not just wording."""
    text = SRC.read_text(encoding="utf-8")
    assert "PHASE 1" in text
    assert "PHASE 2" in text


def test_provider_routing():
    assert anthropic_path._is_openai_model("gpt-5.1")
    assert anthropic_path._is_openai_model("o3-mini")
    assert not anthropic_path._is_openai_model("claudesonnet4.6")
    assert not anthropic_path._is_openai_model("qwen3.6-27b")


def test_dereferenced_schema_has_no_refs():
    """The gateway cannot compile $ref/anyOf; the invariant is load-bearing."""
    schema = anthropic_path._dereference_schema(
        {
            "$defs": {
                "Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}
            },
            "type": "object",
            "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        }
    )
    rendered = repr(schema)
    assert "$ref" not in rendered
    assert "$defs" not in rendered
    assert schema["properties"]["inner"]["properties"]["x"]["type"] == "integer"


class _RecordingClient:
    """Minimal stand-in for the Anthropic client: records create() kwargs.

    ``_run_one_turn`` awaits ``client.beta.messages.create(**kwargs)`` on the
    vLLM path, so this is the whole surface that path touches.
    """

    def __init__(self):
        self.kwargs: dict = {}
        outer = self

        class _Messages:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                return "sentinel-message"

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


async def _qwen_turn_kwargs() -> dict:
    client = _RecordingClient()
    result = await anthropic_path._run_one_turn(
        client=client,
        model="qwen3.6-27b",
        max_tokens=4096,
        system="s",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result == "sentinel-message"
    return client.kwargs


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("false", False),
        ("FALSE", False),
        (" false ", False),
        ("true", True),
        (None, True),
    ],
)
def test_qwen_thinking_flag_reaches_the_request(monkeypatch, env_value, expected):
    """CHAMBER_QWEN_ENABLE_THINKING must actually change the outbound request.

    This used to set the env var and then grep the source for two substrings,
    never reading the variable back -- so it passed whatever the parse did, and
    would have kept passing if the flag were wired to nothing. The flag is
    fidelity-critical (upstream QwenLM/Qwen3#1817: thinking on drops tool
    calls), so assert the value that lands in extra_body.
    """
    if env_value is None:
        monkeypatch.delenv("CHAMBER_QWEN_ENABLE_THINKING", raising=False)
    else:
        monkeypatch.setenv("CHAMBER_QWEN_ENABLE_THINKING", env_value)

    kwargs = asyncio.run(_qwen_turn_kwargs())
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is expected
