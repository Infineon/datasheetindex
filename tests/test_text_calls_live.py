"""Live text and structured calls against the real gateway. Skipped without credentials.

The counterpart to ``tests/test_figure_captions_live.py``, and it exists for the
same reason on the other half of the client. 0.28.0 moved captioning to Chat
Completions after measuring a silent 50% caption loss over the Responses API and
left that file behind as a permanent tripwire; 0.30.0 moved the text and
structured calls for the same measured reason and would otherwise have left none.

The credential-skip pattern is ``_has_env`` (in ``tests/conftest.py``), which
skips when ``.env`` is missing, when ``python-dotenv``/``httpx``/``openai`` are
absent (a plain ``uv sync`` excludes the ``[llm]`` extra), or when the two
LiteLLM variables are unset -- and which opts these tests out of the autouse
``_hermetic_llm_env`` fixture that scrubs credentials everywhere else.
"""

from __future__ import annotations

import json

import pytest

#: Small enough that a reply cannot plausibly hit the model's default output cap,
#: since a truncated answer would fail the JSON assertion below for a reason that
#: has nothing to do with the transport under test.
SAMPLE_TEXT = (
    "--- PAGE 1 ---\n"
    "TLE9350BSJ High-speed CAN transceiver\nData Sheet Rev. 1.2\n\n"
    "--- PAGE 2 ---\n"
    "1 Overview\nThe device provides an interface between the CAN protocol\n"
    "controller and the physical bus.\n\n"
    "--- PAGE 3 ---\n"
    "2 Block diagram\n3 Pin configuration\n"
)

#: Six calls put a 50%-per-call failure at well under 1 in 64. A one-shot test
#: passes roughly half the time on a broken transport, which is worse than no
#: test at all.
ATTEMPTS = 6


def _resolved_text_model(client: object) -> str | None:
    return getattr(client, "_model", None)


@pytest.mark.usefixtures("_has_env")
@pytest.mark.integration
def test_text_calls_never_return_empty_over_repeated_calls():
    """The regression guard for the text transport, which one call cannot be.

    Over the Responses API this path returned an *empty* string for 7 to 8 of 15
    real ToC chunks against the self-hosted ``qwen3.6-27b``, and 1 of 1 on the
    short PCN fixture: the gateway's bridge filed the answer as a ``reasoning``
    item, which ``output_text`` ignores. Caught directly on a raw request -- same
    model, same prompt, two runs -- as ``output`` item types ``["message"]`` with
    3727 characters and then ``["reasoning"]`` with 0, both reporting
    ``status: "completed"``.

    **Each call must vary the prompt.** The gateway caches identical payloads
    (repeats return the same completion id plus an ``x-litellm-cache-key``
    header), so six identical requests are one call and five replays, and a
    replayed success cannot reveal a per-call coin flip.

    **It only has teeth against a model the gateway bridges.** On the default
    ``gpt-4.1`` this passed on the *broken* code too -- 0 empty in 90 calls --
    which is precisely why the bug survived to 0.30.0 with two live integration
    tests already pointed at this path. Set ``DATASHEETINDEX_MODEL`` to a
    self-hosted or otherwise non-native model to give it teeth. The resolved
    model is named in the failure message so a green run is not mistaken for
    coverage it did not provide, and no internal alias is hardcoded here: it
    would fail outright on a gateway that does not serve it.
    """
    from datasheetindex.llm.client import close_llm_client, create_llm_client
    from datasheetindex.llm.toc_fallback import INIT_USER_PROMPT, SYSTEM_PROMPT
    from datasheetindex.llm.untrusted import wrap_document_text

    client = create_llm_client()
    resolved_model = _resolved_text_model(client)
    try:
        replies = [
            client(
                SYSTEM_PROMPT,
                INIT_USER_PROMPT.format(text=wrap_document_text(SAMPLE_TEXT))
                + f"\n\n(attempt {attempt})",
            )
            for attempt in range(ATTEMPTS)
        ]
    finally:
        close_llm_client(client)

    empty = [i for i, reply in enumerate(replies) if not reply.strip()]
    assert not empty, (
        f"model {resolved_model!r} returned empty text replies at attempts "
        f"{empty} of {len(replies)}"
    )


@pytest.mark.usefixtures("_has_env")
@pytest.mark.integration
def test_structured_calls_return_parseable_json_over_repeated_calls():
    """The same guard for the structured path, plus the risk 0.30.0 introduced.

    Two failures are in scope here and they are not the same. The first is the
    empty reply the sibling test above describes, which reached the ToC fallback
    as ``""`` and made it retry the entire document over the same transport for
    nothing. The second is new: moving to Chat Completions re-expressed the
    schema request as ``response_format={"type": "json_schema", ...}``, one
    nesting level deeper than the Responses API's ``text.format``, and
    strict-schema support is **not** uniform across gateway backends. Verified on
    gpt-4.1 and qwen3.6-27b before shipping; this keeps it verified.

    Parsing is asserted rather than merely non-emptiness because that is the
    contract ``_parse_structured_chunk_response`` depends on -- a backend that
    silently ignores ``strict`` returns prose, which is not an empty reply and
    would slip past the other assertion.

    Teeth, prompt variation and the absence of a hardcoded alias all work as
    described on the sibling test above.
    """
    from datasheetindex.llm.client import (
        close_llm_client,
        create_llm_client,
        get_structured_output_client,
    )
    from datasheetindex.llm.toc_fallback import (
        STRUCTURED_INIT_USER_PROMPT,
        STRUCTURED_SYSTEM_PROMPT,
        STRUCTURED_TOC_SCHEMA,
    )
    from datasheetindex.llm.untrusted import wrap_document_text

    client = create_llm_client()
    resolved_model = _resolved_text_model(client)
    try:
        structured = get_structured_output_client(client)
        assert structured is not None
        results = [
            structured.structured_json(
                STRUCTURED_SYSTEM_PROMPT,
                STRUCTURED_INIT_USER_PROMPT.format(text=wrap_document_text(SAMPLE_TEXT))
                + f"\n\n(attempt {attempt})",
                name="datasheet_toc_chunk",
                schema=STRUCTURED_TOC_SCHEMA,
            )
            for attempt in range(ATTEMPTS)
        ]
    finally:
        close_llm_client(client)

    empty = [i for i, result in enumerate(results) if not result.output_text.strip()]
    assert not empty, (
        f"model {resolved_model!r} returned empty structured replies at attempts "
        f"{empty} of {len(results)}"
    )

    # A truncated reply is a real but different failure, and reporting it as
    # "invalid JSON" would send the next reader after the wrong cause.
    truncated = [i for i, result in enumerate(results) if result.status == "incomplete"]
    assert not truncated, (
        f"model {resolved_model!r} truncated structured replies at attempts "
        f"{truncated}: {[results[i].incomplete_details for i in truncated]}"
    )

    for attempt, result in enumerate(results):
        try:
            payload = json.loads(result.output_text)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"model {resolved_model!r} ignored the strict JSON schema on "
                f"attempt {attempt}; reply began {result.output_text[:120]!r}"
            ) from exc
        assert isinstance(payload, dict), (
            f"model {resolved_model!r} returned a non-object payload on attempt "
            f"{attempt}"
        )
        assert isinstance(payload.get("entries"), list), (
            f"model {resolved_model!r} returned no 'entries' list on attempt {attempt}"
        )
