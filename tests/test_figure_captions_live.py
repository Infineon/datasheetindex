"""Live VLM captioning against the real gateway. Skipped without credentials.

Follows the credential-skip pattern used throughout the suite for real-gateway
tests (see ``tests/test_summarizer.py``, ``tests/test_llm_client.py``,
``tests/test_toc_fallback.py``): ``_has_env`` (in ``tests/conftest.py``) skips
when ``.env`` is missing, when ``python-dotenv``/``httpx``/``openai`` are not
installed (a plain ``uv sync`` excludes the ``[llm]`` extra), or when the two
LiteLLM environment variables are unset -- and it opts this test out of the
autouse ``_hermetic_llm_env`` fixture that otherwise scrubs LITELLM
credentials from every test for hermeticity.
"""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("_has_env")
@pytest.mark.integration
def test_describe_image_returns_a_single_line_for_a_real_render():
    import pymupdf

    from datasheetindex.llm.client import (
        close_llm_client,
        create_llm_client,
        get_vision_client,
    )
    from datasheetindex.tools.vision import inspect_page

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(100, 100, 500, 400), color=(0, 0, 0), width=2)
    page.insert_text((150, 200), "VCC", fontsize=24)
    blocks = inspect_page(doc, page=1)
    doc.close()

    client = create_llm_client(model="gpt-4.1")
    try:
        vision_client = get_vision_client(client)
        assert vision_client is not None
        caption = vision_client.describe_image(
            "Name this figure in one sentence.", blocks[0]["data"]
        )
    finally:
        close_llm_client(client)

    assert caption.strip()
    assert "\n" not in caption.strip()


@pytest.mark.usefixtures("_has_env")
@pytest.mark.integration
def test_describe_image_never_returns_an_empty_caption_over_repeated_calls():
    """The regression guard for the transport, which a single call cannot be.

    Captioning over the Responses API returned an *empty* caption for 8 to 12
    of 16 real figure regions against the self-hosted ``qwen3.6-27b`` over five
    runs, a different subset each run: the gateway's bridge filed the answer as
    a reasoning item, which ``output_text`` ignores. A one-shot test passes
    roughly half the time on a broken transport, which is worse than no test.
    Six calls put that at well under 1 in 64.

    **Each call must vary the prompt.** The gateway caches identical payloads
    (verified: repeats return the same completion id plus an
    ``x-litellm-cache-key`` header), so six identical requests are one call and
    five replays -- and a replayed success cannot show a per-call coin flip.

    Runs against whatever this environment has configured, so it covers the
    default model and, where ``DATASHEETINDEX_VISION_MODEL`` is set, that one.
    **It only has teeth against a model the gateway bridges**; on the default
    gpt-4.1 it passes on the reverted code too. The resolved model is therefore
    named in the failure message, so a green run is not mistaken for coverage
    it did not provide. A hardcoded internal alias does not belong here.
    """
    import pymupdf

    from datasheetindex.llm.client import (
        close_llm_client,
        create_llm_client,
        get_vision_client,
    )
    from datasheetindex.llm.figure_captions import CAPTION_SYSTEM_PROMPT
    from datasheetindex.tools.vision import inspect_page

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(80, 80, 520, 500), color=(0, 0, 0), width=2)
    for offset, label in enumerate(("VCC", "GND", "SCL", "SDA")):
        page.insert_text((120, 160 + offset * 60), label, fontsize=22)
    blocks = inspect_page(doc, page=1)
    doc.close()

    client = create_llm_client()
    resolved_model = getattr(client, "_vision_model", None)
    try:
        vision_client = get_vision_client(client)
        assert vision_client is not None
        captions = [
            vision_client.describe_image(
                f"{CAPTION_SYSTEM_PROMPT} (attempt {attempt})", blocks[0]["data"]
            )
            for attempt in range(6)
        ]
    finally:
        close_llm_client(client)

    empty = [i for i, caption in enumerate(captions) if not caption.strip()]
    assert not empty, (
        f"model {resolved_model!r} returned empty captions at attempts {empty} "
        f"of {len(captions)}"
    )
