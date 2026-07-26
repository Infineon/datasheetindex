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
