"""Shared test fixtures and helpers for datasheetindex tests."""

import importlib
import os
from collections.abc import Generator
from pathlib import Path

import pymupdf
import pytest

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"


class DummyDoc:
    """Minimal stand-in for pymupdf.Document in unit tests."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeResponse:
    """Fake urllib response for testing URL downloads."""

    def __init__(
        self,
        data: bytes,
        status: int = 200,
        content_type: str = "application/pdf",
    ):
        self._data = data
        self._read_once = False
        self._status = status
        self.headers = {"Content-Type": content_type}

    def getcode(self):
        return self._status

    def geturl(self):
        return "https://example.com/test.pdf"

    def read(self, _size: int = -1):
        if self._read_once:
            return b""
        self._read_once = True
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def sdk_envelope_to_content(envelope: dict) -> list[dict]:
    """Convert a neutral envelope the way ``claude-agent-sdk`` really does.

    A deliberate, key-for-key duplicate of the conversion inside
    ``claude_agent_sdk.create_sdk_mcp_server``'s ``call_tool`` -- notably that it
    reads ``item["mimeType"]`` (camelCase) for images while reading
    ``result["is_error"]`` (snake_case) for the envelope. The SDK's format is
    mixed-case; matching it is the contract, not a style choice.

    This exists because the SDK tests used to stub the conversion out entirely:
    the fake ``create_sdk_mcp_server`` accepted the envelope and never read a
    single key from it, so the envelope was free to spell a key any way it liked
    and every SDK test still passed. That blind spot is how #13 -- inspect_page
    raising ``KeyError('mimeType')`` on every call through the SDK surface --
    survived for two months.

    Keep the subscripts as subscripts. A ``KeyError`` here is the entire point;
    switching any of them to ``.get()`` restores the blind spot this closes.
    ``tests/test_sdk_integration.py`` pins this mirror against the real SDK.
    """
    content: list[dict] = []
    for item in envelope.get("content", []):
        item_type = item.get("type")
        if item_type == "text":
            content.append({"type": "text", "text": item["text"]})
        elif item_type == "image":
            content.append(
                {
                    "type": "image",
                    "data": item["data"],
                    "mimeType": item["mimeType"],
                }
            )
    return content


_LLM_ENV_VARS = (
    "LITELLM_BASE_URL",
    "LITELLM_MASTER_KEY",
    "LITELLM_TLS_VERIFY",
    "LITELLM_TIMEOUT_SECONDS",
    "LITELLM_MAX_RETRIES",
    # Not credentials, but the same hermeticity argument: a developer who has
    # pointed captioning or the ToC fallback at their gateway's own models would
    # otherwise run every test against different models than CI does.
    "DATASHEETINDEX_VISION_MODEL",
    "DATASHEETINDEX_MODEL",
)


@pytest.fixture(autouse=True)
def _hermetic_llm_env(request, monkeypatch):
    """Keep the LLM gateway out of every test that did not explicitly opt in.

    build_datasheet regenerates a low-quality ToC through the LLM whenever
    ambient LITELLM credentials exist: index.build -> _try_create_default_llm_client
    -> create_llm_client, which loads .env via python-dotenv. A developer with
    real credentials in .env therefore gets a fabricated ToC (and the search
    breadcrumbs that follow) where CI, running credential-free, gets none -- so
    test_datasheet_tools_build_and_query_artifacts passes in CI and fails
    locally. That is a non-hermetic test, not a product bug.

    Reproduce CI's credential-free environment for every test, EXCEPT the ones
    that request the credentials through the `_has_env` fixture (the integration
    tests). Both steps are needed: create_llm_client re-reads .env on each call,
    so without neutralising load_dotenv the delenv would be silently undone.
    """
    if "_has_env" in request.fixturenames:
        return
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError:
        return
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)


@pytest.fixture(autouse=True)
def _hermetic_wsl_env(monkeypatch):
    """Keep the developer's own distro out of every test.

    ``index._posix_paths_for_windows`` unwraps a ``\\\\wsl.localhost\\<distro>``
    path only when it names *this* distro, which it learns from
    ``WSL_DISTRO_NAME``. That variable is set inside WSL and absent in CI's
    Linux container, so a test that neither sets nor clears it asserts one
    thing on a maintainer's machine and the opposite in CI -- which is exactly
    how a green local suite shipped a red pipeline once already.

    Cleared here so the ambient value can never leak in; tests that need a
    distro set it explicitly with ``monkeypatch.setenv``.
    """
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)


@pytest.fixture()
def _has_env():
    """Skip if .env credentials are not available."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        pytest.skip(".env file not found")
    try:
        dotenv = importlib.import_module("dotenv")
    except ImportError:
        pytest.skip("python-dotenv not installed")
    dotenv.load_dotenv(env_path)
    try:
        importlib.import_module("httpx")
        importlib.import_module("openai")
    except ImportError:
        pytest.skip("LLM client dependencies not installed")
    if not os.environ.get("LITELLM_BASE_URL") or not os.environ.get(
        "LITELLM_MASTER_KEY"
    ):
        pytest.skip("LiteLLM env vars not set")


def spy_on_toc_fallback(monkeypatch, nodes):
    """Replace ``generate_toc_from_text`` with a spy returning ``nodes``.

    Returns the list of ``total_pages`` values it was called with, so an empty
    list means the LLM ToC fallback never ran. ``index.build`` imports the
    function inside the branch that uses it, so patching the module attribute
    is what a call there resolves to.

    Shared because the ``regenerate_toc`` escalation is tested at two levels --
    ``DatasheetIndex.build`` and ``DatasheetTools.build_datasheet`` -- and the
    two must be spying on the same thing for the pair to mean anything.
    """
    import datasheetindex.llm.toc_fallback as fallback_module

    calls: list[int] = []

    def spy(_text, total_pages, _client):
        calls.append(total_pages)
        return list(nodes)

    monkeypatch.setattr(fallback_module, "generate_toc_from_text", spy)
    return calls


@pytest.fixture
def toc_pdf(tmp_path):
    """A synthetic PDF whose ToC quality clears TOC_FALLBACK_THRESHOLD.

    A PDF with no bookmarks scores 0.00 against a threshold of 0.3, so the
    fallback is eligible, CI has no credentials, no client can be created, and
    the build is marked ``toc_fallback_pending`` -- which makes every reuse-hit
    test unpassable. Two ``set_toc`` entries on three pages score 0.62.

    Shared here rather than owned by one module because two very different
    suites need the same shape. The reuse tests need an artifact that is
    cacheable at all. The ``regenerate_toc`` escalation tests need a document
    where the escalation is the *only* thing that can make the LLM fallback
    eligible: this one is above the threshold and carries no raster figure, so
    ``needs_toc_fallback`` and ``has_caption_candidates`` are both False and
    every LLM branch is unreachable without an explicit request.

    Hermetic by construction -- built with pymupdf into ``tmp_path``, with no
    dependence on a bundled PDF. Those are gitignored, so a test resting on one
    silently skips in the CI clone that gates releases.
    """
    from datasheetindex.core.quality import assess_toc_quality
    from datasheetindex.core.structure import build_tree, extract_toc
    from datasheetindex.index import TOC_FALLBACK_THRESHOLD

    doc = pymupdf.open()
    for _ in range(3):
        page = doc.new_page()
        writer = pymupdf.TextWriter(page.rect)
        # y=400 sits well outside the top/bottom furniture bands (20%/80% of
        # an 842pt page): identical text on every page would otherwise be
        # detected as a running header and stripped from scan_pages' output.
        writer.append((72, 400), "Body text for this page of the datasheet")
        writer.write_text(page)
    doc.set_toc([[1, "Overview", 1], [1, "Electrical Characteristics", 2]])
    pdf_path = tmp_path / "ds.pdf"
    doc.save(str(pdf_path))

    nodes = build_tree(extract_toc(doc), len(doc))
    score = assess_toc_quality(nodes, len(doc)).score
    doc.close()
    assert score >= TOC_FALLBACK_THRESHOLD, (
        f"fixture ToC scores {score}, below the {TOC_FALLBACK_THRESHOLD} "
        "threshold; the LLM fallback would be eligible without an explicit "
        "request, and the escalation tests would pass for the wrong reason"
    )
    return pdf_path


@pytest.fixture(scope="session")
def pdf_tle9350_path() -> Path:
    """Return the path to the TLE9350BSJ PDF, skipping if not found."""
    if not TLE9350_PATH.exists():
        pytest.skip("TLE9350BSJ PDF not found in data2page directory")
    return TLE9350_PATH


@pytest.fixture(scope="session")
def pdf_tle9350(pdf_tle9350_path: Path) -> Generator[pymupdf.Document]:
    """Open the TLE9350BSJ PDF as a pymupdf.Document."""
    doc = pymupdf.open(str(pdf_tle9350_path))
    yield doc
    doc.close()
