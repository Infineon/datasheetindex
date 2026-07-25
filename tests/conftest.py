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
