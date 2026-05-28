"""Shared test fixtures and helpers for datasheetindex tests."""

import importlib
import os
from collections.abc import Generator
from pathlib import Path

import pymupdf
import pytest

DATA2PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data2page"
TLE9350_PATH = DATA2PAGE_DIR / "Infineon-TLE9350BSJ-DataSheet-v01_00-EN.pdf"
TLE9371_PATH = DATA2PAGE_DIR / "infineon-tle9371vle-datasheet-en.pdf"


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
