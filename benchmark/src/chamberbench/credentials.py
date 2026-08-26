"""Resolve LLM credentials for the parts of the benchmark that call a model.

Nothing in the offline scoring path imports this. It exists for the
classifier's auto-labelling pass, which is the only Tier-1 component that
talks to a provider.

This replaces ``datasheet_agent.agent.setup_sdk_environment``, which the
benchmark previously borrowed from the internal service. That function also
configured the Claude Agent SDK (stream timeouts, beta-header suppression,
retry counts); none of that applies here, and depending on it dragged the
whole service package -- and ``claude-agent-sdk`` -- into an otherwise
provider-agnostic benchmark.

Credential resolution deliberately accepts three shapes, in order:

1. ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` -- the public API when the
   base URL is unset, which is the path an external reproduction takes.
2. ``LITELLM_MASTER_KEY`` / ``LITELLM_BASE_URL`` -- a gateway. The published
   runs were produced this way; see docs/reproducing.md for why that matters
   when comparing numbers.
3. A ``.env`` file in the current directory or the benchmark root.

``ANTHROPIC_BASE_URL`` is only exported when a base URL was actually found,
so an unset gateway leaves the Anthropic SDK pointed at its own default.
That is what makes the public-API path work with no code change.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_configured = False

# Single source of truth; do not recompute it here.
from chamberbench.claimsio import BENCHMARK_ROOT


def setup_credentials() -> None:
    """Populate ``ANTHROPIC_API_KEY`` (and optionally ``ANTHROPIC_BASE_URL``).

    Raises ``RuntimeError`` if no key can be found, rather than failing later
    inside a provider SDK with a less obvious message.
    """
    global _configured
    if _configured:
        return

    # Load .env UNCONDITIONALLY. Gating this on "no key is set yet" -- which an
    # earlier version did -- means an `ANTHROPIC_API_KEY` exported in the shell
    # suppresses the whole file, including `LITELLM_BASE_URL`, and the client
    # then silently talks to the public API instead of the configured gateway.
    # `load_dotenv` does not override already-set variables, so loading it
    # always is both safe and what the originating project did.
    for env_path in (Path.cwd() / ".env", BENCHMARK_ROOT / ".env"):
        if not env_path.exists():
            continue
        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover - optional convenience
            break
        load_dotenv(env_path)
        logger.info("Loaded environment from %s", env_path)
        break

    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LITELLM_MASTER_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY (public API) or "
            "LITELLM_MASTER_KEY (gateway), or provide a .env file. "
            "The offline scoring path needs no credentials at all -- see "
            "docs/reproducing.md."
        )
    os.environ["ANTHROPIC_API_KEY"] = api_key

    base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("LITELLM_BASE_URL")
    if base_url:
        os.environ["ANTHROPIC_BASE_URL"] = base_url

    _configured = True


def tls_verify_disabled() -> bool:
    """Whether to skip TLS verification (self-signed internal gateway).

    Reads ``DISABLE_TLS_VERIFY``. The replaced internal helper translated that
    into ``NODE_TLS_REJECT_UNAUTHORIZED=0`` -- a *Node.js* variable, inherited
    from a JavaScript agent SDK -- and the classifier still tested for the Node
    name after the translation step was dropped, so the escape hatch silently
    stopped working. One variable, read in one place.
    """
    return os.environ.get("DISABLE_TLS_VERIFY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
