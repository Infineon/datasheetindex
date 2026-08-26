"""Resolve LLM credentials for the parts of the benchmark that call a model.

Nothing in the offline scoring path imports this. It exists for the
classifier's auto-labelling pass, which is the only Tier-1 component that
talks to a provider.

This replaces the private repository's ``agent.setup_sdk_environment``,
which the benchmark previously borrowed from the internal service. That function also
configured the Claude Agent SDK (stream timeouts, beta-header suppression,
retry counts); none of that applies here, and depending on it dragged the
whole service package -- and ``claude-agent-sdk`` -- into an otherwise
provider-agnostic benchmark.

Credential resolution deliberately accepts three shapes, in order:

1. ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` -- the public Anthropic API
   when the base URL is unset.
2. ``LITELLM_MASTER_KEY`` / ``LITELLM_BASE_URL`` -- a gateway. The published
   runs were produced this way; see docs/reproducing.md for why that matters
   when comparing numbers.
3. A ``.env`` file in the current directory or the benchmark root.

``ANTHROPIC_BASE_URL`` is only exported when a base URL was actually found,
so an unset gateway leaves the Anthropic SDK pointed at its own default.

**That base-URL-optional behaviour is the Claude leg only, and the asymmetry
is worth stating because an earlier version of this docstring did not.**
``datasheet_tools._create_client`` omits ``base_url`` when none was resolved,
so the Claude leg runs against the public API with a key alone.
``openai_path._create_openai_client`` does the opposite: it raises
``ValueError("No gateway base URL...")`` when neither ``LITELLM_BASE_URL``
nor ``ANTHROPIC_BASE_URL`` is set, and it never reads ``OPENAI_API_KEY`` --
that name appears only inside ``gateway/litellm_config.yaml``, i.e. it is
read by LiteLLM, not by us. So the GPT-5.1 leg always needs a base URL, and
reaching the public OpenAI API means pointing the gateway variables at it
(``LITELLM_BASE_URL=https://api.openai.com`` with the OpenAI key in
``LITELLM_MASTER_KEY``; the ``/v1`` suffix is appended for you). The Qwen leg
needs a real gateway either way -- ``extra_body`` -> ``chat_template_kwargs``
has no public-API equivalent. See gateway/README.md.
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
    """Whether to skip TLS verification (a gateway with a self-signed cert).

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
