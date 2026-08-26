# Reference gateway

Every call in the published runs went through a gateway addressed by
`ANTHROPIC_BASE_URL` / `LITELLM_BASE_URL` (see
`src/chamberbench/credentials.py`), and the archive in this repository was
produced against a [LiteLLM](https://github.com/BerriAI/litellm) proxy. This
directory is a config for standing up an equivalent proxy of your own.

That is how the archive was produced, not a hard constraint on all three legs,
and the difference matters if you are re-running one of them. The Claude leg
will talk to the public Anthropic API with a key and no base URL. The GPT leg
always needs a base URL — it refuses to construct a client without one — but
that URL may be OpenAI's own (`LITELLM_BASE_URL=https://api.openai.com`, key in
`LITELLM_MASTER_KEY`). The Qwen leg genuinely cannot skip a proxy: the
`extra_body` surface below has no public-API equivalent.
`litellm_config.yaml` is the whole model-naming fix: it maps the three
aliases the harness asks for -- `claudesonnet4.6`, `gpt-5.1`, `qwen3.6-27b`
-- onto models you can actually reach, so nothing in `src/` needs to change
and the archive's filenames (`latest_chamber.qwen3.6-27b.json`) keep
matching what you run.

## The three surfaces your gateway must expose

| Surface | Used by | Why |
|---|---|---|
| `/v1/messages` (Anthropic message shape) | Claude, Qwen | Both engines drive the vLLM-hosted Qwen model over the same Anthropic-shape passthrough Claude uses -- see `anthropic_path.py`'s `_run_turn` docstring. LiteLLM translates it to the underlying provider. |
| `/v1/responses` (OpenAI Responses API) | GPT | `openai_path.py` calls `client.responses.create(...)`, not Chat Completions -- this is what supplies the reasoning summaries the traces record. |
| `extra_body` -> `chat_template_kwargs` | Qwen only | The only channel that reaches vLLM's `enable_thinking` chat-template kwarg. The Anthropic-native `thinking` parameter does **not** work for the Qwen leg -- LiteLLM mistranslates it into a vLLM `reasoning_effort` field that never triggers the thinking turn. |

## Running the proxy

Pin a specific LiteLLM release rather than `:main` -- `extra_body`
pass-through to a `hosted_vllm/` model is exactly the kind of parameter
LiteLLM's provider-translation layer can regress or drop silently (see
upstream reports such as
[BerriAI/litellm#18039](https://github.com/BerriAI/litellm/issues/18039)),
and there is no error when it does; the run just gets quietly worse (see
"Fidelity" below). Use **LiteLLM >= 1.60.0** as a floor -- this repository
does not record the exact version the published archive's gateway ran (the
same gap `docs/reproducing.md` already discloses for `datasheetindex`), and
`extra_body` pass-through for `hosted_vllm/`-style models has had open
upstream reports even in releases newer than that, so a version number alone
is necessary but not sufficient. Verify `extra_body` actually arrives with
the check in the next section every time you change the LiteLLM version or
point at a new backend, not just once.

```bash
docker run \
  --rm \
  -p 4000:4000 \
  -v "$(pwd)/litellm_config.yaml:/app/config.yaml" \
  -e ANTHROPIC_API_KEY \
  -e OPENAI_API_KEY \
  -e QWEN_API_BASE \
  -e QWEN_API_KEY \
  ghcr.io/berriai/litellm:1.60.0 \
  --config /app/config.yaml
```

Point the harness at it:

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=<your LiteLLM master key or a per-user key>
```

`openai_path.py` reads `LITELLM_BASE_URL` in preference to
`ANTHROPIC_BASE_URL`, and appends `/v1` itself if it is missing -- either
variable pointed at the same proxy works for both engines.

## TLS: verified by default, with two opt-in escape hatches

Every engine verifies the gateway's certificate. If your proxy terminates TLS
with a certificate your trust store does not accept -- a self-signed cert on a
proxy you run yourself is the usual case -- set:

```bash
export DISABLE_TLS_VERIFY=true      # also accepts 1, yes, on
```

`chamberbench.credentials.tls_verify_disabled()` is the single place that
variable is read, and every engine routes through it. Nothing in this
repository sets it for you, deliberately: an unverified connection to an
`https://` gateway sends your API key over a channel an interposer can read,
and that is not a decision code should make on a reader's behalf. Prefer
adding your CA to the trust store; reach for this only for a proxy on
`localhost` whose certificate you issued.

**A self-signed gateway needs a second variable, because a second client
exists.** The `harness` extra installs `datasheetindex`, whose own LLM client
reaches the same gateway on its own -- the ToC fallback fires automatically on
a document whose outline scores poorly -- and it reads `LITELLM_TLS_VERIFY`,
not `DISABLE_TLS_VERIFY`. It verifies by default too, so on a self-signed
gateway set both:

```bash
export LITELLM_TLS_VERIFY=false     # also accepts 0, no, off
```

Without it the engines connect and indexing fails on its own certificate
error. Two variables rather than one is not a design; it is that the library
is a separate package with its own published environment contract, and
quietly overriding a dependency's security default from inside the harness is
the behaviour that made this section wrong in the first place.

One path this repository does **not** verify: `datasheetindex` downloads a
datasheet from a URL through a helper that retries a certificate failure with
verification off, because some vendor sites chain their certificates badly.
It is out of the benchmark's path -- the corpus ships as files -- but read the
heading above as "every engine talking to the gateway", not as a
repository-wide guarantee.

## Fidelity: confirm the flag actually arrived

This is the one setting in this file that is fidelity-critical, not just
convenience. If your gateway drops `extra_body`, the Qwen arm runs with
thinking enabled regardless of the flag. Upstream issue
[QwenLM/Qwen3#1817](https://github.com/QwenLM/Qwen3/issues/1817) associates
that configuration with a roughly 23% engine-error rate against 1.3% with
thinking off, so a silently dropped flag does not fail loudly -- it produces
a materially worse run that still looks like a run.

`drop_params: false` is set in `litellm_config.yaml` for exactly this
reason: a gateway that silently discards a parameter it does not recognise
is the failure this section warns about, and the default (`true`) is what
lets it happen without so much as a log line.

Confirm the flag reached the server before trusting a full run:

1. Start the proxy with `--detailed_debug` (or watch the vLLM server's own
   logs, whichever sits closer to the actual chat template render).
2. Run a single claim with thinking explicitly disabled:

   ```bash
   CHAMBER_QWEN_ENABLE_THINKING=false \
     chamber-run --model qwen3.6-27b --claim-id <one claim id> --out /tmp/gateway-check
   ```

3. Grep the logs for `enable_thinking`. You should see it recorded as
   `false` on the outbound request. If it is missing entirely, or you see it
   as `true` despite the env var, `extra_body` is being dropped somewhere in
   the chain -- check `drop_params`, the LiteLLM version, and (if you are
   behind an additional reverse proxy) that nothing upstream of LiteLLM is
   itself stripping unrecognised body fields.
4. Repeat with the env var unset (or `true`) and confirm the value flips.
   Seeing a *fixed* value regardless of the env var is the actual failure
   mode -- a silently constant `enable_thinking` is worse than a missing
   one, because it means every run so far believed it was toggling a knob
   that never moved.
