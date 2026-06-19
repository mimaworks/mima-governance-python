# mima-governance

Attest AI executions, push GRC evidence records, and run governance policy tests against your codebase — with or without a Mima account.

## No account needed to start

```bash
pip install mima-governance
mima init .                             # scan codebase, generate tests/test_governance.py
mima test tests/test_governance.py      # run immediately — no API key, no network
```

`mima scan` and `mima test` are fully local. A Mima account unlocks `mima push` (evidence records), `mima status` (readiness scores), and the compliance dashboard.

## Install

```bash
pip install mima-governance
```

## Quick Start — SDK attestation

```python
from mima_governance import MimaGovernance

mima = MimaGovernance(
    workspace_id="your-workspace-id",
    api_key="mima_ext_...",
    system_name="my-ai-pipeline",
)

# Decorator — wraps a function; every call writes a GRC evidence record
@mima.attest(tool_name="generate_report")
def generate_report(data):
    return call_llm(data)
```

Each `@mima.attest()` call writes a row to `v2.grc_evidence_records` with `source = 'sdk'`. These records map automatically to SOC 2, ISO 27001:2022, ISO 42001, EU AI Act, and NIST AI RMF controls — that cross-framework mapping is what evidences the compliance dashboard.

## Framework Integrations

### LangChain

```python
from mima_governance.integrations import MimaLangChainCallback

chain = my_chain.with_config(callbacks=[MimaLangChainCallback(mima)])
# Every LLM call, tool invocation, and chain step is auto-attested
```

### LlamaIndex

```python
from mima_governance.integrations import MimaLlamaIndexHandler
import llama_index.core

llama_index.core.global_handler = MimaLlamaIndexHandler(mima)
```

## Sync vs Batch

```python
# Sync (default) — immediate push, blocks ~50ms
@mima.attest(tool_name="credit_decision")
def decide(app): ...

# Batch — buffered, flushed every 30s or 100 items
@mima.attest(tool_name="classify_email", mode="batch")
def classify(email): ...
```

## Ed25519 Signing

Records are stored as append-only rows in Postgres. Workspace admins can purge records via the dashboard. To detect deletion or tampering in a signed chain, use Ed25519 signing:

```python
from nacl.signing import SigningKey

key = SigningKey.generate()

mima = MimaGovernance(
    workspace_id="...",
    api_key="...",
    system_name="...",
    signing_key=key.encode(),  # 32-byte seed
)
# Attestations are cryptographically signed → trust_tier: "verified"
# A deleted or modified record breaks the chain and is detectable
```

Keep the private key outside the Mima account (local HSM or secrets manager). The signature is stored alongside the record; Mima cannot forge or reconstruct it.

## Delegation Chain

```python
from mima_governance import MimaGovernance, AuthorisedBy

mima = MimaGovernance(
    ...,
    authorised_by=AuthorisedBy(
        identity="analyst@corp.com",
        role="credit-analyst",
        session_id="sso_abc123",
    ),
)
# Every attestation records WHO authorised the agent to act
```

## How inferred evidence works

The Mima platform runs a nightly job (03:45 UTC) that reads AI system classifications already in the estate — from cloud integrations and CMDB data you have already connected — and converts them into evidence records with `source = 'estate_auto'`. No network scanning. No discovery beyond what's already in your connected estate.

**What inferred evidence is:** the AI risk classification process (tier determination, prohibited-use check) is itself a real risk assessment. It honestly evidences `EUAIA_ART9`, `EUAIA_ART11`, and related controls.

**What it is not:** proof that anyone evaluated model accuracy, governed training data, or operated a human oversight mechanism. The bridge explicitly does not generate `model_evaluation`, `training_data_governance`, or `human_oversight` records — auto-generating those would produce false evidence.

Inferred records are marked "indicative only" in the dashboard until a workspace admin validates the control list. SDK-attested records (`source = 'sdk'`) carry higher weight and are required for formal audit submissions.

## Scan limitations

`mima scan` uses static token analysis. It detects AI library calls of the form `openai.`, `anthropic.`, `langchain.`, etc. It does **not** detect:

- **Aliased imports:** `from openai import OpenAI; client = OpenAI()` — the token `openai.` never appears
- **Wrapper abstractions:** `my_llm.generate()` — the library name is not visible at the call site
- **Runtime-constructed calls** or non-Python code

When `mima scan` reports zero unattested calls, it means the tokeniser found none — not that none exist. Use `--strict` as a CI gate for what the scanner can see; complement it with code review for abstractions it cannot reach.

## Readiness score — how it's calculated

`overall_pct` is the **minimum** across all frameworks with defined controls (weakest-link). If SOC 2 is at 80% and EU AI Act is at 30%, `overall_pct` is 30. A certification chain is only as strong as its weakest framework; averaging would overstate readiness.

Per-framework `score_pct` = `controls_covered / controls_required × 100`.

## Credential storage

`mima login` saves your API key to `~/.mima/config.json` with `0o600` permissions (owner read/write only). The file is plaintext — keep your home directory encrypted (FileVault on macOS, LUKS on Linux) if this is a shared or managed machine.

For CI/CD, use environment variables instead of the config file:

```bash
export MIMA_API_KEY=mima_ext_...
export MIMA_WORKSPACE_ID=ws-...
mima push change_event \
  --by ci-bot@company.com \
  --description "Deploy v1.2.3" \
  --environment production \
  --system api-service \
  --no-delta   # skip readiness fetch in CI
```
