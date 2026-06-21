# mima-governance

Attest AI executions, push GRC evidence records, and run governance policy tests — one call maps to EU AI Act, ISO 42001, SOC 2, and NIST AI RMF simultaneously.

## Four frameworks, one attestation call

| Framework | What it covers |
|---|---|
| EU AI Act | Art. 9 risk assessments, Art. 13 transparency, Art. 14 human oversight, Art. 15 accuracy |
| ISO 42001 | AI management system controls — A.6.x risk treatment, A.9.x performance evaluation |
| SOC 2 | CC3.x risk assessment, CC5.x control activities, CC7.x change management |
| NIST AI RMF | GOVERN, MAP, MEASURE, MANAGE functions |

One `@mima.attest()` call earns controls across whichever frameworks apply — no per-regulation wiring, no separate pipelines. `human_oversight` earns `EUAIA_ART14`, `EUAIA_ART13`, `ISO42001_A.6.6`, and `NIST_GOV1` in a single write. Your readiness score updates across all four.

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
    api_key="mima_ext_...",
    system_name="my-ai-pipeline",
)
# workspace_id is resolved automatically from the API key.

# Decorator — wraps a function; every call writes a GRC evidence record
# and maps to applicable controls across EU AI Act, ISO 42001, SOC 2, NIST AI RMF
@mima.attest(tool_name="generate_report")
def generate_report(data):
    return call_llm(data)
```

Each `@mima.attest()` call writes a row to `v2.grc_evidence_records` with `source = 'sdk'`. The cross-framework control mapping is automatic — the same record that evidences `EUAIA_ART13` also earns `ISO42001_A.6.2` and the relevant NIST AI RMF function. That compounding is what makes mima different from a per-regulation tool.

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

`mima scan` uses AST-based analysis (with a tokenizer fallback for files that can't be parsed). It correctly detects:

- **Direct usage:** `openai.chat.completions.create()`
- **Aliased imports:** `from openai import OpenAI; client = OpenAI(); client.chat.completions.create()`
- **Constructor-assigned handles:** `client = OpenAI()` → `client.chat.completions.create()`
- **Function-scope attestation:** `@mima.attest()` covers every AI call in the decorated function body, not just the nearest lines

It does **not** detect:

- **Wrapper abstractions:** `my_llm.generate()` where `my_llm` is not a direct AI constructor assignment
- **Runtime-constructed calls** or non-Python code

When `mima scan` reports zero unattested calls, the AST scanner found none in reachable call sites — not that none exist. Use `--strict` as a CI gate; complement with code review for deep wrapper abstractions it cannot reach.

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
