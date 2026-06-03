# mima-governance

Attest any AI execution in one line. Push signed execution records to Mima's immutable governance ledger.

## Install

```bash
pip install mima-governance
```

## Quick Start

```python
from mima_governance import MimaGovernance

mima = MimaGovernance(
    workspace_id="your-workspace-id",
    api_key="mima_ext_...",
    system_name="my-ai-pipeline",
)

# Decorator — zero effort
@mima.attest(tool_name="generate_report")
def generate_report(data):
    return call_llm(data)

# That's it. Every call is now attested in the Mima governance ledger.
```

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

```python
from nacl.signing import SigningKey

key = SigningKey.generate()

mima = MimaGovernance(
    workspace_id="...",
    api_key="...",
    system_name="...",
    signing_key=key.encode(),  # 32-byte seed
)
# All attestations now cryptographically signed → trust_tier: "verified"
```

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
