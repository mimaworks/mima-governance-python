# Mima Governance — Hardening Plan

## Problem Statement

Five weaknesses identified in security/product review:

1. **Shallow scan** — tokenizer detects `openai.` but not `from openai import OpenAI; client = OpenAI()`. Misses 60-80% of real-world AI calls that use aliased imports.
2. **Fragile attestation heuristic** — 10-line proximity check breaks when `@mima.attest` is on a function whose AI call is at line 45. Need function-scope awareness.
3. **No runtime enforcement** — scan is point-in-time; if someone removes a decorator, evidence stops flowing silently. Need an opt-in runtime guard.
4. **Abrupt wall at `mima login`** — everything before login is local; everything after requires an account. Transition should be smoother.
5. **Silent drift** — if `mima scan` isn't in CI, removal of decorators goes unnoticed. Need self-healing via pre-commit hook.

---

## Architecture Decision: AST over Tokens

The current scanner uses `tokenize` (Python's lexer). It sees tokens but has no knowledge of:
- Import aliases (`from openai import OpenAI` → `OpenAI` is an AI client)
- Function boundaries (which decorator covers which lines)
- Class instantiation (`client = OpenAI()` → `client.chat.completions.create()` is an AI call)

Solution: add an `ast`-based scanner that runs alongside the tokenizer. The AST scanner handles:
- Import tracking: `from openai import OpenAI` → register `OpenAI` as an AI symbol
- Variable assignment tracking: `client = OpenAI()` → register `client` as an AI handle
- Function-scope attestation: `@mima.attest` → all AI calls within that function body are attested
- Call detection: `client.chat.completions.create()` → detected via registered handle

The token scanner remains as fallback (handles malformed files that AST can't parse).

---

## Task Breakdown

### Phase 1: AST Scanner (fixes weakness 1 + 2)

**Task 1.1: `_scan_file_ast()` — import + alias tracking**
- Parse file with `ast.parse()`
- Walk `Import` / `ImportFrom` nodes
- Build `alias_map: dict[str, str]` mapping local names to known AI libraries
- e.g., `from openai import OpenAI` → `{"OpenAI": "openai"}`
- e.g., `import anthropic as ant` → `{"ant": "anthropic"}`

Acceptance criteria:
- `from openai import OpenAI` → `OpenAI` registered as openai alias
- `import langchain as lc` → `lc` registered as langchain alias
- `from anthropic import Anthropic, AsyncAnthropic` → both registered

**Task 1.2: Variable assignment tracking (AI handle detection)**
- Walk `Assign` nodes
- If RHS is a `Call` whose function is a known alias → register the target as an AI handle
- e.g., `client = OpenAI()` → `{"client": "openai"}`
- e.g., `llm = ChatAnthropic(...)` → `{"llm": "anthropic"}`

Acceptance criteria:
- `client = OpenAI()` → `client` registered as openai handle
- `c = anthropic.Anthropic()` → `c` registered as anthropic handle
- Chained: `x = client.chat` does NOT register `x` (only direct constructors)

**Task 1.3: Call site detection via handles**
- Walk `Call` nodes
- If the call's function is an `Attribute` on a registered handle → detection
- e.g., `client.chat.completions.create()` → detected as openai, high confidence

Acceptance criteria:
- `client.chat.completions.create()` where `client = OpenAI()` → detected
- `ant.messages.create()` where `import anthropic as ant` → detected
- `unrelated.method()` → not detected

**Task 1.4: Function-scope attestation (fixes weakness 2)**
- Walk `FunctionDef` / `AsyncFunctionDef` nodes
- If any decorator matches attest patterns → ALL call sites within that function body are attested
- No more 10-line proximity — uses AST parent scope

Acceptance criteria:
- `@mima.attest()` on a 60-line function → AI call at line 55 is attested
- `@mima.attest()` on function A; AI call in function B (not decorated) → NOT attested
- Nested function: decorator on outer → inner function's calls are also attested

**Task 1.5: Merge AST + token results**
- `_scan_file()` tries AST first; falls back to tokenizer if `ast.parse()` raises `SyntaxError`
- AST detections supersede token detections for the same file (higher confidence)
- `Detection` namedtuple gets new field: `method: str` ("ast" | "token")

Acceptance criteria:
- File with valid Python → AST scanner used, token scanner skipped
- File with syntax error → token scanner used as fallback
- Results include `method` field

---

### Phase 2: Runtime Enforcement (fixes weakness 3)

**Task 2.1: `mima_governance.guard` module — import hook**

An opt-in module that monkey-patches AI library clients at import time to emit a warning
(or raise) if called outside an attested context.

```python
# Enable in your app's entry point:
from mima_governance.guard import enable_guard
enable_guard(mode="warn")  # "warn" | "block" | "report"
```

Implementation:
- Uses `sys.meta_path` or post-import hooks to wrap known AI client classes
- Wraps `__call__` / key methods on `openai.OpenAI`, `anthropic.Anthropic`, etc.
- Each wrapped method checks a thread-local `_attested_context` flag
- `@mima.attest()` sets the flag before calling the function, clears after
- If flag is not set and mode is "warn" → `warnings.warn()`
- If mode is "block" → raise `MimaAttestationError`
- If mode is "report" → silently log to a local file for later `mima scan --runtime`

Acceptance criteria:
- `enable_guard("warn")` + unattested `openai.OpenAI().chat.completions.create()` → UserWarning
- `enable_guard("block")` + unattested call → MimaAttestationError raised
- `@mima.attest()` + guarded call → no warning, executes normally
- Guard doesn't break if openai/anthropic not installed (graceful no-op)
- Performance overhead < 1ms per call (just a thread-local check)

**Task 2.2: Thread-local context in `@mima.attest()`**
- Before calling decorated function: set `_guard_context.attested = True`
- After return: clear flag
- Works correctly with threading (thread-local) and async (contextvars)

Acceptance criteria:
- Two threads: one attested, one not → only the unattested one warns
- Async: `@mima.attest()` on async function → flag set via ContextVar

---

### Phase 3: Progressive Disclosure (fixes weakness 4)

**Task 3.1: `mima push --dry-run` (no credentials needed)**

When run without credentials (or with `--dry-run` flag), `mima push` shows:
- What record would be created
- Which controls it would evidence
- What framework scores would be affected

No API call — uses the local control mapping from `_base.py` to compute the mapping.

```
$ mima push change_event --by ci-bot --description "Deploy" --environment prod --system api --dry-run

  DRY RUN — no credentials, nothing sent

  Record: change_event
  Controls that would be evidenced:
    SOC2_CC6.1   Logical and physical access controls
    SOC2_CC8.1   Change management
    ISO27001_A.12.1  Operational procedures

  To push for real: mima login && mima push change_event ...
```

Acceptance criteria:
- `mima push --dry-run` works without MIMA_API_KEY set
- Shows correct control mapping for all 11 record types
- Shows "To push for real" with next step
- Does NOT make any HTTP calls

**Task 3.2: `mima status --demo` (no credentials needed)**

Shows a mock readiness view using locally available data (scan results + control mappings)
to demonstrate what the dashboard would show if evidence were pushed.

Acceptance criteria:
- `mima status --demo` works without credentials
- Shows "DEMO MODE — connect with `mima login` to see real scores"
- Shows which frameworks are relevant based on `mima scan` results

---

### Phase 4: Self-Healing Drift Detection (fixes weakness 5)

**Task 4.1: `mima init --hook` generates pre-commit hook**

Extends `mima init` to optionally write `.git/hooks/pre-commit` (or append to existing)
that runs `mima scan --strict` before each commit.

```
$ mima init . --hook

  ...existing init output...

  Pre-commit hook installed: .git/hooks/pre-commit
  Every commit will now fail if unattested AI calls are introduced.
  Remove with: rm .git/hooks/pre-commit
```

Acceptance criteria:
- `mima init --hook` writes executable pre-commit hook to `.git/hooks/pre-commit`
- Hook runs `mima scan . --strict` and blocks commit on exit 1
- If hook already exists, appends (doesn't overwrite) with a clear comment marker
- Works without any config file or credentials

**Task 4.2: `mima init --github-action` generates workflow file**

Extends `mima init` to optionally write `.github/workflows/governance.yml` directly.

Acceptance criteria:
- `mima init --github-action` writes a valid workflow YAML
- Workflow runs `mima test tests/test_governance.py` on push
- If file already exists, prints "already exists" and skips

---

## Dependency Graph

```
Phase 1 (AST Scanner)
  1.1 → 1.2 → 1.3 → 1.4 → 1.5
                              ↓
Phase 2 (Runtime Guard)     Phase 3 (Progressive Disclosure)
  2.1 → 2.2                  3.1, 3.2 (independent)
                              ↓
                            Phase 4 (Self-Healing)
                              4.1, 4.2 (independent)
```

Phase 1 is the foundation — all other phases can proceed in parallel after 1.5.
Phase 2, 3, 4 are independent of each other.

---

## Checkpoints

**Checkpoint 1** (after Phase 1): Run `mima scan` on a real project with aliased imports.
Verify detection rate improves from ~40% to >90%. Run full test suite — 140+ tests pass.

**Checkpoint 2** (after Phase 2): Enable guard in a test file with an unattested OpenAI
call. Verify warning fires. Verify attested calls don't fire. Run full test suite.

**Checkpoint 3** (after Phase 3): Run `mima push --dry-run` for all 11 record types
without credentials. Verify each shows correct control mapping.

**Checkpoint 4** (after Phase 4): Run `mima init --hook` in a git repo. Make a commit
with an unattested AI call. Verify commit is blocked.

---

## Not Doing

| Item | Reason |
|---|---|
| Dashboard → CLI webhook notifications | Requires server infra changes; separate workstream |
| Full data-flow analysis (track AI handles through function args) | Diminishing returns; handle tracking at assignment level catches 90% |
| LSP integration (editor warnings) | Phase 2 project; depends on AST scanner being stable |
| `mima check` alias | Cognitive load isn't the bottleneck |
| Auto-fix (insert @mima.attest automatically) | Too risky — changes code semantics |

---

## Estimated effort

| Phase | Tasks | Complexity | Estimate |
|---|---|---|---|
| Phase 1 | 5 tasks | High (AST walking, edge cases) | ~3 hours |
| Phase 2 | 2 tasks | Medium (import hooks, thread-locals) | ~2 hours |
| Phase 3 | 2 tasks | Low (local computation, no new I/O) | ~1 hour |
| Phase 4 | 2 tasks | Low (file generation) | ~1 hour |

**Total: ~7 hours focused implementation.**
