"""mima scan — static analysis tool for unattested AI call sites.

Usage:
    mima scan <path> [--json] [--include PATTERN]

Known limitations (see --help for details):
  - Aliased imports (from openai import OpenAI; client = OpenAI()) are not detected.
  - Class-level @mima decorators do not cover method-level calls inside the class.
  - Indirect calls through wrappers (my_llm.call()) are not detected.
  - The tokenizer sees source tokens only; runtime behaviour is not analysed.
"""

from __future__ import annotations

import json
import sys
import textwrap
import tokenize
from pathlib import Path
from typing import Iterator, List, NamedTuple

# AI library names that are considered AI call sites when used as `name.`.
_AI_LIBRARY_NAMES = frozenset(
    ["openai", "anthropic", "langchain", "llama_index", "autogen", "crewai", "litellm"]
)

# Decorator names that indicate a call site is attested.
_ATTEST_DECORATORS = frozenset(["mima", "client"])


class Detection(NamedTuple):
    file: str
    line: int
    library: str
    attested: bool
    confidence: str  # "high" | "low"


def _scan_file(path: Path) -> Iterator[Detection]:
    """Yield detections for a single Python source file."""
    try:
        tokens = list(tokenize.open(str(path)).read())
    except (OSError, UnicodeDecodeError):
        return

    # Re-tokenize properly.
    try:
        with tokenize.open(str(path)) as fh:
            raw_tokens = list(tokenize.generate_tokens(fh.readline))
    except tokenize.TokenError:
        return

    # Build a list of decorator names seen before each function/class definition.
    # We record line numbers of @mima / @client.attest decorators.
    decorator_lines: set[int] = set()
    for i, tok in enumerate(raw_tokens):
        if tok.type == tokenize.OP and tok.string == "@":
            # Look at the next NAME token.
            for j in range(i + 1, min(i + 4, len(raw_tokens))):
                if raw_tokens[j].type == tokenize.NAME:
                    if raw_tokens[j].string in _ATTEST_DECORATORS:
                        # Record lines from decorator to the next def/class (approx ±5 lines).
                        decorator_lines.add(tok.start[0])
                    break

    # Walk tokens looking for AI library name followed by ".".
    for i, tok in enumerate(raw_tokens):
        if tok.type != tokenize.NAME:
            continue
        if tok.string not in _AI_LIBRARY_NAMES:
            continue

        # Must be followed (possibly after whitespace tokens) by "."
        next_meaningful = None
        for j in range(i + 1, min(i + 4, len(raw_tokens))):
            if raw_tokens[j].type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                                       tokenize.DEDENT, tokenize.COMMENT):
                continue
            next_meaningful = raw_tokens[j]
            break

        if next_meaningful is None or next_meaningful.string != ".":
            # Name appears but not as `name.` — could be a string or comment mention.
            confidence = "low"
        else:
            confidence = "high"

        # Determine whether this call site is within 10 lines of a known attest decorator.
        call_line = tok.start[0]
        attested = any(abs(call_line - dl) <= 10 for dl in decorator_lines)

        yield Detection(
            file=str(path),
            line=call_line,
            library=tok.string,
            attested=attested,
            confidence=confidence,
        )


def _scan_path(root: Path, include: str = "**/*.py") -> List[Detection]:
    """Walk root recursively and collect all detections."""
    if not root.exists():
        print(f"mima scan: path not found: {root}", file=sys.stderr)
        sys.exit(1)

    paths = list(root.rglob(include)) if root.is_dir() else [root]
    detections: List[Detection] = []
    for p in paths:
        if p.suffix == ".py":
            detections.extend(_scan_file(p))
    return detections


def _print_text(detections: List[Detection]) -> None:
    unattested = [d for d in detections if not d.attested and d.confidence == "high"]
    low_conf   = [d for d in detections if not d.attested and d.confidence == "low"]

    if not detections:
        print("mima scan: no AI library call sites found.")
        return

    if unattested:
        print(f"mima scan: {len(unattested)} unattested AI call site(s) found:\n")
        for d in unattested:
            print(f"  {d.file}:{d.line}  [{d.library}]")
        print()

    if low_conf:
        print(
            f"  {len(low_conf)} low-confidence detection(s) (string/comment mentions, "
            "aliased imports — review manually):"
        )
        for d in low_conf:
            print(f"  {d.file}:{d.line}  [{d.library}] (low confidence)")
        print()

    attested_count = sum(1 for d in detections if d.attested)
    if attested_count:
        print(f"  {attested_count} attested call site(s) — covered by @mima decorator.")


def _cmd_scan(args: List[str]) -> None:
    """Handle `mima scan <path> [--json] [--include PATTERN]`."""
    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima scan — detect unattested AI call sites in Python source code

            Usage:
                mima scan <path> [options]

            Options:
                --json              Emit JSON array of detections to stdout
                --include PATTERN   Glob pattern for files (default: **/*.py)
                -h, --help          Show this message

            Known limitations:
                - Aliased imports (from openai import OpenAI; c = OpenAI()) are NOT
                  detected — the tokenizer sees 'OpenAI.' not 'openai.'.
                - Class-level @mima decorators do not cover method calls inside the class.
                - Indirect calls through wrappers (my_llm.call()) are not detected.
                - confidence="low" detections may be string literals or comments.
        """))
        sys.exit(0)

    emit_json = "--json" in args
    include = "**/*.py"

    cleaned: List[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--include" and i + 1 < len(args):
            include = args[i + 1]
            i += 2
        elif args[i] not in ("--json",):
            cleaned.append(args[i])
            i += 1
        else:
            i += 1

    if not cleaned:
        print("mima scan: specify a path to scan, e.g.  mima scan .", file=sys.stderr)
        sys.exit(1)

    root = Path(cleaned[0])
    detections = _scan_path(root, include)

    if emit_json:
        output = [
            {
                "file":       d.file,
                "line":       d.line,
                "library":    d.library,
                "attested":   d.attested,
                "confidence": d.confidence,
            }
            for d in detections
        ]
        print(json.dumps(output, indent=2))
    else:
        _print_text(detections)


def _cmd_login(args: List[str]) -> None:
    """Handle `mima login [--api-key KEY] [--workspace-id ID] [--url URL]`."""
    from . import config

    api_key = None
    workspace_id = None
    base_url = "https://api.mima.ai"

    i = 0
    while i < len(args):
        if args[i] == "--api-key" and i + 1 < len(args):
            api_key = args[i + 1]
            i += 2
        elif args[i] == "--workspace-id" and i + 1 < len(args):
            workspace_id = args[i + 1]
            i += 2
        elif args[i] == "--url" and i + 1 < len(args):
            base_url = args[i + 1]
            i += 2
        elif args[i] in ("-h", "--help"):
            print(textwrap.dedent("""\
                mima login — authenticate with the Mima governance API

                Usage:
                    mima login --api-key <KEY> --workspace-id <ID> [--url <URL>]

                Or run without flags for interactive mode:
                    mima login

                Credentials are stored in ~/.mima/config.json.
            """))
            sys.exit(0)
        else:
            i += 1

    # Interactive mode if flags not provided
    if not api_key:
        try:
            api_key = input("API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
    if not workspace_id:
        try:
            workspace_id = input("Workspace ID: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)

    if not api_key or not workspace_id:
        print("mima login: both API key and workspace ID are required.", file=sys.stderr)
        sys.exit(1)

    # Verify credentials by hitting the readiness endpoint
    import httpx

    url = f"{base_url.rstrip('/')}/api/workspaces/{workspace_id}/governance/grc/readiness"
    try:
        resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10.0)
        if resp.status_code == 401:
            print("mima login: invalid API key (401 Unauthorized).", file=sys.stderr)
            sys.exit(1)
        if resp.status_code == 403:
            print("mima login: access denied for this workspace (403 Forbidden).", file=sys.stderr)
            sys.exit(1)
        if resp.status_code >= 500:
            print(f"mima login: server error ({resp.status_code}). Try again later.", file=sys.stderr)
            sys.exit(1)
    except httpx.ConnectError:
        print(f"mima login: cannot reach {base_url} — check your network or --url flag.", file=sys.stderr)
        sys.exit(1)
    except httpx.TimeoutException:
        print(f"mima login: connection timed out to {base_url}.", file=sys.stderr)
        sys.exit(1)

    config.set_credentials(api_key, workspace_id, base_url)
    print(f"Authenticated. Credentials saved to ~/.mima/config.json")
    print(f"  Workspace: {workspace_id}")
    print(f"  Endpoint:  {base_url}")


def _cmd_status(args: List[str]) -> None:
    """Handle `mima status` — show certification readiness from the API."""
    from . import config

    if args and args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima status — show certification readiness scores

            Usage:
                mima status [--json]

            Requires: `mima login` first (or MIMA_API_KEY + MIMA_WORKSPACE_ID env vars).
        """))
        sys.exit(0)

    import os
    api_key = os.environ.get("MIMA_API_KEY") or config.get_api_key()
    workspace_id = os.environ.get("MIMA_WORKSPACE_ID") or config.get_workspace_id()
    base_url = os.environ.get("MIMA_BASE_URL") or config.get_base_url()

    if not api_key or not workspace_id:
        print("mima status: not logged in. Run `mima login` first.", file=sys.stderr)
        sys.exit(1)

    import httpx

    url = f"{base_url.rstrip('/')}/api/workspaces/{workspace_id}/governance/grc/readiness"
    try:
        resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"mima status: API returned {e.response.status_code}.", file=sys.stderr)
        sys.exit(1)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(f"mima status: cannot reach API — {e}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    emit_json = "--json" in args

    if emit_json:
        print(json.dumps(data, indent=2))
        return

    # Pretty-print readiness dashboard
    print("\nCertification Readiness")
    print("=" * 50)

    fw_labels = {
        "soc2_type2": "SOC 2 Type II",
        "iso_27001":  "ISO 27001:2022",
        "iso_42001":  "ISO 42001",
    }

    for fw in data.get("frameworks", []):
        label = fw_labels.get(fw["framework"], fw["framework"])
        pct = fw["score_pct"]
        covered = fw["controls_covered"]
        required = fw["controls_required"]

        # Progress bar (20 chars wide)
        filled = int(pct / 5)
        bar = "\u2588" * filled + "\u2591" * (20 - filled)

        validated = fw.get("validated_at")
        badge = " [validated]" if validated else ""

        print(f"  {label:<18} {pct:>3}%  {bar}  ({covered}/{required} controls){badge}")

    overall = data.get("overall_pct", 0)
    print(f"\n  Overall: {overall}% (weakest link)")

    # Show warning if any unvalidated
    unvalidated = [f for f in data.get("frameworks", []) if f["controls_required"] > 0 and not f.get("validated_at")]
    if unvalidated:
        print(f"\n  Warning: {len(unvalidated)} framework(s) not yet validated — scores are indicative only.")

    print()


def _cmd_test(args: List[str]) -> None:
    """Handle `mima test <file_or_path>` — run governance policy assertions."""
    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima test — run governance policy assertions (like DeepEval for compliance)

            Usage:
                mima test <file.py>           Run a test file with GovernanceTest classes
                mima test --coverage <path>   Quick coverage check (% of AI calls attested)

            Test file example:
                from mima_governance.testing import GovernanceTest, assert_attested

                class TestMyAgent(GovernanceTest):
                    def test_full_coverage(self):
                        result = self.scan("src/")
                        return assert_attested(result, min_coverage=0.95)

            Exit codes:
                0  — all tests passed
                1  — one or more tests failed
                2  — file not found or import error
        """))
        sys.exit(0)

    # Quick coverage mode
    if args[0] == "--coverage":
        if len(args) < 2:
            print("mima test --coverage: specify a path to scan.", file=sys.stderr)
            sys.exit(1)
        from .testing import ScanResult
        root = Path(args[1])
        import time
        start = time.perf_counter()
        detections = _scan_path(root)
        duration_ms = (time.perf_counter() - start) * 1000
        result = ScanResult(detections=detections, path=args[1], duration_ms=duration_ms)
        print(f"\nAttestation Coverage: {result.coverage:.0%}")
        print(f"  {result.attested} attested / {result.total} total call sites")
        if result.unattested:
            print(f"  {result.unattested} unattested (high confidence)")
        print(f"  Scanned in {duration_ms:.0f}ms")
        print()
        sys.exit(0 if result.coverage >= 1.0 else 1)

    # Run test file
    from .testing import run_test_file, print_suite_result

    test_path = args[0]
    suite = run_test_file(test_path)
    print(f"\nmima test: {test_path}")
    print("-" * 50)
    print_suite_result(suite)

    sys.exit(0 if suite.all_passed else 1)


_COMMANDS = {
    "scan":   _cmd_scan,
    "login":  _cmd_login,
    "status": _cmd_status,
    "test":   _cmd_test,
}


def main() -> None:
    """Entry point for ``mima`` script."""
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima — AI governance CLI

            Commands:
                mima scan <path>       Detect unattested AI call sites
                mima test <file>       Run governance policy assertions
                mima status            Show certification readiness scores
                mima login             Authenticate with the Mima API

            Run `mima <command> --help` for command-specific options.

            Quick start:
                mima login                         # store API credentials
                mima scan .                        # find unattested AI calls
                mima test tests/test_governance.py # run policy tests
                mima status                        # check readiness scores
        """))
        sys.exit(0)

    if args[0] == "--version":
        from . import __version__
        print(f"mima-governance {__version__}")
        sys.exit(0)

    cmd = args[0]
    if cmd not in _COMMANDS:
        print(f"mima: unknown command '{cmd}' — run 'mima --help' for available commands.", file=sys.stderr)
        sys.exit(1)

    _COMMANDS[cmd](args[1:])
