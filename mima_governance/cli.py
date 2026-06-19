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

# Exact decorator names (without a dot suffix) that indicate attestation.
# "client" was intentionally removed — it is far too common (@client.get in Flask,
# @client.event in Discord bots, etc.) and caused false negatives where real
# unattested AI calls hid behind framework decorators.
# The .attest suffix check below (Pattern 2) covers @anything.attest without
# needing to enumerate variable names.
_ATTEST_EXACT_NAMES = frozenset(["mima", "mima_client", "mima_governance"])


class Detection(NamedTuple):
    file: str
    line: int
    library: str
    attested: bool
    confidence: str  # "high" | "low"


def _scan_file(path: Path) -> Iterator[Detection]:
    """Yield detections for a single Python source file."""
    # Single tokenize pass — no dead pre-read, no leaked file handle.
    try:
        with tokenize.open(str(path)) as fh:
            raw_tokens = list(tokenize.generate_tokens(fh.readline))
    except (OSError, UnicodeDecodeError, tokenize.TokenError):
        return

    # Build a set of line numbers where a Mima attest decorator appears.
    # Two recognition patterns:
    #   Pattern 1 — exact variable name:  @mima, @mima_client, @mima_governance
    #   Pattern 2 — .attest suffix:       @anything.attest(...)
    # Pattern 2 covers @client.attest, @gov.attest etc. without broad false
    # negatives from common names like @client (Flask, Discord, websockets).
    _skip = frozenset([
        tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
        tokenize.DEDENT, tokenize.COMMENT,
    ])

    def _meaningful_after(start: int, n: int = 6) -> List:
        return [t for t in raw_tokens[start:start + n] if t.type not in _skip]

    decorator_lines: set[int] = set()
    for i, tok in enumerate(raw_tokens):
        if tok.type != tokenize.OP or tok.string != "@":
            continue
        following = _meaningful_after(i + 1)
        if not following or following[0].type != tokenize.NAME:
            continue
        first_name = following[0]
        # Pattern 1: exact known name
        if first_name.string in _ATTEST_EXACT_NAMES:
            decorator_lines.add(tok.start[0])
        # Pattern 2: name.attest — the `.attest` suffix is specific enough
        elif (
            len(following) >= 3
            and following[1].type == tokenize.OP and following[1].string == "."
            and following[2].type == tokenize.NAME and following[2].string == "attest"
        ):
            decorator_lines.add(tok.start[0])

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


_DEFAULT_EXCLUDE_PATTERNS = frozenset([
    "*/mima_governance/*",
    "*/.venv/*",
    "*/venv/*",
    "*/node_modules/*",
    "*/__pycache__/*",
    "*/.git/*",
])


def _is_excluded(path: Path, exclude_patterns: frozenset) -> bool:
    """Check if a path matches any exclude pattern."""
    path_str = str(path)
    for pattern in exclude_patterns:
        # Simple glob-like matching: * matches any segment
        import fnmatch
        if fnmatch.fnmatch(path_str, pattern):
            return True
    return False


def _scan_path(
    root: Path,
    include: str = "**/*.py",
    exclude: "frozenset | None" = None,
    verbose: bool = False,
) -> "tuple[List[Detection], int]":
    """Walk root recursively and collect all detections.

    Returns (detections, files_scanned).
    """
    if not root.exists():
        print(f"mima scan: path not found: {root}", file=sys.stderr)
        sys.exit(1)

    exclude_patterns = exclude if exclude is not None else _DEFAULT_EXCLUDE_PATTERNS

    all_paths = list(root.rglob(include)) if root.is_dir() else [root]
    eligible = [p for p in all_paths
                if p.suffix == ".py" and not _is_excluded(p, exclude_patterns)]

    if verbose and len(eligible) > 20:
        print(f"  Scanning {len(eligible):,} Python files...", end=" ", flush=True)

    detections: List[Detection] = []
    for p in eligible:
        detections.extend(_scan_file(p))

    if verbose and len(eligible) > 20:
        print("done")

    return detections, len(eligible)


def _print_text(
    detections: List[Detection],
    files_scanned: int = 0,
    duration_ms: float = 0.0,
) -> None:
    unattested     = [d for d in detections if not d.attested and d.confidence == "high"]
    low_conf       = [d for d in detections if not d.attested and d.confidence == "low"]
    attested_count = sum(1 for d in detections if d.attested)
    total_high     = len(unattested) + attested_count

    timing = f"  ({duration_ms:.0f}ms)" if duration_ms else ""

    if not detections:
        print(f"\n  No AI library call sites found in {files_scanned:,} files.{timing}\n")
        return

    # ── Summary line ──────────────────────────────────────────────────────────
    coverage_pct = int(attested_count / total_high * 100) if total_high else 0
    parts = []
    if unattested:
        parts.append(f"{len(unattested)} unattested")
    if attested_count:
        parts.append(f"{attested_count} attested")
    if low_conf:
        parts.append(f"{len(low_conf)} low-confidence")
    summary = "  " + "  ·  ".join(parts)
    if total_high:
        summary += f"  ·  {coverage_pct}% coverage"
    print(f"\n{summary}{timing}\n")

    # ── Unattested (high confidence) ──────────────────────────────────────────
    if unattested:
        print("  UNATTESTED — add @mima.attest() decorator:")
        for d in unattested:
            relpath = d.file
            print(f"    {relpath}:{d.line:<6}  {d.library}")
        print()

    # ── Low confidence ────────────────────────────────────────────────────────
    if low_conf:
        print("  LOW CONFIDENCE — review manually (aliased import or string/comment mention):")
        for d in low_conf:
            print(f"    {d.file}:{d.line:<6}  {d.library}")
        print()

    # ── Attested summary ──────────────────────────────────────────────────────
    if attested_count:
        print(f"  {attested_count} attested call site(s) covered by @mima.attest().\n")

    # ── Fix hint (only when there are unattested calls) ───────────────────────
    if unattested:
        lib = unattested[0].library
        print("  How to fix — wrap the call with @mima.attest():\n")
        print("    from mima_governance import MimaGovernance")
        print("    mima = MimaGovernance(workspace_id=\"...\", api_key=\"...\")\n")
        print("    @mima.attest(tool_name=\"describe_this_call\")")
        print("    def your_function(...):")
        print(f"        return {lib}.your_call(...)  # ← this call is now evidenced\n")
        print("  Use --strict to exit 1 on unattested findings (CI/CD gate).")
        print("  Run `mima status` to see how attestation affects compliance scores.\n")


def _cmd_scan(args: List[str]) -> None:
    """Handle `mima scan <path> [--json] [--strict] [--include PATTERN] [--exclude PATTERN]`."""
    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima scan — detect unattested AI call sites in Python source code

            Usage:
                mima scan <path> [options]

            Options:
                --strict            Exit 1 if any unattested high-confidence call sites
                                    are found. Use this as a CI/CD gate.
                --json              Emit JSON array of detections to stdout
                --include PATTERN   Glob pattern for files (default: **/*.py)
                --exclude PATTERN   Glob pattern to exclude (repeatable)
                --no-default-excludes  Disable built-in excludes (mima_governance/,
                                       .venv/, node_modules/, __pycache__/, .git/)
                -h, --help          Show this message

            Known limitations:
                - Aliased imports (from openai import OpenAI; c = OpenAI()) are NOT
                  detected — the tokenizer sees 'OpenAI.' not 'openai.'.
                - Class-level @mima decorators do not cover method calls inside the class.
                - Indirect calls through wrappers (my_llm.call()) are not detected.
                - confidence="low" detections may be string literals or comments.
        """))
        sys.exit(0)

    emit_json           = "--json" in args
    no_default_excludes = "--no-default-excludes" in args
    strict              = "--strict" in args
    include             = "**/*.py"
    extra_excludes: List[str] = []

    cleaned: List[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--include" and i + 1 < len(args):
            include = args[i + 1]
            i += 2
        elif args[i] == "--exclude" and i + 1 < len(args):
            extra_excludes.append(args[i + 1])
            i += 2
        elif args[i] not in ("--json", "--no-default-excludes", "--strict"):
            cleaned.append(args[i])
            i += 1
        else:
            i += 1

    if not cleaned:
        print("mima scan: specify a path to scan, e.g.  mima scan .", file=sys.stderr)
        sys.exit(1)

    root = Path(cleaned[0])
    if no_default_excludes:
        exclude = frozenset(extra_excludes) if extra_excludes else frozenset()
    else:
        exclude = _DEFAULT_EXCLUDE_PATTERNS | frozenset(extra_excludes)

    import time
    t0 = time.perf_counter()
    detections, files_scanned = _scan_path(
        root, include, exclude=exclude, verbose=not emit_json
    )
    duration_ms = (time.perf_counter() - t0) * 1000

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
        _print_text(detections, files_scanned=files_scanned, duration_ms=duration_ms)

    if strict:
        unattested_count = sum(1 for d in detections if not d.attested and d.confidence == "high")
        if unattested_count:
            sys.exit(1)


def _cmd_login(args: List[str]) -> None:
    """Handle `mima login [--api-key KEY] [--workspace-id ID] [--url URL]`."""
    from . import config

    api_key      = None
    workspace_id = None
    base_url     = "https://api.mima.ai"

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

    # Interactive mode — use getpass for the API key to keep it out of terminal
    # recordings and shell history.
    import getpass
    print()
    if not api_key:
        print("  Find your API key: dashboard \u2192 Settings \u2192 API Keys\n")
        try:
            api_key = getpass.getpass("  API key (mima_ext_...): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)
    if not workspace_id:
        try:
            workspace_id = input("  Workspace ID:           ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(1)

    if not api_key or not workspace_id:
        print("\nmima login: both API key and workspace ID are required.", file=sys.stderr)
        sys.exit(1)

    print("\n  Verifying credentials...", end=" ", flush=True)

    import httpx

    url = f"{base_url.rstrip('/')}/api/workspaces/{workspace_id}/governance/grc/readiness"
    resp = None
    try:
        resp = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10.0)
        if resp.status_code == 401:
            print("failed")
            print("mima login: invalid API key (401 Unauthorized).", file=sys.stderr)
            sys.exit(1)
        if resp.status_code == 403:
            print("failed")
            print("mima login: access denied for this workspace (403 Forbidden).", file=sys.stderr)
            sys.exit(1)
        if resp.status_code == 404:
            print("failed")
            print("mima login: workspace not found — check your workspace ID.", file=sys.stderr)
            sys.exit(1)
        if resp.status_code >= 500:
            print("failed")
            print(f"mima login: server error ({resp.status_code}). Try again later.", file=sys.stderr)
            sys.exit(1)
    except httpx.ConnectError:
        print("failed")
        print(f"mima login: cannot reach {base_url} — check your network or --url flag.", file=sys.stderr)
        sys.exit(1)
    except httpx.TimeoutException:
        print("failed")
        print(f"mima login: connection timed out to {base_url}.", file=sys.stderr)
        sys.exit(1)

    print("\u2713")

    config.set_credentials(api_key, workspace_id, base_url)
    print(f"\n  Workspace:  {workspace_id}")
    print(f"  Endpoint:   {base_url}")
    print(f"  Saved to:   ~/.mima/config.json")

    # Show current posture from the readiness response already fetched — free data.
    if resp is not None and resp.status_code == 200:
        fw_labels = {
            "soc2_type2": "SOC 2 Type II",
            "iso_27001":  "ISO 27001:2022",
            "iso_42001":  "ISO 42001",
            "eu_ai_act":  "EU AI Act",
            "nist_airf":  "NIST AI RMF",
        }
        readiness = resp.json()
        active = [fw for fw in readiness.get("frameworks", []) if fw["controls_required"] > 0]
        if active:
            print("\n  Current posture:")
            for fw in active[:4]:
                label    = fw_labels.get(fw["framework"], fw["framework"])
                pct      = fw["score_pct"]
                attested = fw.get("controls_covered_attested", fw["controls_covered"])
                inferred = fw["controls_covered"] - attested
                filled   = int(pct / 5)
                bar      = "\u2588" * filled + "\u2591" * (20 - filled)
                source   = f"{attested} attested"
                if inferred > 0:
                    source += f"  \u00b7  {inferred} inferred"
                print(f"    {label:<18} {pct:>3}%  {bar}  {source}")
            overall = readiness.get("overall_pct", 0)
            print(f"\n    Overall: {overall}%")

    print("\n  Run `mima scan .` to find unattested AI call sites.\n")


def _compute_quickest_wins(
    frameworks_data: list,
    readiness_data: list,
    max_wins: int = 3,
) -> list:
    """Return top record_types to push, ranked by potential control coverage gain.

    Only considers frameworks that currently have a gap. Returns a list of dicts
    with keys: record_type, controls (count), frameworks (list of display names).
    """
    gap_slugs = {
        fw["framework"]
        for fw in readiness_data
        if fw["controls_required"] > fw["controls_covered"]
    }
    if not gap_slugs:
        return []

    rt_coverage: dict = {}  # record_type → {"controls": int, "frameworks": set}
    for fw_detail in frameworks_data:
        slug = fw_detail.get("framework", "")
        if slug not in gap_slugs:
            continue
        for ctrl in fw_detail.get("controls", []):
            for rt in ctrl.get("evidence_record_types", []):
                if rt not in rt_coverage:
                    rt_coverage[rt] = {"controls": 0, "frameworks": set()}
                rt_coverage[rt]["controls"] += 1
                rt_coverage[rt]["frameworks"].add(slug)

    fw_labels = {
        "soc2_type2": "SOC 2",
        "iso_27001":  "ISO 27001",
        "iso_42001":  "ISO 42001",
        "eu_ai_act":  "EU AI Act",
        "nist_airf":  "NIST AI RMF",
    }
    ranked = sorted(rt_coverage.items(), key=lambda x: x[1]["controls"], reverse=True)
    wins = []
    for rt, info in ranked[:max_wins]:
        fw_names = sorted(fw_labels.get(f, f) for f in info["frameworks"])
        wins.append({"record_type": rt, "controls": info["controls"], "frameworks": fw_names})
    return wins


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
    api_key      = os.environ.get("MIMA_API_KEY")      or config.get_api_key()
    workspace_id = os.environ.get("MIMA_WORKSPACE_ID") or config.get_workspace_id()
    base_url     = os.environ.get("MIMA_BASE_URL")     or config.get_base_url()

    if not api_key or not workspace_id:
        print("mima status: not logged in. Run `mima login` first.", file=sys.stderr)
        sys.exit(1)

    emit_json = "--json" in args

    import httpx
    base    = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = httpx.get(
            f"{base}/api/workspaces/{workspace_id}/governance/grc/readiness",
            headers=headers, timeout=10.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"mima status: API returned {e.response.status_code}.", file=sys.stderr)
        sys.exit(1)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(f"mima status: cannot reach API — {e}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()

    if emit_json:
        print(json.dumps(data, indent=2))
        return

    # Frameworks detail for quickest-wins — best-effort, don't fail status on error.
    frameworks_data: list = []
    try:
        fw_resp = httpx.get(
            f"{base}/api/workspaces/{workspace_id}/governance/grc/frameworks",
            headers=headers, timeout=10.0,
        )
        if fw_resp.status_code == 200:
            frameworks_data = fw_resp.json().get("frameworks", [])
    except Exception:
        pass

    fw_labels = {
        "soc2_type2": "SOC 2 Type II",
        "iso_27001":  "ISO 27001:2022",
        "iso_42001":  "ISO 42001",
        "eu_ai_act":  "EU AI Act",
        "nist_airf":  "NIST AI RMF",
    }

    short_id = workspace_id[:8] + "…" if len(workspace_id) > 8 else workspace_id
    print(f"\n  Certification Readiness  ·  workspace: {short_id}\n")

    has_inferred = False
    unvalidated_labels: List[str] = []

    for fw in data.get("frameworks", []):
        label    = fw_labels.get(fw["framework"], fw["framework"])
        pct      = fw["score_pct"]
        required = fw["controls_required"]
        attested = fw.get("controls_covered_attested", fw["controls_covered"])
        inferred = fw["controls_covered"] - attested

        if inferred > 0:
            has_inferred = True

        filled = int(pct / 5)
        bar    = "\u2588" * filled + "\u2591" * (20 - filled)

        if inferred > 0:
            source_str = f"{attested} attested  \u00b7  {inferred} inferred"
        elif attested > 0:
            source_str = f"{attested} attested"
        elif required == 0:
            source_str = "no controls defined"
        else:
            source_str = "0 attested"

        if fw.get("validated_at"):
            badge = "  \u2713 validated"
        elif required > 0:
            badge = "  \u26a0 not validated"
            unvalidated_labels.append(label)
        else:
            badge = ""

        print(f"  {label:<18} {pct:>3}%  {bar}  {source_str}{badge}")

    overall = data.get("overall_pct", 0)
    active_fws = [fw for fw in data.get("frameworks", []) if fw["controls_required"] > 0]
    weakest = min(active_fws, key=lambda f: f["score_pct"], default=None)
    weakest_label = fw_labels.get(weakest["framework"], weakest["framework"]) if weakest else ""
    print(f"\n  Overall: {overall}%  (weakest link: {weakest_label})")

    if unvalidated_labels:
        names = ", ".join(unvalidated_labels)
        print(f"\n  \u26a0 Not yet validated: {names}")
        print("    Scores are indicative only — validate via the dashboard before audit use.")

    if has_inferred:
        print()
        print("  Inferred controls are covered by Mima\u2019s estate inference (source: estate_auto).")
        print("  They satisfy the awareness gate but not external certification.")
        print("  Replace with SDK calls to upgrade to certified attestation.")

    # ── Quickest wins ─────────────────────────────────────────────────────────
    if frameworks_data:
        wins = _compute_quickest_wins(frameworks_data, data.get("frameworks", []))
        if wins:
            print("\n  Quickest wins:")
            for w in wins:
                fws = "  ".join(w["frameworks"])
                print(f"    +{w['controls']:>2} controls  mima push {w['record_type']:<28}  {fws}")

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
        detections, _files_scanned = _scan_path(root)
        duration_ms = (time.perf_counter() - start) * 1000
        result = ScanResult(detections=detections, path=args[1], duration_ms=duration_ms)
        attested_of = f"({result.attested}/{result.attested + result.unattested} attested)"
        print(f"\n  Attestation Coverage: {result.coverage:.0%}  {attested_of}")
        unattested_dets = [d for d in detections if not d.attested and d.confidence == "high"]
        for d in unattested_dets:
            print(f"  Unattested: {d.file}:{d.line}  [{d.library}]")
        print(f"  Scanned in {duration_ms:.0f}ms")
        exit_code = 0 if result.coverage >= 1.0 else 1
        print(f"  \u2192 exit {exit_code}\n")
        sys.exit(exit_code)

    # Run test file
    from .testing import run_test_file, print_suite_result

    test_path = args[0]
    suite = run_test_file(test_path)
    print(f"\n  mima test  \u00b7  {test_path}")
    print_suite_result(suite)
    exit_code = 0 if suite.all_passed else 1
    print(f"  \u2192 exit {exit_code}\n")
    sys.exit(exit_code)


def _print_push_delta(
    record_type: str,
    push_data: dict,
    before: dict,
    after: dict,
) -> None:
    """Print push result with per-framework readiness score delta."""
    record_id = push_data.get("record_id", "?")
    controls  = push_data.get("mapped_controls", [])

    fw_labels = {
        "soc2_type2": "SOC 2 Type II",
        "iso_27001":  "ISO 27001:2022",
        "iso_42001":  "ISO 42001",
        "eu_ai_act":  "EU AI Act",
        "nist_airf":  "NIST AI RMF",
    }

    print(f"\n  {record_type}  \u00b7  record saved")
    if controls:
        print(f"  Controls evidenced: {', '.join(controls)}")
    print(f"  Record ID: {record_id}")

    before_map = {fw["framework"]: fw for fw in before.get("frameworks", [])}
    after_map  = {fw["framework"]: fw for fw in after.get("frameworks", [])}

    changed = []
    for slug, fw_after in after_map.items():
        fw_before = before_map.get(slug)
        if fw_before and fw_after["score_pct"] != fw_before["score_pct"]:
            ctrl_delta = fw_after["controls_covered"] - fw_before["controls_covered"]
            changed.append((slug, fw_before["score_pct"], fw_after["score_pct"], ctrl_delta))

    if changed:
        print("\n  Readiness change:")
        for slug, pct_before, pct_after, ctrl_delta in changed:
            label     = fw_labels.get(slug, slug)
            delta     = pct_after - pct_before
            sign      = "+" if delta > 0 else ""
            ctrl_note = (
                f"  (+{ctrl_delta} control{'s' if ctrl_delta != 1 else ''})"
                if ctrl_delta > 0 else ""
            )
            print(f"    {label:<18}  {pct_before}% \u2192 {pct_after}%  ({sign}{delta}%){ctrl_note}")
        overall_before = before.get("overall_pct", 0)
        overall_after  = after.get("overall_pct", 0)
        if overall_after != overall_before:
            d = overall_after - overall_before
            print(
                f"\n    Overall: {overall_before}% \u2192 {overall_after}%"
                f"  ({'+' if d > 0 else ''}{d}%)"
            )
    else:
        overall = after.get("overall_pct", 0)
        print(f"\n  No score change yet  \u00b7  Overall: {overall}%")
        print("  Push more records to evidence additional controls.")

    print()


def _cmd_push(args: List[str]) -> None:
    """Handle `mima push <record_type> [field=value ...] [--stdin] [--json]`.

    Pushes a single GRC evidence record to the Mima API from the terminal
    or a CI/CD pipeline step.

    Usage:
        mima push change_event \\
            --by "ci-bot@company.com" \\
            --description "Deploy v1.2.3 to production" \\
            --environment production \\
            --system api-service \\
            --change-id "JIRA-1234"

        echo '{"record_type":"change_event","by":"ci-bot",...}' | mima push --stdin

    Supported record types and their required fields:

        access_review    --user EMAIL --resource NAME --granted true|false
                         --reviewed-by EMAIL [--review-type periodic|triggered|offboarding]

        change_event     --by EMAIL --description TEXT --environment ENV --system NAME
                         [--change-id ID]

        vendor_risk      --vendor NAME --tier critical|high|medium|low
                         --last-reviewed YYYY-MM-DD [--findings N]

        policy_acknowledged  --policy NAME --user EMAIL --version VER
                             [--channel in-app|email|slack]

        incident_report  --title TEXT --severity critical|high|medium|low
                         --description TEXT --affected-systems SYS1,SYS2
                         [--detected-at ISO8601] [--authority-notified-at ISO8601]

        ai_risk_assessment  --system NAME --risk-tier unacceptable|high|limited|minimal
                            --use-case TEXT --impact-domains DOM1,DOM2
                            --art5-self-assessment true|false --assessor EMAIL
                            [--assessment-date ISO8601] [--technical-doc-url URL]

        training_data_governance  --model-id ID --dataset-id ID --record-count N
                                  --bias-checks-performed true|false --approved-by EMAIL
                                  --data-sources SRC1,SRC2 --data-categories CAT1,CAT2
                                  [--approval-date ISO8601] [--known-limitations TEXT]

        model_evaluation  --model-id ID --dataset ID --accuracy FLOAT
                          --evaluated-by EMAIL
                          [--evaluation-type initial|quarterly|triggered]
                          [--passed-threshold true|false] [--evaluation-date ISO8601]

        human_oversight   --decision-id ID --ai-recommendation TEXT
                          --human-decision TEXT --reviewer EMAIL
                          [--rationale TEXT] [--model-id ID]

        model_drift_event  --model-id ID --metric NAME --baseline FLOAT
                           --current FLOAT --threshold FLOAT --detected-by EMAIL
                           [--drift-type performance|data|concept]
                           [--action-taken TEXT] [--detection-date ISO8601]

        governance_review  --reviewed-by IDENTITY --report-type TYPE
                           --frameworks FW1,FW2 --overall-readiness 0-100
                           [--action-items N] [--review-date ISO8601]
    """
    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent(_cmd_push.__doc__ or ""))
        sys.exit(0)

    from . import config

    import os
    api_key      = os.environ.get("MIMA_API_KEY")      or config.get_api_key()
    workspace_id = os.environ.get("MIMA_WORKSPACE_ID") or config.get_workspace_id()
    base_url     = os.environ.get("MIMA_BASE_URL")     or config.get_base_url()
    system_name  = os.environ.get("MIMA_SYSTEM_NAME",  "mima-cli")

    if not api_key or not workspace_id:
        print("mima push: not logged in. Run `mima login` first.", file=sys.stderr)
        sys.exit(1)

    emit_json  = "--json" in args
    use_stdin  = "--stdin" in args
    show_delta = not emit_json and "--no-delta" not in args
    clean_args = [a for a in args if a not in ("--json", "--stdin", "--no-delta")]

    # ── stdin JSON mode ──────────────────────────────────────────────────────
    if use_stdin:
        try:
            raw = sys.stdin.read()
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"mima push: invalid JSON on stdin — {e}", file=sys.stderr)
            sys.exit(1)
        record_type = payload.get("record_type")
        if not record_type:
            print("mima push: JSON must include \"record_type\".", file=sys.stderr)
            sys.exit(1)
    else:
        # ── positional record_type + flag parsing ────────────────────────────
        if not clean_args:
            print("mima push: specify a record_type or use --stdin. Run 'mima push --help'.",
                  file=sys.stderr)
            sys.exit(1)

        record_type = clean_args[0]
        valid_types = (
            "access_review", "change_event", "vendor_risk",
            "policy_acknowledged", "incident_report",
            "ai_risk_assessment", "training_data_governance", "model_evaluation",
            "human_oversight", "model_drift_event", "governance_review",
        )
        if record_type not in valid_types:
            print(
                f"mima push: unknown record_type '{record_type}'.\n"
                f"  Valid types: {', '.join(valid_types)}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Parse --key value flags from remaining args.
        # Detect trailing flags (no value) and error immediately rather than
        # silently dropping them — a flag at end-of-args or followed by another
        # flag is almost certainly a typo.
        flags: dict = {}
        i = 1
        while i < len(clean_args):
            arg = clean_args[i]
            if arg.startswith("--"):
                key = arg[2:].replace("-", "_")  # --reviewed-by → reviewed_by
                if i + 1 >= len(clean_args) or clean_args[i + 1].startswith("--"):
                    print(
                        f"mima push: flag {arg} requires a value but none was provided.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                flags[key] = clean_args[i + 1]
                i += 2
            else:
                i += 1

        payload = _build_push_payload(record_type, flags)
        if payload is None:
            sys.exit(1)
        payload["record_type"] = record_type

    payload.setdefault("system_name", system_name)

    import httpx

    _readiness_url = (
        f"{base_url.rstrip('/')}/api/workspaces/{workspace_id}/governance/grc/readiness"
    )
    _auth_headers = {"Authorization": f"Bearer {api_key}"}

    # ── Snapshot before push (for delta display) ─────────────────────────────
    readiness_before: "dict | None" = None
    if show_delta:
        try:
            snap = httpx.get(_readiness_url, headers=_auth_headers, timeout=10.0)
            if snap.status_code == 200:
                readiness_before = snap.json()
        except Exception:
            pass

    # ── HTTP push ────────────────────────────────────────────────────────────
    url = f"{base_url.rstrip('/')}/api/workspaces/{workspace_id}/governance/grc/evidence"
    try:
        resp = httpx.post(
            url,
            json=payload,
            headers=_auth_headers,
            timeout=15.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:200]
        print(f"mima push: API returned {e.response.status_code} — {body}", file=sys.stderr)
        sys.exit(1)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print(f"mima push: cannot reach API — {e}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()

    if emit_json:
        print(json.dumps(data, indent=2))
    elif show_delta:
        # Snapshot after push and display delta
        readiness_after: "dict | None" = None
        try:
            after_snap = httpx.get(_readiness_url, headers=_auth_headers, timeout=10.0)
            if after_snap.status_code == 200:
                readiness_after = after_snap.json()
        except Exception:
            pass

        if readiness_before is not None and readiness_after is not None:
            _print_push_delta(record_type, data, readiness_before, readiness_after)
        else:
            # Delta fetch failed — graceful fallback
            record_id = data.get("record_id", "?")
            controls  = data.get("mapped_controls", [])
            print(f"\n  {record_type}  \u00b7  record saved")
            if controls:
                print(f"  Controls evidenced: {', '.join(controls)}")
            print(f"  Record ID: {record_id}\n")
    else:
        record_id = data.get("record_id", "?")
        controls  = data.get("mapped_controls", [])
        print(f"\n  {record_type}  \u00b7  record saved")
        if controls:
            print(f"  Controls evidenced: {', '.join(controls)}")
        print(f"  Record ID: {record_id}\n")


def _build_push_payload(record_type: str, flags: dict) -> "dict | None":
    """Build and validate the evidence payload from parsed CLI flags.

    Returns None (after printing an error) if required fields are missing.
    """

    def require(*keys: str) -> bool:
        missing = [k for k in keys if not flags.get(k)]
        if missing:
            flag_strs = ", ".join(f"--{k.replace('_', '-')}" for k in missing)
            print(f"mima push {record_type}: missing required flag(s): {flag_strs}",
                  file=sys.stderr)
            return False
        return True

    if record_type == "access_review":
        if not require("user", "resource", "granted", "reviewed_by"):
            return None
        granted_raw = flags["granted"].lower()
        if granted_raw not in ("true", "false", "1", "0", "yes", "no"):
            print("mima push: --granted must be true or false", file=sys.stderr)
            return None
        return {
            "payload": {
                "user":        flags["user"],
                "resource":    flags["resource"],
                "granted":     granted_raw in ("true", "1", "yes"),
                "reviewed_by": flags["reviewed_by"],
                "review_type": flags.get("review_type", "periodic"),
                **({} if not flags.get("reason") else {"reason": flags["reason"]}),
            },
            "identity":    flags["user"],
            "resource":    flags["resource"],
        }

    if record_type == "change_event":
        if not require("by", "description", "environment", "system"):
            return None
        p = {
            "payload": {
                "type":        flags.get("type", "deployment"),
                "by":          flags["by"],
                "description": flags["description"],
                "environment": flags["environment"],
                "system":      flags["system"],
            },
            "identity":    flags["by"],
            "resource":    flags["system"],
            "environment": flags["environment"],
        }
        if flags.get("change_id"):
            p["payload"]["change_id"] = flags["change_id"]
        return p

    if record_type == "vendor_risk":
        if not require("vendor", "tier", "last_reviewed"):
            return None
        valid_tiers = ("critical", "high", "medium", "low")
        if flags["tier"] not in valid_tiers:
            print(f"mima push: --tier must be one of: {', '.join(valid_tiers)}", file=sys.stderr)
            return None
        return {
            "payload": {
                "vendor":        flags["vendor"],
                "tier":          flags["tier"],
                "last_reviewed": flags["last_reviewed"],
                "findings":      int(flags.get("findings", "0")),
            },
            "resource": flags["vendor"],
        }

    if record_type == "policy_acknowledged":
        if not require("policy", "user", "version"):
            return None
        return {
            "payload": {
                "policy":  flags["policy"],
                "user":    flags["user"],
                "version": flags["version"],
                "channel": flags.get("channel", "in-app"),
                **({} if not flags.get("session_id") else {"session_id": flags["session_id"]}),
            },
            "identity": flags["user"],
            "resource":  flags["policy"],
        }

    if record_type == "incident_report":
        if not require("title", "severity", "description", "affected_systems"):
            return None
        valid_severities = ("critical", "high", "medium", "low")
        if flags["severity"] not in valid_severities:
            print(f"mima push: --severity must be one of: {', '.join(valid_severities)}",
                  file=sys.stderr)
            return None
        systems = [s.strip() for s in flags["affected_systems"].split(",") if s.strip()]
        incident_payload: dict = {
            "title":            flags["title"],
            "severity":         flags["severity"],
            "description":      flags["description"],
            "affected_systems": systems,
        }
        if flags.get("authority_notified_at"):
            incident_payload["authority_notified_at"] = flags["authority_notified_at"]
        p: dict = {"payload": incident_payload}
        if flags.get("detected_at"):
            p["occurred_at"] = flags["detected_at"]
        return p

    if record_type == "ai_risk_assessment":
        if not require("system", "risk_tier", "use_case", "impact_domains",
                       "art5_self_assessment", "assessor"):
            return None
        valid_tiers = ("unacceptable", "high", "limited", "minimal")
        if flags["risk_tier"] not in valid_tiers:
            print(f"mima push: --risk-tier must be one of: {', '.join(valid_tiers)}",
                  file=sys.stderr)
            return None
        a5_raw = flags["art5_self_assessment"].lower()
        if a5_raw not in ("true", "false", "1", "0", "yes", "no"):
            print("mima push: --art5-self-assessment must be true or false", file=sys.stderr)
            return None
        domains = [d.strip() for d in flags["impact_domains"].split(",") if d.strip()]
        ai_payload: dict = {
            "system_name":          flags["system"],
            "risk_tier":            flags["risk_tier"],
            "use_case":             flags["use_case"],
            "impact_domains":       domains,
            "art5_self_assessment": a5_raw in ("true", "1", "yes"),
            "assessor":             flags["assessor"],
        }
        if flags.get("technical_doc_url"):
            ai_payload["technical_doc_url"] = flags["technical_doc_url"]
        p = {"payload": ai_payload, "resource": flags["system"]}
        if flags.get("assessment_date"):
            p["occurred_at"] = flags["assessment_date"]
        return p

    if record_type == "training_data_governance":
        if not require("model_id", "dataset_id", "record_count",
                       "bias_checks_performed", "approved_by",
                       "data_sources", "data_categories"):
            return None
        bc_raw = flags["bias_checks_performed"].lower()
        if bc_raw not in ("true", "false", "1", "0", "yes", "no"):
            print("mima push: --bias-checks-performed must be true or false", file=sys.stderr)
            return None
        sources = [s.strip() for s in flags["data_sources"].split(",") if s.strip()]
        categories = [c.strip() for c in flags["data_categories"].split(",") if c.strip()]
        tdg_payload: dict = {
            "model_id":              flags["model_id"],
            "dataset_id":            flags["dataset_id"],
            "record_count":          int(flags["record_count"]),
            "bias_checks_performed": bc_raw in ("true", "1", "yes"),
            "approved_by":           flags["approved_by"],
            "data_sources":          sources,
            "data_categories":       categories,
        }
        if flags.get("known_limitations"):
            tdg_payload["known_limitations"] = flags["known_limitations"]
        p = {
            "payload":  tdg_payload,
            "resource": flags["dataset_id"],
            "identity": flags["approved_by"],
        }
        if flags.get("approval_date"):
            p["occurred_at"] = flags["approval_date"]
        return p

    if record_type == "model_evaluation":
        if not require("model_id", "dataset", "accuracy", "evaluated_by"):
            return None
        valid_eval_types = ("initial", "quarterly", "triggered")
        eval_type = flags.get("evaluation_type", "quarterly")
        if eval_type not in valid_eval_types:
            print(f"mima push: --evaluation-type must be one of: {', '.join(valid_eval_types)}",
                  file=sys.stderr)
            return None
        me_payload: dict = {
            "model_id":        flags["model_id"],
            "dataset":         flags["dataset"],
            "accuracy":        float(flags["accuracy"]),
            "evaluated_by":    flags["evaluated_by"],
            "evaluation_type": eval_type,
        }
        if flags.get("passed_threshold") is not None:
            pt_raw = flags["passed_threshold"].lower()
            me_payload["passed_threshold"] = pt_raw in ("true", "1", "yes")
        p = {
            "payload":  me_payload,
            "resource": flags["model_id"],
            "identity": flags["evaluated_by"],
        }
        if flags.get("evaluation_date"):
            p["occurred_at"] = flags["evaluation_date"]
        return p

    if record_type == "human_oversight":
        if not require("decision_id", "ai_recommendation", "human_decision", "reviewer"):
            return None
        ho_payload: dict = {
            "decision_id":       flags["decision_id"],
            "ai_recommendation": flags["ai_recommendation"],
            "human_decision":    flags["human_decision"],
            "reviewer":          flags["reviewer"],
            "override":          flags["ai_recommendation"] != flags["human_decision"],
        }
        if flags.get("rationale"):
            ho_payload["rationale"] = flags["rationale"]
        if flags.get("model_id"):
            ho_payload["model_id"] = flags["model_id"]
        return {
            "payload":  ho_payload,
            "resource": flags["decision_id"],
            "identity": flags["reviewer"],
        }

    if record_type == "model_drift_event":
        if not require("model_id", "metric", "baseline", "current",
                       "threshold", "detected_by"):
            return None
        valid_drift_types = ("performance", "data", "concept")
        drift_type = flags.get("drift_type", "performance")
        if drift_type not in valid_drift_types:
            print(f"mima push: --drift-type must be one of: {', '.join(valid_drift_types)}",
                  file=sys.stderr)
            return None
        mde_payload: dict = {
            "model_id":    flags["model_id"],
            "metric":      flags["metric"],
            "baseline":    float(flags["baseline"]),
            "current":     float(flags["current"]),
            "threshold":   float(flags["threshold"]),
            "drift_type":  drift_type,
            "detected_by": flags["detected_by"],
        }
        if flags.get("action_taken"):
            mde_payload["action_taken"] = flags["action_taken"]
        p = {
            "payload":  mde_payload,
            "resource": flags["model_id"],
            "identity": flags["detected_by"],
        }
        if flags.get("detection_date"):
            p["occurred_at"] = flags["detection_date"]
        return p

    if record_type == "governance_review":
        if not require("reviewed_by", "report_type", "frameworks", "overall_readiness"):
            return None
        readiness = int(flags["overall_readiness"])
        if not (0 <= readiness <= 100):
            print("mima push: --overall-readiness must be 0–100", file=sys.stderr)
            return None
        fws = [f.strip() for f in flags["frameworks"].split(",") if f.strip()]
        gr_payload: dict = {
            "reviewed_by":         flags["reviewed_by"],
            "report_type":         flags["report_type"],
            "frameworks_reviewed": fws,
            "overall_readiness":   readiness,
            "action_items":        int(flags.get("action_items", "0")),
        }
        if flags.get("notes"):
            gr_payload["notes"] = flags["notes"]
        p = {"payload": gr_payload, "identity": flags["reviewed_by"]}
        if flags.get("review_date"):
            p["occurred_at"] = flags["review_date"]
        return p

    return None


_INIT_TEST_TEMPLATE = '''\
"""Governance policy tests — generated by `mima init`.

Edit SCAN_PATH to narrow the scan (e.g. "src/" instead of ".").
Run:  mima test {output_path}
"""
from mima_governance.testing import GovernanceTest, assert_attested

SCAN_PATH = {scan_path_repr}  # folder scanned; edit to narrow scope


class TestAttestation(GovernanceTest):
    """Assert AI call sites in SCAN_PATH are covered by @mima.attest().{detected_comment}
    """

    def test_all_calls_attested(self):
        """Fail if any unattested high-confidence AI call sites exist."""
        result = self.scan(SCAN_PATH)
        return assert_attested(result, min_coverage=1.0)

    def test_coverage_threshold(self):
        """Soft gate — passes at 80%. Raise min_coverage to tighten."""
        result = self.scan(SCAN_PATH)
        return assert_attested(result, min_coverage=0.8)
'''


def _cmd_init(args: List[str]) -> None:
    """Handle `mima init [path] [--output FILE] [--force]`."""
    if args and args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima init — generate a governance test file from your codebase

            Usage:
                mima init [path] [--output FILE] [--force]

            Arguments:
                path              Codebase path to scan (default: .)
                --output FILE     Where to write the test file
                                  (default: tests/test_governance.py)
                --force           Overwrite existing output file

            What it does:
                Scans your code for AI library call sites, then generates a
                GovernanceTest file with assertions you can run immediately
                and commit to CI/CD.

            After running:
                mima test tests/test_governance.py   # run the generated tests
                mima login                           # connect to compliance dashboard
        """))
        sys.exit(0)

    force     = "--force" in args
    output    = "tests/test_governance.py"
    scan_path = "."
    clean     = [a for a in args if a != "--force"]

    i = 0
    while i < len(clean):
        if clean[i] == "--output" and i + 1 < len(clean):
            output = clean[i + 1]
            i += 2
        elif not clean[i].startswith("--"):
            scan_path = clean[i]
            i += 1
        else:
            i += 1

    out_path = Path(output)
    if out_path.exists() and not force:
        print(
            f"mima init: {output} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n  Scanning {scan_path} for AI call sites...", end=" ", flush=True)
    import time
    t0 = time.perf_counter()
    detections, files_scanned = _scan_path(Path(scan_path))
    duration_ms = (time.perf_counter() - t0) * 1000

    high_conf  = [d for d in detections if d.confidence == "high"]
    unattested = [d for d in high_conf if not d.attested]
    libs       = sorted({d.library for d in high_conf})

    if high_conf:
        n_calls = len(high_conf)
        n_files = len({d.file for d in high_conf})
        print(
            f"{n_calls} call{'s' if n_calls != 1 else ''} across "
            f"{n_files} file{'s' if n_files != 1 else ''} ({duration_ms:.0f}ms)"
        )
    else:
        print(f"none found in {files_scanned:,} files ({duration_ms:.0f}ms)")

    # Build the comment block embedded in the class docstring
    if libs:
        coverage_now = (
            int((len(high_conf) - len(unattested)) / len(high_conf) * 100)
            if high_conf else 100
        )
        libs_str = ", ".join(libs)
        detected_comment = (
            f"\n\n    mima init detected: {libs_str}"
            f"\n    {len(unattested)} unattested  \u00b7  {coverage_now}% coverage now"
            f"\n    Run `mima scan {scan_path}` to see the full list."
        )
    else:
        detected_comment = (
            "\n\n    mima init found no AI library calls — tests will pass immediately."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = _INIT_TEST_TEMPLATE.format(
        output_path=output,
        scan_path_repr=repr(scan_path),
        detected_comment=detected_comment,
    )
    out_path.write_text(content)

    print(f"  Writing {output}")
    print(f"    TestAttestation.test_all_calls_attested"
          f"   \u2014 assert 0 unattested")
    print(f"    TestAttestation.test_coverage_threshold"
          f"   \u2014 assert \u226580% attested")
    print(f"\n  Run:    mima test {output}")
    print(f"\n  Add to CI/CD (.github/workflows/governance.yml):")
    print(f"    - name: Governance check")
    print(f"      run: |")
    print(f"        pip install mima-governance")
    print(f"        mima test {output}")
    print(f"\n  Next: `mima login` to connect results to your compliance dashboard.\n")


_COMMANDS = {
    "scan":   _cmd_scan,
    "init":   _cmd_init,
    "login":  _cmd_login,
    "status": _cmd_status,
    "test":   _cmd_test,
    "push":   _cmd_push,
}


def main() -> None:
    """Entry point for ``mima`` script."""
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima — AI governance CLI

            Commands:
                mima init [path]          Generate a governance test file from your codebase
                mima scan <path>          Detect unattested AI call sites
                mima test <file>          Run governance policy assertions
                mima status               Show certification readiness scores
                mima login                Authenticate with the Mima API
                mima push <record_type>   Push a GRC evidence record from CI/CD

            Run `mima <command> --help` for command-specific options.

            Quick start (no account needed):
                mima init .                             # generate tests/test_governance.py
                mima test tests/test_governance.py      # run policy tests immediately

            With a Mima account:
                mima login                              # store API credentials
                mima scan .                             # find unattested AI calls
                mima status                             # check readiness scores
                mima push change_event --by ci \\       # push evidence from CI/CD
                    --description "Deploy v1.2" \\
                    --environment production \\
                    --system api-service
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
