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


def main() -> None:
    """Entry point for ``mima`` script."""
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima scan — detect unattested AI call sites in Python source code

            Usage:
                mima scan <path> [options]

            Options:
                --json              Emit JSON array of detections to stdout
                --include PATTERN   Glob pattern for files (default: **/*.py)
                -h, --help          Show this message

            Each detection includes:
                file, line, library, attested (bool), confidence ("high"|"low")

            Known limitations:
                - Aliased imports (from openai import OpenAI; c = OpenAI()) are NOT
                  detected — the tokenizer sees 'OpenAI.' not 'openai.'.
                - Class-level @mima decorators do not cover method calls inside the class.
                - Indirect calls through wrappers (my_llm.call()) are not detected.
                - confidence="low" detections may be string literals or comments —
                  review manually before treating as real call sites.

            Exit codes:
                0  — scan completed (unattested sites may still be present)
                1  — path not found or scan error
        """))
        sys.exit(0)

    if args[0] != "scan":
        print(f"mima: unknown command '{args[0]}' — try 'mima scan --help'", file=sys.stderr)
        sys.exit(1)

    scan_args = args[1:]
    emit_json = "--json" in scan_args
    include   = "**/*.py"

    # Parse --include PATTERN
    cleaned: List[str] = []
    i = 0
    while i < len(scan_args):
        if scan_args[i] == "--include" and i + 1 < len(scan_args):
            include = scan_args[i + 1]
            i += 2
        elif scan_args[i] not in ("--json",):
            cleaned.append(scan_args[i])
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
