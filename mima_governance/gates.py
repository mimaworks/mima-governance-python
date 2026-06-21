"""mima gates — manage and check governance gate policies.

Usage:
    mima gates                                        List all configured gates + current status
    mima gates check [--system NAME] [--json]         Exit 1 if any required gate is below threshold
    mima gates set FRAMEWORK MODE [--threshold N]     Configure a gate (MODE = required|advisory)
              [--system NAME]
    mima gates unset FRAMEWORK [--system NAME]        Remove a gate policy

Credentials:
    Set MIMA_API_KEY. The workspace is resolved automatically from the key.
    Run `mima login` to store credentials, or set MIMA_API_KEY in CI.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

_FRAMEWORK_LABELS = {
    "soc2_type2": "SOC 2 Type II",
    "iso_27001":  "ISO 27001:2022",
    "iso_42001":  "ISO 42001",
    "eu_ai_act":  "EU AI Act",
    "nist_airf":  "NIST AI RMF",
}


def _label(slug: str) -> str:
    return _FRAMEWORK_LABELS.get(slug, slug)


def _resolve_workspace_from_key(api_key: str, base_url: str) -> Optional[str]:
    """Call GET /me to discover the workspace_id for this API key."""
    import httpx
    try:
        r = httpx.get(
            f"{base_url}/me",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("workspace_id")
    except Exception:
        pass
    return None


def _get_credentials():
    """Return (api_key, workspace_id, base_url) from env or config store.

    If MIMA_WORKSPACE_ID is not set, workspace_id is resolved automatically
    from the API key via GET /me and cached in ~/.mima/config.json so
    subsequent calls skip the network round-trip.
    """
    from . import config as _config
    api_key      = os.environ.get("MIMA_API_KEY")      or _config.get_api_key()
    workspace_id = os.environ.get("MIMA_WORKSPACE_ID") or _config.get_workspace_id()
    base_url     = os.environ.get("MIMA_BASE_URL")     or _config.get_base_url() or "https://api.mima.ai/api"

    # Auto-resolve workspace_id from API key — no need to set MIMA_WORKSPACE_ID.
    if api_key and not workspace_id:
        workspace_id = _resolve_workspace_from_key(api_key, base_url)
        if workspace_id:
            cfg = _config.load()
            cfg["workspace_id"] = workspace_id
            _config.save(cfg)

    return api_key, workspace_id, base_url


def _api_get(path: str, api_key: str, base_url: str):
    """Perform an authenticated GET and return parsed JSON."""
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = httpx.get(f"{base_url}{path}", headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"mima gates: API error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(3)
    except httpx.RequestError as e:
        print(f"mima gates: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


def _api_put(path: str, body: dict, api_key: str, base_url: str):
    import httpx
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = httpx.put(f"{base_url}{path}", json=body, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"mima gates: API error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(3)
    except httpx.RequestError as e:
        print(f"mima gates: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


def _api_delete(path: str, api_key: str, base_url: str) -> int:
    """Return HTTP status code."""
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = httpx.delete(f"{base_url}{path}", headers=headers, timeout=30)
        return r.status_code
    except httpx.RequestError as e:
        print(f"mima gates: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


def _status_str(mode: str, status: str) -> str:
    if mode == "advisory":
        return "passing (advisory)"
    return status.upper() if status == "failing" else "passing"


# ── Sub-commands ───────────────────────────────────────────────────────────────

def cmd_list(args: List[str]) -> None:
    """List all configured gate policies with live status."""
    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima gates: credentials not set — run `mima login` or set MIMA_API_KEY + MIMA_WORKSPACE_ID.", file=sys.stderr)
        sys.exit(1)

    data = _api_get(f"/workspaces/{workspace_id}/governance/grc/gates", api_key, base_url)
    policies     = data.get("policies", [])
    unconfigured = data.get("unconfigured_frameworks", [])

    print(f"\n  Governance gates — workspace {workspace_id[:8]}\n")

    ws_policies = [p for p in policies if p["scope"] == "workspace"]
    sys_policies = [p for p in policies if p["scope"] != "workspace"]

    if ws_policies:
        print("  WORKSPACE-WIDE")
        for p in ws_policies:
            pct_str  = f"{p['current_pct']}%"
            thresh   = f"{p['threshold_pct']}%"
            status   = _status_str(p["mode"], p["status"])
            fw_label = _label(p["framework"]).ljust(18)
            print(f"    {fw_label}  {p['mode']:<10}  {thresh}  current: {pct_str:<5}  {status}")
        print()

    if sys_policies:
        # Group by scope.
        by_scope: dict = {}
        for p in sys_policies:
            by_scope.setdefault(p["scope"], []).append(p)
        print("  PER-SYSTEM OVERRIDES")
        for scope, ps in by_scope.items():
            print(f"    {scope}")
            for p in ps:
                pct_str  = f"{p['current_pct']}%"
                thresh   = f"{p['threshold_pct']}%"
                status   = _status_str(p["mode"], p["status"])
                fw_label = _label(p["framework"]).ljust(18)
                print(f"      {fw_label}  {p['mode']:<10}  {thresh}  current: {pct_str:<5}  {status}")
        print()

    if unconfigured:
        labels = ", ".join(_label(s) for s in unconfigured)
        print(f"  Advisory by default: {labels}")
        print()

    required_total  = sum(1 for p in policies if p["mode"] == "required")
    required_pass   = sum(1 for p in policies if p["mode"] == "required" and p["status"] == "passing")
    required_fail   = required_total - required_pass

    if required_total > 0:
        print(f"  {required_total} required gate{'s' if required_total != 1 else ''}  ·  "
              f"{required_pass} passing  ·  {required_fail} failing")
        if required_fail > 0:
            print("  Run `mima gates check` to exit with code 1 if any required gate fails.")
    else:
        print("  No required gates configured — all advisory.")

    print()


def cmd_check(args: List[str]) -> None:
    """Check gates and exit 1 if any required gate is below threshold."""
    system_name: Optional[str] = None
    emit_json = False

    i = 0
    while i < len(args):
        if args[i] == "--system" and i + 1 < len(args):
            system_name = args[i + 1]
            i += 2
        elif args[i] in ("--json", "-j"):
            emit_json = True
            i += 1
        else:
            i += 1

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima gates check: credentials not set — run `mima login`.", file=sys.stderr)
        sys.exit(1)

    path = f"/workspaces/{workspace_id}/governance/grc/gates/check"
    if system_name:
        path += f"?system_name={system_name}"

    data = _api_get(path, api_key, base_url)

    if emit_json:
        print(json.dumps(data, indent=2))
        sys.exit(0 if data.get("passed") else 1)

    sn_label = f" for system: {system_name}" if system_name else " (workspace-wide)"
    print(f"\n  Checking governance gates{sn_label}\n")

    for r in data.get("results", []):
        advisory_note = "  (advisory — not enforced)" if r["mode"] == "advisory" else ""
        status_str    = r["status"].upper()
        fw_label      = _label(r["framework"]).ljust(18)
        print(f"  {fw_label}  {r['mode']:<10}  {r['threshold_pct']}%  current: {r['current_pct']}%  "
              f"{status_str}{advisory_note}")

    print()
    if data.get("passed"):
        failing_required = [r for r in data.get("results", []) if r["mode"] == "required" and r["status"] == "failing"]
        print(f"  Result: PASS — all required gates at or above threshold")
        print()
        sys.exit(0)
    else:
        failing = [r for r in data.get("results", []) if r["mode"] == "required" and r["status"] == "failing"]
        count = len(failing)
        print(f"  Result: FAIL ({count} required gate{'s' if count != 1 else ''} below threshold)")
        print()
        sys.exit(1)


def cmd_set(args: List[str]) -> None:
    """Set (upsert) a gate policy."""
    if len(args) < 2:
        print("Usage: mima gates set FRAMEWORK required|advisory [--threshold N] [--system NAME]",
              file=sys.stderr)
        sys.exit(2)

    framework = args[0]
    mode      = args[1]
    if mode not in ("required", "advisory"):
        print("mima gates set: MODE must be 'required' or 'advisory'.", file=sys.stderr)
        sys.exit(2)

    threshold_pct = 80
    system_name: Optional[str] = None

    i = 2
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args):
            try:
                threshold_pct = int(args[i + 1])
            except ValueError:
                print("mima gates set: --threshold must be an integer 0–100.", file=sys.stderr)
                sys.exit(2)
            i += 2
        elif args[i] == "--system" and i + 1 < len(args):
            system_name = args[i + 1]
            i += 2
        else:
            i += 1

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima gates set: credentials not set — run `mima login`.", file=sys.stderr)
        sys.exit(1)

    scope = system_name if system_name else "workspace"
    body  = {"framework": framework, "mode": mode, "threshold_pct": threshold_pct, "scope": scope}
    data  = _api_put(f"/workspaces/{workspace_id}/governance/grc/gates", body, api_key, base_url)

    scope_label = f"  {system_name}" if system_name else "  workspace-wide"
    fw_label    = _label(data["framework"])
    print(f"  Gate set: {fw_label}  {data['mode']}  {data['threshold_pct']}%{scope_label}")


def cmd_unset(args: List[str]) -> None:
    """Remove a gate policy by framework (and optional system)."""
    if not args:
        print("Usage: mima gates unset FRAMEWORK [--system NAME]", file=sys.stderr)
        sys.exit(2)

    framework = args[0]
    system_name: Optional[str] = None

    i = 1
    while i < len(args):
        if args[i] == "--system" and i + 1 < len(args):
            system_name = args[i + 1]
            i += 2
        else:
            i += 1

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima gates unset: credentials not set — run `mima login`.", file=sys.stderr)
        sys.exit(1)

    # First: find the policy ID for this (framework, scope).
    data  = _api_get(f"/workspaces/{workspace_id}/governance/grc/gates", api_key, base_url)
    scope = system_name if system_name else "workspace"
    match = next(
        (p for p in data.get("policies", []) if p["framework"] == framework and p["scope"] == scope),
        None,
    )
    if match is None:
        fw_label    = _label(framework)
        scope_label = f" / {system_name}" if system_name else " / workspace"
        print(f"mima gates unset: no policy found for {fw_label}{scope_label}.", file=sys.stderr)
        sys.exit(1)

    status = _api_delete(
        f"/workspaces/{workspace_id}/governance/grc/gates/{match['id']}",
        api_key, base_url,
    )
    if status not in (200, 204):
        print(f"mima gates unset: delete returned unexpected status {status}.", file=sys.stderr)
        sys.exit(1)

    fw_label    = _label(framework)
    scope_label = f"  {system_name}" if system_name else "  workspace"
    print(f"  Gate removed: {fw_label}{scope_label}")
    if system_name:
        # Check if there's a workspace fallback.
        ws_match = next(
            (p for p in data.get("policies", []) if p["framework"] == framework and p["scope"] == "workspace"),
            None,
        )
        if ws_match:
            print(f"  (Falls back to workspace-wide policy: {ws_match['mode']}, {ws_match['threshold_pct']}%)")


# ── Entry point (called by cli.py _cmd_gates) ─────────────────────────────────

def run(args: List[str]) -> None:
    """Dispatch `mima gates [subcommand]` to sub-command handlers."""
    import textwrap

    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima gates — manage governance gate policies

            Usage:
                mima gates                                      List configured gates + current status
                mima gates check [--system NAME] [--json]       Exit 1 if any required gate fails
                mima gates set FRAMEWORK MODE                   Set a gate (MODE=required|advisory)
                              [--threshold N] [--system NAME]
                mima gates unset FRAMEWORK [--system NAME]      Remove a gate policy

            Framework slugs:
                eu_ai_act   soc2_type2   iso_27001   iso_42001   nist_airf

            Examples:
                mima gates set eu_ai_act required --threshold 60
                mima gates set eu_ai_act required --threshold 80 --system inference-service
                mima gates check
                mima gates check --system inference-service --json
                mima gates unset eu_ai_act --system inference-service
        """))
        sys.exit(0)

    sub = args[0]
    rest = args[1:]

    if sub in ("list", "ls"):
        cmd_list(rest)
    elif sub == "check":
        cmd_check(rest)
    elif sub == "set":
        cmd_set(rest)
    elif sub in ("unset", "rm", "remove"):
        cmd_unset(rest)
    else:
        print(f"mima gates: unknown subcommand '{sub}'. "
              "Use list, check, set, or unset.", file=sys.stderr)
        sys.exit(2)
