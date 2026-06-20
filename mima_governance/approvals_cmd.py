"""mima approvals — list and decide on pending human-approval requests.

Usage:
    mima approvals                               List pending approvals (short)
    mima approvals list [--all] [--json]         List approvals (--all includes decided/expired)
    mima approvals decide ID approve             Approve a pending request
    mima approvals decide ID reject              Reject a pending request
                          [--reason "text"]

Credentials:
    Set MIMA_API_KEY and MIMA_WORKSPACE_ID environment variables,
    or run `mima login` to store them.

Exit codes:
    0  success
    1  credential / not-found error
    2  usage error
    3  API / network error
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

_STATUS_LABELS = {
    "pending":  "PENDING",
    "approved": "approved",
    "rejected": "rejected",
    "expired":  "expired",
    "cancelled":"cancelled",
}

_STATUS_MARKER = {
    "pending":  "·",
    "approved": "✓",
    "rejected": "✗",
    "expired":  "—",
    "cancelled":"—",
}


def _get_credentials():
    """Return (api_key, workspace_id, base_url) from env or config store."""
    from . import config as _config
    api_key      = os.environ.get("MIMA_API_KEY")      or _config.get_api_key()
    workspace_id = os.environ.get("MIMA_WORKSPACE_ID") or _config.get_workspace_id()
    base_url     = os.environ.get("MIMA_BASE_URL")     or _config.get_base_url() or "https://api.mima.ai/api"
    return api_key, workspace_id, base_url


def _api_get(path: str, api_key: str, base_url: str):
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = httpx.get(f"{base_url}{path}", headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"mima approvals: API error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(3)
    except httpx.RequestError as e:
        print(f"mima approvals: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


def _api_post(path: str, body: dict, api_key: str, base_url: str):
    import httpx
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = httpx.post(f"{base_url}{path}", json=body, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"mima approvals: API error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(3)
    except httpx.RequestError as e:
        print(f"mima approvals: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


def _fmt_ts(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    return iso[:16].replace("T", " ")


# ── Sub-commands ────────────────────────────────────────────────────────────────

def cmd_list(args: List[str]) -> None:
    """List approval requests — pending only by default, or all with --all."""
    include_all = "--all" in args or "-a" in args
    emit_json   = "--json" in args or "-j" in args

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima approvals: credentials not set — run `mima login` or set MIMA_API_KEY + MIMA_WORKSPACE_ID.", file=sys.stderr)
        sys.exit(1)

    path = f"/workspaces/{workspace_id}/governance/approvals"
    if include_all:
        path += "?status=all"

    data = _api_get(path, api_key, base_url)
    approvals = data.get("approvals", [])

    if emit_json:
        print(json.dumps(approvals, indent=2))
        return

    if not approvals:
        if include_all:
            print("\n  No approval requests found.\n")
        else:
            print("\n  No pending approvals.\n")
        return

    pending = [a for a in approvals if a.get("status") == "pending"]
    others  = [a for a in approvals if a.get("status") != "pending"]

    label = "All approvals" if include_all else "Pending approvals"
    print(f"\n  {label} — workspace {workspace_id[:8]}\n")

    def _print_row(a: dict) -> None:
        aid     = a.get("id", "")[:8]
        status  = a.get("status", "pending")
        marker  = _STATUS_MARKER.get(status, "·")
        label   = _STATUS_LABELS.get(status, status.upper())
        action  = a.get("action_type", "unknown")
        system  = a.get("system_name") or a.get("agent_name") or ""
        req_by  = a.get("requested_by") or ""
        req_at  = _fmt_ts(a.get("requested_at"))
        expires = _fmt_ts(a.get("expires_at"))

        detail_parts = [f"action: {action}"]
        if system:
            detail_parts.append(f"system: {system}")
        if req_by:
            detail_parts.append(f"by: {req_by}")
        detail_str = "  ·  ".join(detail_parts)

        print(f"  {marker}  {aid}  {label:<9}  {detail_str}")
        print(f"            requested: {req_at}  expires: {expires}")

        if a.get("description"):
            desc = a["description"]
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"            {desc}")

        if status in ("approved", "rejected") and a.get("decided_by"):
            reason = f"  reason: {a['reason']}" if a.get("reason") else ""
            print(f"            decided by: {a['decided_by']}  at: {_fmt_ts(a.get('decided_at'))}{reason}")

        print()

    for a in pending:
        _print_row(a)

    if include_all and others:
        if pending:
            print("  ── Decided / expired ──────────────────────────────────────────────\n")
        for a in others:
            _print_row(a)

    if pending:
        print(f"  {len(pending)} pending  ·  run `mima approvals decide ID approve|reject` to action\n")


def cmd_decide(args: List[str]) -> None:
    """Approve or reject a pending approval request."""
    if len(args) < 2:
        print("Usage: mima approvals decide APPROVAL_ID approve|reject [--reason TEXT]",
              file=sys.stderr)
        sys.exit(2)

    approval_id = args[0]
    decision    = args[1].lower()

    if decision not in ("approve", "approved", "reject", "rejected"):
        print("mima approvals decide: decision must be 'approve' or 'reject'.", file=sys.stderr)
        sys.exit(2)

    # Normalise to the verb form the API expects.
    decision = "approve" if decision in ("approve", "approved") else "reject"

    reason: Optional[str] = None
    i = 2
    while i < len(args):
        if args[i] == "--reason" and i + 1 < len(args):
            reason = args[i + 1]
            i += 2
        else:
            i += 1

    if decision == "reject" and not reason:
        # Encourage a reason but don't require it.
        pass

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima approvals decide: credentials not set — run `mima login`.", file=sys.stderr)
        sys.exit(1)

    body: dict = {"decision": decision}
    if reason:
        body["reason"] = reason

    data = _api_post(
        f"/workspaces/{workspace_id}/governance/approvals/{approval_id}/decide",
        body, api_key, base_url,
    )

    status     = data.get("status", decision + "d")
    action     = data.get("action_type", "")
    system     = data.get("system_name") or data.get("agent_name") or ""
    decided_by = data.get("decided_by", "")

    if decision == "approve":
        print(f"\n  ✓  Approved — {approval_id[:8]}")
    else:
        print(f"\n  ✗  Rejected — {approval_id[:8]}")

    if action:
        print(f"     action: {action}" + (f"  ·  system: {system}" if system else ""))
    if decided_by:
        print(f"     decided by: {decided_by}")
    if reason:
        print(f"     reason: {reason}")

    # If the server returned a timeout token (unblocked after approval), surface it.
    if data.get("unblocked_at"):
        print(f"     agent unblocked at: {_fmt_ts(data['unblocked_at'])}")

    print()


# ── Entry point (called by cli.py _cmd_approvals) ─────────────────────────────

def run(args: List[str]) -> None:
    """Dispatch `mima approvals [subcommand]` to sub-command handlers."""
    import textwrap

    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima approvals — list and action human-approval requests

            Usage:
                mima approvals                          List pending approvals
                mima approvals list [--all] [--json]    List approvals (--all: include decided)
                mima approvals decide ID approve        Approve a pending request
                mima approvals decide ID reject         Reject a pending request
                              [--reason "text"]

            Examples:
                mima approvals
                mima approvals list --all
                mima approvals decide a1b2c3d4 approve
                mima approvals decide a1b2c3d4 reject --reason "Violates DLP policy section 4.2"
        """))
        sys.exit(0)

    sub  = args[0]
    rest = args[1:]

    if sub in ("list", "ls"):
        cmd_list(rest)
    elif sub == "decide":
        cmd_decide(rest)
    else:
        # Bare `mima approvals` with no subcommand defaults to listing pending.
        # But if the first arg looks like a flag, pass everything to cmd_list.
        if sub.startswith("-"):
            cmd_list(args)
        else:
            print(f"mima approvals: unknown subcommand '{sub}'. "
                  "Use list or decide.", file=sys.stderr)
            sys.exit(2)
