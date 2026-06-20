"""mima webhooks — manage governance event webhook registrations.

Usage:
    mima webhooks                                        List registered webhooks
    mima webhooks register URL [--events E1,E2...]       Register a new webhook endpoint
                              [--secret SECRET]
    mima webhooks delete WEBHOOK_ID                      Remove a webhook registration

Credentials:
    Set MIMA_API_KEY and MIMA_WORKSPACE_ID environment variables,
    or run `mima login` to store them.

Events (default: all):
    evidence.created          New evidence record pushed
    approval.requested        Human approval requested by a guarded agent
    approval.decided          Approval approved or rejected
    approval.expired          Approval timed out with no decision
    gate.failed               A required governance gate dropped below threshold
    policy.check.failed       A policy assertion check returned failing
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

_ALL_EVENTS = [
    "evidence.created",
    "approval.requested",
    "approval.decided",
    "approval.expired",
    "gate.failed",
    "policy.check.failed",
]


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
        print(f"mima webhooks: API error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(3)
    except httpx.RequestError as e:
        print(f"mima webhooks: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


def _api_post(path: str, body: dict, api_key: str, base_url: str):
    import httpx
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        r = httpx.post(f"{base_url}{path}", json=body, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"mima webhooks: API error {e.response.status_code}: {e.response.text}", file=sys.stderr)
        sys.exit(3)
    except httpx.RequestError as e:
        print(f"mima webhooks: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


def _api_delete(path: str, api_key: str, base_url: str) -> int:
    import httpx
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r = httpx.delete(f"{base_url}{path}", headers=headers, timeout=30)
        return r.status_code
    except httpx.RequestError as e:
        print(f"mima webhooks: could not reach API — {e}", file=sys.stderr)
        sys.exit(3)


# ── Sub-commands ────────────────────────────────────────────────────────────────

def cmd_list(args: List[str]) -> None:
    """List all registered webhook endpoints."""
    emit_json = "--json" in args or "-j" in args

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima webhooks: credentials not set — run `mima login` or set MIMA_API_KEY + MIMA_WORKSPACE_ID.", file=sys.stderr)
        sys.exit(1)

    data = _api_get(f"/workspaces/{workspace_id}/governance/webhooks", api_key, base_url)
    webhooks = data.get("webhooks", [])

    if emit_json:
        print(json.dumps(webhooks, indent=2))
        return

    if not webhooks:
        print("\n  No webhooks registered.")
        print("  Run `mima webhooks register URL` to add one.\n")
        return

    print(f"\n  Webhooks — workspace {workspace_id[:8]}\n")
    for wh in webhooks:
        wid    = wh.get("id", "")[:8]
        url    = wh.get("url", "")
        events = wh.get("events") or _ALL_EVENTS
        masked = "***" if wh.get("has_secret") else "none"
        ev_str = ", ".join(events) if len(events) <= 3 else f"{len(events)} events"
        print(f"  {wid}  {url}")
        print(f"          events: {ev_str}  ·  secret: {masked}")
        if wh.get("last_delivery_at"):
            status = wh.get("last_delivery_status", "?")
            print(f"          last delivery: {wh['last_delivery_at'][:19].replace('T', ' ')}  status: {status}")
    print()


def cmd_register(args: List[str]) -> None:
    """Register a new webhook endpoint."""
    if not args:
        print("Usage: mima webhooks register URL [--events E1,E2] [--secret SECRET]", file=sys.stderr)
        sys.exit(2)

    url: str = args[0]
    events: Optional[List[str]] = None
    secret: Optional[str] = None

    i = 1
    while i < len(args):
        if args[i] == "--events" and i + 1 < len(args):
            events = [e.strip() for e in args[i + 1].split(",") if e.strip()]
            i += 2
        elif args[i] == "--secret" and i + 1 < len(args):
            secret = args[i + 1]
            i += 2
        else:
            i += 1

    # Validate events.
    if events:
        unknown = [e for e in events if e not in _ALL_EVENTS]
        if unknown:
            print(f"mima webhooks register: unknown event(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"  Valid events: {', '.join(_ALL_EVENTS)}", file=sys.stderr)
            sys.exit(2)

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima webhooks register: credentials not set — run `mima login`.", file=sys.stderr)
        sys.exit(1)

    body: dict = {"url": url}
    if events:
        body["events"] = events
    if secret:
        body["secret"] = secret

    data = _api_post(f"/workspaces/{workspace_id}/governance/webhooks", body, api_key, base_url)

    wid    = data.get("id", "")
    ev_out = data.get("events") or _ALL_EVENTS
    ev_str = ", ".join(ev_out) if len(ev_out) <= 3 else f"{len(ev_out)} events"
    print(f"\n  Webhook registered")
    print(f"    ID:     {wid}")
    print(f"    URL:    {url}")
    print(f"    Events: {ev_str}")
    if secret:
        print(f"    Secret: set (HMAC-SHA256 — verify X-Mima-Signature header in your receiver)")
    print()


def cmd_delete(args: List[str]) -> None:
    """Remove a webhook registration."""
    if not args:
        print("Usage: mima webhooks delete WEBHOOK_ID", file=sys.stderr)
        sys.exit(2)

    webhook_id = args[0]

    api_key, workspace_id, base_url = _get_credentials()
    if not api_key or not workspace_id:
        print("mima webhooks delete: credentials not set — run `mima login`.", file=sys.stderr)
        sys.exit(1)

    status = _api_delete(
        f"/workspaces/{workspace_id}/governance/webhooks/{webhook_id}",
        api_key, base_url,
    )
    if status == 404:
        print(f"mima webhooks delete: webhook {webhook_id!r} not found.", file=sys.stderr)
        sys.exit(1)
    if status not in (200, 204):
        print(f"mima webhooks delete: unexpected status {status}.", file=sys.stderr)
        sys.exit(1)

    print(f"  Webhook {webhook_id} removed.")


# ── Entry point (called by cli.py _cmd_webhooks) ───────────────────────────────

def run(args: List[str]) -> None:
    """Dispatch `mima webhooks [subcommand]` to sub-command handlers."""
    import textwrap

    if not args or args[0] in ("-h", "--help"):
        print(textwrap.dedent("""\
            mima webhooks — manage governance event webhooks

            Usage:
                mima webhooks                                   List registered webhooks
                mima webhooks register URL                      Register a new endpoint
                              [--events E1,E2] [--secret S]
                mima webhooks delete WEBHOOK_ID                 Remove a webhook

            Events (comma-separated, default: all):
                evidence.created      approval.requested    approval.decided
                approval.expired      gate.failed           policy.check.failed

            Examples:
                mima webhooks register https://your-server/mima-hook
                mima webhooks register https://your-server/mima-hook \\
                    --events approval.requested,approval.decided \\
                    --secret my-signing-secret
                mima webhooks delete a1b2c3d4
        """))
        sys.exit(0)

    sub  = args[0]
    rest = args[1:]

    if sub in ("list", "ls"):
        cmd_list(rest)
    elif sub in ("register", "add"):
        cmd_register(rest)
    elif sub in ("delete", "rm", "remove", "unregister"):
        cmd_delete(rest)
    else:
        print(f"mima webhooks: unknown subcommand '{sub}'. "
              "Use list, register, or delete.", file=sys.stderr)
        sys.exit(2)
