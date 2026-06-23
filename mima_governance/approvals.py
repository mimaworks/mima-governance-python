"""Pre-approval gates — ApprovalToken, TimeoutToken, polling loops, and exceptions.

`mima.require_approval()` blocks until a human approves or rejects via the dashboard.
On approval it returns an `ApprovalToken` that can be passed to `@mima.attest()` to link
the evidence record to the approval. On rejection/timeout it raises.

Poll schedule: 2s → 4s → 8s → 16s → 30s (cap). Server does inline expiry on GET.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Union


# ── Base exception ─────────────────────────────────────────────────────────────


class MimaGovernanceError(Exception):
    """Base class for all mima-governance runtime exceptions."""


# ── Approval exceptions ────────────────────────────────────────────────────────


class ApprovalDenied(MimaGovernanceError):
    """Raised when the GRC manager rejects the approval request."""

    def __init__(self, approval_id: str, rejected_by: str, reason: Optional[str] = None):
        self.approval_id = approval_id
        self.rejected_by = rejected_by
        self.reason = reason
        msg = f"Approval {approval_id!r} denied by {rejected_by!r}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ApprovalTimeout(MimaGovernanceError):
    """Raised when timeout_seconds elapsed with no decision (on_timeout='raise')."""

    def __init__(self, approval_id: str, timeout_seconds: int):
        self.approval_id = approval_id
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Approval {approval_id!r} not received within {timeout_seconds}s — "
            f"action was not taken. Re-trigger when the approver is available."
        )


class ApprovalCancelled(MimaGovernanceError):
    """Raised when the approval request was cancelled by the requester."""

    def __init__(self, approval_id: str):
        self.approval_id = approval_id
        super().__init__(f"Approval {approval_id!r} was cancelled")


# ── Token types ────────────────────────────────────────────────────────────────


@dataclass
class ApprovalToken:
    """Returned by require_approval() when a human approves the request.

    Single-use: pass once to @mima.attest(approval_token=token).
    A second use raises ValueError.
    """

    approval_id: str
    action_type: str
    approved_by: str
    approved_at: datetime
    expires_at: datetime
    _used: bool = field(default=False, repr=False, compare=False)

    def _mark_used(self) -> None:
        """Mark the token consumed. Raises ValueError on a second call."""
        if self._used:
            raise ValueError(
                f"ApprovalToken {self.approval_id!r} has already been consumed by "
                "@mima.attest(). Approval tokens are single-use — each approval "
                "covers exactly one @mima.attest() call."
            )
        self._used = True


@dataclass
class TimeoutToken:
    """Returned by require_approval(on_timeout='warn') when the request expires.

    Distinct from ApprovalToken — not a subclass. When passed to @mima.attest(),
    produces a human_oversight record with oversight_status='timeout_unblocked'.
    Does NOT earn EUAIA_ART14 or other human-oversight controls.
    """

    approval_id: str
    action_type: str
    timed_out_at: datetime
    timeout_seconds: int
    # No approved_by — no human approved this.


# ── Internal helpers ───────────────────────────────────────────────────────────

_POLL_INTERVALS = [2, 4, 8, 16, 30]  # seconds; last value repeats indefinitely


def _next_interval(idx: int) -> float:
    return float(_POLL_INTERVALS[min(idx, len(_POLL_INTERVALS) - 1)])


def _parse_dt(s: Optional[str]) -> datetime:
    """Parse an ISO-8601 timestamp; fall back to now() if absent."""
    if not s:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _handle_timeout(
    approval_id: str,
    action_type: str,
    timeout_seconds: int,
    on_timeout: str,
) -> TimeoutToken:
    """Either raise ApprovalTimeout or return a TimeoutToken, per on_timeout."""
    if on_timeout == "raise":
        raise ApprovalTimeout(approval_id, timeout_seconds)
    warnings.warn(
        f"[mima-governance] Approval {approval_id!r} timed out after {timeout_seconds}s "
        "— proceeding without human sign-off. "
        "The resulting evidence record will carry oversight_status='timeout_unblocked' "
        "and will NOT earn EUAIA_ART14 or other human-oversight controls.",
        stacklevel=5,
    )
    return TimeoutToken(
        approval_id=approval_id,
        action_type=action_type,
        timed_out_at=datetime.now(timezone.utc),
        timeout_seconds=timeout_seconds,
    )


# ── Sync polling loop ──────────────────────────────────────────────────────────


def poll_approval_sync(
    http,                 # httpx.Client, already configured with auth headers
    workspace_id: str,
    approval_id: str,
    timeout_seconds: int,
    on_timeout: str,
) -> Union[ApprovalToken, TimeoutToken]:
    """Block (with exponential backoff) until the approval reaches a terminal state."""
    url = f"/api/workspaces/{workspace_id}/governance/approvals/{approval_id}"
    interval_idx = 0
    deadline = time.monotonic() + timeout_seconds

    while True:
        resp = http.get(url)
        if resp.status_code >= 400:
            raise MimaGovernanceError(
                f"Approval poll failed: HTTP {resp.status_code} — {resp.text}"
            )
        data = resp.json()
        status = data["status"]

        if status == "approved":
            return ApprovalToken(
                approval_id=approval_id,
                action_type=data.get("action_type", ""),
                approved_by=data.get("approved_by", ""),
                approved_at=_parse_dt(data.get("decided_at")),
                expires_at=_parse_dt(data.get("expires_at")),
            )
        if status == "rejected":
            raise ApprovalDenied(
                approval_id,
                data.get("approved_by", ""),
                data.get("rejection_reason"),
            )
        if status in ("expired",):
            return _handle_timeout(
                approval_id, data.get("action_type", ""), timeout_seconds, on_timeout
            )
        if status == "cancelled":
            raise ApprovalCancelled(approval_id)

        # Still pending — sleep then retry.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Client-side guard: server expiry should have fired, but poll again
            # to get the fresh status from the server's inline expiry on GET.
            # If still pending, treat as expired client-side.
            return _handle_timeout(
                approval_id, data.get("action_type", ""), timeout_seconds, on_timeout
            )
        sleep_time = min(_next_interval(interval_idx), remaining)
        interval_idx += 1
        time.sleep(sleep_time)


# ── Async polling loop ─────────────────────────────────────────────────────────


async def poll_approval_async(
    http,                 # httpx.AsyncClient, already configured with auth headers
    workspace_id: str,
    approval_id: str,
    timeout_seconds: int,
    on_timeout: str,
) -> Union[ApprovalToken, TimeoutToken]:
    """Async variant — uses asyncio.sleep so the event loop is never blocked."""
    import asyncio

    url = f"/api/workspaces/{workspace_id}/governance/approvals/{approval_id}"
    interval_idx = 0
    deadline = time.monotonic() + timeout_seconds

    while True:
        resp = await http.get(url)
        if resp.status_code >= 400:
            raise MimaGovernanceError(
                f"Approval poll failed: HTTP {resp.status_code} — {resp.text}"
            )
        data = resp.json()
        status = data["status"]

        if status == "approved":
            return ApprovalToken(
                approval_id=approval_id,
                action_type=data.get("action_type", ""),
                approved_by=data.get("approved_by", ""),
                approved_at=_parse_dt(data.get("decided_at")),
                expires_at=_parse_dt(data.get("expires_at")),
            )
        if status == "rejected":
            raise ApprovalDenied(
                approval_id,
                data.get("approved_by", ""),
                data.get("rejection_reason"),
            )
        if status == "expired":
            return _handle_timeout(
                approval_id, data.get("action_type", ""), timeout_seconds, on_timeout
            )
        if status == "cancelled":
            raise ApprovalCancelled(approval_id)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _handle_timeout(
                approval_id, data.get("action_type", ""), timeout_seconds, on_timeout
            )
        sleep_time = min(_next_interval(interval_idx), remaining)
        interval_idx += 1
        await asyncio.sleep(sleep_time)
