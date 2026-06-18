"""AutoGen integration — attest every agent exchange automatically.

Targets AutoGen 0.2/0.3 ConversableAgent (synchronous API).

Usage:
    from mima_governance.integrations import MimaAutoGenMiddleware

    middleware = MimaAutoGenMiddleware(mima)
    agent.register_hook("process_message_before_send", middleware.process_message_before_send)
    agent.register_hook("process_last_message", middleware.process_last_message)
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, List, Optional, Union

try:
    import pyautogen  # noqa: F401 — import check only
except ImportError:
    raise ImportError(
        "AutoGen integration requires pyautogen. "
        "Install with: pip install mima-governance[autogen]"
    )

from mima_governance.client import MimaGovernance
from mima_governance.types import AttestationRecord


class MimaAutoGenMiddleware:
    """
    Hook-based AutoGen middleware that attests every agent exchange.

    Registers two hooks on a ConversableAgent:
      - ``process_message_before_send``: captures the outbound message for hashing.
      - ``process_last_message``: builds and enqueues an AttestationRecord.

    Each hook must be registered separately:

        middleware = MimaAutoGenMiddleware(mima, agent_name="procurement-agent")
        agent.register_hook(
            "process_message_before_send",
            middleware.process_message_before_send,
        )
        agent.register_hook(
            "process_last_message",
            middleware.process_last_message,
        )
    """

    def __init__(
        self,
        client: MimaGovernance,
        *,
        agent_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self._client     = client
        self._agent_name = agent_name or "autogen_agent"
        self._model_id   = model_id
        # Stash the last outbound message hash between the two hooks.
        self._pending_input_hash: Optional[str] = None

    # ── Hook: process_message_before_send ────────────────────────────────────

    def process_message_before_send(
        self,
        message: Union[dict, str],
        sender: Any,
        recipient: Any,
        silent: bool,
    ) -> Union[dict, str]:
        """Capture the outbound message for later attestation. Returns message unchanged."""
        self._pending_input_hash = _hash_message(message)
        return message

    # ── Hook: process_last_message ───────────────────────────────────────────

    def process_last_message(
        self,
        messages: List[Union[dict, str]],
    ) -> List[Union[dict, str]]:
        """Attest the last message exchange. Returns messages unchanged.

        Enqueues one AttestationRecord per call. The record is flushed to the
        Mima ledger on the next batch flush interval (default 30 s) or on close().
        """
        last = messages[-1] if messages else {}
        output_hash = _hash_message(last)
        input_hash  = self._pending_input_hash or ""
        self._pending_input_hash = None

        record = AttestationRecord(
            tool_name=f"autogen:{self._agent_name}",
            input_hash=input_hash,
            output_hash=output_hash,
            model_id=self._model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self._client.authorised_by,
        )
        self._client._enqueue(record)
        return messages


# ── Utilities ─────────────────────────────────────────────────────────────────


def _hash_message(message: Union[dict, str, Any]) -> str:
    """Deterministic SHA-256 of an AutoGen message dict or string."""
    import json

    try:
        if isinstance(message, dict):
            text = json.dumps(message, sort_keys=True, default=str)
        else:
            text = str(message)
    except (TypeError, ValueError):
        text = repr(message)

    return hashlib.sha256(text.encode()).hexdigest()
