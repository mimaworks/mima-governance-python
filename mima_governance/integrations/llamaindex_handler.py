"""LlamaIndex callback integration — attest every query engine call.

Usage:
    from mima_governance.integrations import MimaLlamaIndexHandler

    llama_index.global_handler = MimaLlamaIndexHandler(mima)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mima_governance.client import MimaGovernance
from mima_governance.types import AttestationRecord

try:
    from llama_index.core.callbacks import CallbackManager, CBEventType
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install with: pip install mima-governance[llamaindex]"
    )


class MimaLlamaIndexHandler(BaseCallbackHandler):
    """
    LlamaIndex callback handler that attests LLM calls and retrieval steps.
    """

    def __init__(self, client: MimaGovernance, *, model_id: Optional[str] = None):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._client = client
        self._model_id = model_id

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if payload is None:
            return

        tool_name = f"llamaindex:{event_type.value}"
        input_hash = _hash(json.dumps(payload.get("input", ""), default=str))
        output_hash = _hash(json.dumps(payload.get("output", ""), default=str))

        record = AttestationRecord(
            tool_name=tool_name,
            input_hash=input_hash,
            output_hash=output_hash,
            model_id=self._model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self._client.authorised_by,
        )
        self._client._enqueue(record)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Flush batch on trace end
        self._client._flush_batch()


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()
