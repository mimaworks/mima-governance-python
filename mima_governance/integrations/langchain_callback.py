"""LangChain callback integration — attest every chain step automatically.

Usage:
    from mima_governance.integrations import MimaLangChainCallback

    chain = my_chain.with_config(callbacks=[MimaLangChainCallback(mima)])
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from mima_governance.client import MimaGovernance
from mima_governance.types import AttestationRecord

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install mima-governance[langchain]"
    )


class MimaLangChainCallback(BaseCallbackHandler):
    """
    LangChain callback that attests every LLM call, tool invocation, and chain step.

    Attestations are batched and flushed every 10 steps or 5 seconds.
    """

    def __init__(self, client: MimaGovernance, *, model_id: Optional[str] = None):
        self._client = client
        self._model_id = model_id

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pass  # We attest on completion, not start

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        input_hash = _hash(str(kwargs.get("prompts", "")))
        output_hash = _hash(str(response))
        model = self._model_id or _extract_model(kwargs)

        record = AttestationRecord(
            tool_name="langchain_llm_call",
            input_hash=input_hash,
            output_hash=output_hash,
            model_id=model,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self._client.authorised_by,
        )
        self._client._enqueue(record)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pass

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        tool_name = kwargs.get("name", "langchain_tool")
        input_hash = _hash(kwargs.get("input_str", ""))
        output_hash = _hash(output)

        record = AttestationRecord(
            tool_name=f"langchain:{tool_name}",
            input_hash=input_hash,
            output_hash=output_hash,
            model_id=self._model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self._client.authorised_by,
        )
        self._client._enqueue(record)

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any) -> None:
        output_hash = _hash(json.dumps(outputs, default=str, sort_keys=True))
        record = AttestationRecord(
            tool_name="langchain_chain_complete",
            input_hash="",  # chain input already attested at step level
            output_hash=output_hash,
            model_id=self._model_id,
            executed_at=datetime.now(timezone.utc).isoformat(),
            authorised_by=self._client.authorised_by,
        )
        self._client._enqueue(record)


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def _extract_model(kwargs: Dict[str, Any]) -> Optional[str]:
    invocation = kwargs.get("invocation_params", {})
    return invocation.get("model_name") or invocation.get("model")
