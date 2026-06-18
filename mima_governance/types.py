"""Shared types for the Mima Governance SDK."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AuthorisedBy:
    """Human principal who authorised the agent execution."""

    identity: str
    role: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"identity": self.identity}
        if self.role:
            d["role"] = self.role
        if self.session_id:
            d["session_id"] = self.session_id
        return d


@dataclass
class AttestationResult:
    """Result from pushing an attestation to the Mima ledger."""

    attestation_id: str
    external_verified: bool
    trust_tier: str
    detail: str


@dataclass
class AttestationRecord:
    """Internal record buffered for batch submission."""

    tool_name: str
    input_hash: str
    output_hash: str
    model_id: Optional[str] = None
    executed_at: Optional[str] = None
    authorised_by: Optional[AuthorisedBy] = None


@dataclass
class GrcRecord:
    """A GRC control evidence record to be pushed to the Mima ledger."""

    record_type: str
    payload: dict
    system_name: str
    identity: Optional[str] = None
    resource: Optional[str] = None
    environment: Optional[str] = None
    occurred_at: Optional[str] = None


@dataclass
class GrcResult:
    """Result from pushing a GRC evidence record to the Mima ledger.

    Check ``record_id == ''`` to detect a failed push — the SDK never raises
    on network errors; failures are surfaced via ``detail`` and a stderr warning.
    """

    record_id: str
    record_type: str
    mapped_controls: List[str]
    detail: str
