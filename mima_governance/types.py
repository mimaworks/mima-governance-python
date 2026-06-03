"""Shared types for the Mima Governance SDK."""

from dataclasses import dataclass, field
from typing import Optional


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
