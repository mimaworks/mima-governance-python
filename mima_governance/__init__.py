"""Mima AI Governance SDK — attest any AI execution in one line."""

from mima_governance.async_client import AsyncMimaGovernance
from mima_governance.client import MimaGovernance
from mima_governance._base import MimaAttestationError
from mima_governance.types import AttestationResult, AuthorisedBy, GrcRecord, GrcResult
from mima_governance.guard import enable_guard
from mima_governance.approvals import (
    ApprovalToken,
    TimeoutToken,
    ApprovalDenied,
    ApprovalTimeout,
    ApprovalCancelled,
    MimaGovernanceError,
)

__all__ = [
    "MimaGovernance",
    "AsyncMimaGovernance",
    "MimaAttestationError",
    "AttestationResult",
    "AuthorisedBy",
    "GrcRecord",
    "GrcResult",
    "enable_guard",
    # Pre-approval gates
    "ApprovalToken",
    "TimeoutToken",
    "ApprovalDenied",
    "ApprovalTimeout",
    "ApprovalCancelled",
    "MimaGovernanceError",
]
__version__ = "0.3.0"
