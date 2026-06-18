"""Mima AI Governance SDK — attest any AI execution in one line."""

from mima_governance.client import MimaGovernance
from mima_governance.types import AttestationResult, AuthorisedBy, GrcRecord, GrcResult

__all__ = ["MimaGovernance", "AttestationResult", "AuthorisedBy", "GrcRecord", "GrcResult"]
__version__ = "0.2.0"
