"""Tests for the Mima Governance SDK."""

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from mima_governance import MimaGovernance, AttestationResult, AuthorisedBy


@pytest.fixture
def client():
    return MimaGovernance(
        workspace_id="test-workspace-id",
        api_key="mima_ext_test_key",
        system_name="test-system",
        base_url="https://test.mima.ai",
        on_error="raise",
    )


class TestDecorator:
    def test_attest_decorator_calls_function(self, client):
        """Decorated function still executes and returns its value."""

        with patch.object(client, "_push_sync", return_value=AttestationResult(
            attestation_id="test-id",
            external_verified=False,
            trust_tier="declared",
            detail="ok",
        )):

            @client.attest(tool_name="my_tool")
            def my_func(x: int) -> int:
                return x * 2

            result = my_func(5)
            assert result == 10

    def test_attest_decorator_pushes_attestation(self, client):
        """Decorator pushes an attestation with correct hashes."""

        push_mock = MagicMock(return_value=AttestationResult(
            attestation_id="test-id",
            external_verified=False,
            trust_tier="declared",
            detail="ok",
        ))

        with patch.object(client, "_push_sync", push_mock):

            @client.attest(tool_name="hash_test")
            def identity(x):
                return x

            identity({"key": "value"})

            push_mock.assert_called_once()
            record = push_mock.call_args[0][0]
            assert record.tool_name == "hash_test"
            assert len(record.input_hash) == 64  # SHA-256 hex
            assert len(record.output_hash) == 64

    def test_attest_batch_mode_enqueues(self, client):
        """Batch mode enqueues instead of pushing immediately."""

        with patch.object(client, "_push_sync") as push_mock:
            with patch.object(client, "_enqueue") as enqueue_mock:

                @client.attest(tool_name="batch_tool", mode="batch")
                def batch_fn():
                    return "result"

                batch_fn()

                push_mock.assert_not_called()
                enqueue_mock.assert_called_once()


class TestExplicitPush:
    def test_push_builds_correct_payload(self, client):
        """Explicit push sends correct payload structure."""

        with patch.object(client._http, "post") as http_mock:
            http_mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "attestation_id": "abc-123",
                    "external_verified": False,
                    "trust_tier": "declared",
                    "detail": "stored",
                },
            )

            result = client.push(
                tool_name="explicit_tool",
                input_hash="a" * 64,
                output_hash="b" * 64,
                model_id="gpt-4o",
            )

            assert result.attestation_id == "abc-123"
            assert result.trust_tier == "declared"

            call_kwargs = http_mock.call_args
            payload = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
            assert payload["system_name"] == "test-system"
            assert payload["tool_name"] == "explicit_tool"
            assert payload["model_id"] == "gpt-4o"
            assert payload["schema_version"] == 2


class TestAuthorisedBy:
    def test_authorised_by_included_in_payload(self, client):
        """AuthorisedBy is serialized into the payload."""
        client.authorised_by = AuthorisedBy(
            identity="user@corp.com",
            role="analyst",
            session_id="sess_123",
        )

        with patch.object(client._http, "post") as http_mock:
            http_mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "attestation_id": "x",
                    "external_verified": False,
                    "trust_tier": "declared",
                    "detail": "ok",
                },
            )

            client.push("tool", "a" * 64, "b" * 64)

            payload = http_mock.call_args.kwargs.get("json") or http_mock.call_args[1]["json"]
            assert payload["authorised_by"]["identity"] == "user@corp.com"
            assert payload["authorised_by"]["role"] == "analyst"


class TestHashing:
    def test_sha256_deterministic(self):
        """Same input always produces same hash."""
        from mima_governance.client import _sha256

        h1 = _sha256('{"key": "value"}')
        h2 = _sha256('{"key": "value"}')
        assert h1 == h2
        assert len(h1) == 64

    def test_different_inputs_different_hashes(self):
        from mima_governance.client import _sha256

        h1 = _sha256("input_a")
        h2 = _sha256("input_b")
        assert h1 != h2


class TestErrorHandling:
    def test_raise_mode_throws(self, client):
        """on_error='raise' throws MimaAttestationError on failure."""

        with patch.object(client._http, "post") as http_mock:
            http_mock.return_value = MagicMock(
                status_code=500,
                text="Internal Server Error",
            )

            from mima_governance.client import MimaAttestationError

            with pytest.raises(MimaAttestationError):
                client.push("tool", "a" * 64, "b" * 64)

    def test_warn_mode_returns_empty_result(self):
        """on_error='warn' returns a result with empty attestation_id."""
        c = MimaGovernance(
            workspace_id="w",
            api_key="k",
            system_name="s",
            base_url="https://test.mima.ai",
            on_error="warn",
        )

        with patch.object(c._http, "post") as http_mock:
            http_mock.return_value = MagicMock(
                status_code=500,
                text="error",
            )

            import warnings
            with warnings.catch_warnings(record=True):
                result = c.push("tool", "a" * 64, "b" * 64)
                assert result.attestation_id == ""
                assert result.trust_tier == "declared"
