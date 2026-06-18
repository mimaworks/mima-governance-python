"""Tests for framework integrations (T10 AutoGen, T11 CrewAI stub)."""

from unittest.mock import MagicMock, patch

import pytest

from mima_governance import MimaGovernance
from mima_governance.types import AttestationRecord


@pytest.fixture
def client():
    return MimaGovernance(
        workspace_id="test-workspace-id",
        api_key="mima_ext_test_key",
        system_name="test-system",
        base_url="https://test.mima.ai",
        on_error="warn",
    )


# ── T10: AutoGen middleware ───────────────────────────────────────────────────


class TestAutoGenMiddleware:
    def test_import_fails_without_pyautogen(self):
        """Import of MimaAutoGenMiddleware raises ImportError if pyautogen missing."""
        import sys

        # Temporarily hide pyautogen if it happens to be installed.
        pyautogen_backup = sys.modules.pop("pyautogen", None)
        # Also hide from autogen_middleware module cache to force re-evaluation.
        autogen_mod_backup = sys.modules.pop(
            "mima_governance.integrations.autogen_middleware", None
        )
        try:
            with pytest.raises(ImportError, match="pyautogen"):
                from mima_governance.integrations import autogen_middleware  # noqa: F401
                import importlib
                importlib.reload(autogen_middleware)
        except ImportError:
            pass  # expected path when pyautogen is absent
        finally:
            if pyautogen_backup is not None:
                sys.modules["pyautogen"] = pyautogen_backup
            if autogen_mod_backup is not None:
                sys.modules["mima_governance.integrations.autogen_middleware"] = autogen_mod_backup

    def test_process_last_message_enqueues_one_record(self, client):
        """process_last_message enqueues exactly one AttestationRecord per call."""
        # Patch pyautogen so the import succeeds without the real package.
        fake_autogen = MagicMock()
        with patch.dict("sys.modules", {"pyautogen": fake_autogen}):
            # Force reload so the patched sys.modules takes effect.
            import importlib
            import sys

            sys.modules.pop("mima_governance.integrations.autogen_middleware", None)
            from mima_governance.integrations.autogen_middleware import MimaAutoGenMiddleware

            middleware = MimaAutoGenMiddleware(client, agent_name="test-agent")

        enqueued: list[AttestationRecord] = []

        def capture_enqueue(record: AttestationRecord) -> None:
            enqueued.append(record)

        with patch.object(client, "_enqueue", side_effect=capture_enqueue):
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
            result = middleware.process_last_message(messages)

        assert len(enqueued) == 1
        assert enqueued[0].tool_name == "autogen:test-agent"
        assert len(enqueued[0].output_hash) == 64  # SHA-256 hex
        assert result == messages  # messages returned unchanged

    def test_process_message_before_send_returns_message_unchanged(self, client):
        """process_message_before_send must return the message object unmodified."""
        fake_autogen = MagicMock()
        with patch.dict("sys.modules", {"pyautogen": fake_autogen}):
            import sys
            sys.modules.pop("mima_governance.integrations.autogen_middleware", None)
            from mima_governance.integrations.autogen_middleware import MimaAutoGenMiddleware

            middleware = MimaAutoGenMiddleware(client)

        msg = {"role": "user", "content": "test message"}
        returned = middleware.process_message_before_send(
            msg, sender=None, recipient=None, silent=False
        )
        assert returned is msg

    def test_input_hash_captured_before_send_used_in_last_message(self, client):
        """Input hash captured in process_message_before_send appears in the attestation."""
        fake_autogen = MagicMock()
        with patch.dict("sys.modules", {"pyautogen": fake_autogen}):
            import sys
            sys.modules.pop("mima_governance.integrations.autogen_middleware", None)
            from mima_governance.integrations.autogen_middleware import MimaAutoGenMiddleware

            middleware = MimaAutoGenMiddleware(client)

        captured: list[AttestationRecord] = []
        with patch.object(client, "_enqueue", side_effect=captured.append):
            msg = {"role": "user", "content": "query"}
            middleware.process_message_before_send(msg, None, None, False)
            middleware.process_last_message([{"role": "assistant", "content": "reply"}])

        assert len(captured) == 1
        # Input hash must be non-empty (was set by process_message_before_send).
        assert captured[0].input_hash != ""
        assert len(captured[0].input_hash) == 64

    def test_empty_messages_list_does_not_raise(self, client):
        """process_last_message handles an empty message list without error."""
        fake_autogen = MagicMock()
        with patch.dict("sys.modules", {"pyautogen": fake_autogen}):
            import sys
            sys.modules.pop("mima_governance.integrations.autogen_middleware", None)
            from mima_governance.integrations.autogen_middleware import MimaAutoGenMiddleware

            middleware = MimaAutoGenMiddleware(client)

        with patch.object(client, "_enqueue"):
            result = middleware.process_last_message([])
        assert result == []


# ── T11: CrewAI (deferred) ───────────────────────────────────────────────────
#
# CrewAI integration is deferred pending target version confirmation.
# Placeholder test ensures the test suite does not fail on import.


class TestCrewAICallback:
    def test_crewai_integration_deferred(self):
        """CrewAI integration is deferred — no MimaCrewAICallback exists yet."""
        with pytest.raises((ImportError, AttributeError)):
            from mima_governance.integrations import MimaCrewAICallback  # noqa: F401
