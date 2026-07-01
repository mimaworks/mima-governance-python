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


# ── T12: LangChain callback ───────────────────────────────────────────────────


class TestLangChainCallback:
    def test_import_fails_without_langchain_core(self):
        """Import raises ImportError with install hint when langchain-core is absent."""
        import sys
        import importlib

        # Block langchain_core at the sys.modules level (None = "not installed")
        block = {k: None for k in list(sys.modules) if k.startswith("langchain")}
        block["langchain_core"] = None
        block["langchain_core.callbacks"] = None
        mod_key = "mima_governance.integrations.langchain_callback"
        mod_backup = sys.modules.pop(mod_key, None)
        try:
            with patch.dict("sys.modules", block):
                sys.modules.pop(mod_key, None)
                with pytest.raises(ImportError, match="langchain"):
                    import mima_governance.integrations.langchain_callback as _m  # noqa: F401
                    importlib.reload(_m)
        finally:
            if mod_backup is not None:
                sys.modules[mod_key] = mod_backup

    def test_on_llm_end_enqueues_one_record(self, client):
        """on_llm_end enqueues exactly one AttestationRecord with tool_name langchain_llm_call."""
        import sys
        from unittest.mock import MagicMock

        # Build a minimal fake langchain_core.callbacks module
        fake_handler = type("BaseCallbackHandler", (), {
            "__init__": lambda self: None,
            "on_llm_start": lambda *a, **kw: None,
            "on_llm_end": lambda *a, **kw: None,
            "on_tool_start": lambda *a, **kw: None,
            "on_tool_end": lambda *a, **kw: None,
            "on_chain_end": lambda *a, **kw: None,
        })
        fake_callbacks_mod = MagicMock()
        fake_callbacks_mod.BaseCallbackHandler = fake_handler
        fake_lc_core = MagicMock()
        fake_lc_core.callbacks = fake_callbacks_mod

        with patch.dict("sys.modules", {
            "langchain_core": fake_lc_core,
            "langchain_core.callbacks": fake_callbacks_mod,
        }):
            sys.modules.pop("mima_governance.integrations.langchain_callback", None)
            from mima_governance.integrations.langchain_callback import MimaLangChainCallback

            cb = MimaLangChainCallback(client, model_id="test-model")

        enqueued = []
        with patch.object(client, "_enqueue", side_effect=enqueued.append):
            from uuid import uuid4
            cb.on_llm_end(response=MagicMock(), run_id=uuid4())

        assert len(enqueued) == 1
        assert enqueued[0].tool_name == "langchain_llm_call"
        assert len(enqueued[0].output_hash) == 64

    def test_on_tool_end_enqueues_record_with_tool_prefix(self, client):
        """on_tool_end enqueues a record whose tool_name starts with 'langchain:'."""
        import sys
        from unittest.mock import MagicMock

        fake_handler = type("BaseCallbackHandler", (), {
            "__init__": lambda self: None,
        })
        fake_callbacks_mod = MagicMock()
        fake_callbacks_mod.BaseCallbackHandler = fake_handler
        fake_lc_core = MagicMock()
        fake_lc_core.callbacks = fake_callbacks_mod

        with patch.dict("sys.modules", {
            "langchain_core": fake_lc_core,
            "langchain_core.callbacks": fake_callbacks_mod,
        }):
            sys.modules.pop("mima_governance.integrations.langchain_callback", None)
            from mima_governance.integrations.langchain_callback import MimaLangChainCallback
            cb = MimaLangChainCallback(client)

        enqueued = []
        with patch.object(client, "_enqueue", side_effect=enqueued.append):
            from uuid import uuid4
            cb.on_tool_end("tool output", run_id=uuid4(), name="search")

        assert len(enqueued) == 1
        assert enqueued[0].tool_name == "langchain:search"

    def test_on_chain_end_enqueues_record(self, client):
        """on_chain_end enqueues a record with tool_name langchain_chain_complete."""
        import sys
        from unittest.mock import MagicMock

        fake_handler = type("BaseCallbackHandler", (), {
            "__init__": lambda self: None,
        })
        fake_callbacks_mod = MagicMock()
        fake_callbacks_mod.BaseCallbackHandler = fake_handler
        fake_lc_core = MagicMock()
        fake_lc_core.callbacks = fake_callbacks_mod

        with patch.dict("sys.modules", {
            "langchain_core": fake_lc_core,
            "langchain_core.callbacks": fake_callbacks_mod,
        }):
            sys.modules.pop("mima_governance.integrations.langchain_callback", None)
            from mima_governance.integrations.langchain_callback import MimaLangChainCallback
            cb = MimaLangChainCallback(client)

        enqueued = []
        with patch.object(client, "_enqueue", side_effect=enqueued.append):
            from uuid import uuid4
            cb.on_chain_end({"result": "done"}, run_id=uuid4())

        assert len(enqueued) == 1
        assert enqueued[0].tool_name == "langchain_chain_complete"


# ── T13: LlamaIndex handler ───────────────────────────────────────────────────


class TestLlamaIndexHandler:
    def test_import_fails_without_llama_index_core(self):
        """Import raises ImportError with install hint when llama-index-core is absent."""
        import sys
        import importlib

        # Block llama_index at the sys.modules level (None = "not installed")
        block = {k: None for k in list(sys.modules) if k.startswith("llama_index")}
        block["llama_index"] = None
        block["llama_index.core"] = None
        block["llama_index.core.callbacks"] = None
        block["llama_index.core.callbacks.base_handler"] = None
        mod_key = "mima_governance.integrations.llamaindex_handler"
        mod_backup = sys.modules.pop(mod_key, None)
        try:
            with patch.dict("sys.modules", block):
                sys.modules.pop(mod_key, None)
                with pytest.raises(ImportError, match="llama"):
                    import mima_governance.integrations.llamaindex_handler as _m  # noqa: F401
                    importlib.reload(_m)
        finally:
            if mod_backup is not None:
                sys.modules[mod_key] = mod_backup

    def test_on_event_end_enqueues_record(self, client):
        """on_event_end enqueues a record for every LLM event."""
        import sys
        from unittest.mock import MagicMock

        # Build minimal fake llama_index.core.callbacks
        fake_cb_event_type = MagicMock()
        fake_cb_event_type.LLM = MagicMock(value="llm")

        fake_base_handler = type("BaseCallbackHandler", (), {
            "__init__": lambda self, **kw: None,
            "on_event_start": lambda *a, **kw: "",
            "on_event_end": lambda *a, **kw: None,
            "start_trace": lambda *a, **kw: None,
            "end_trace": lambda *a, **kw: None,
        })
        fake_callbacks_mod = MagicMock()
        fake_callbacks_mod.CBEventType = fake_cb_event_type
        fake_callbacks_mod.CallbackManager = MagicMock()

        fake_base_handler_mod = MagicMock()
        fake_base_handler_mod.BaseCallbackHandler = fake_base_handler

        fake_core = MagicMock()
        fake_core.callbacks = fake_callbacks_mod

        with patch.dict("sys.modules", {
            "llama_index": MagicMock(),
            "llama_index.core": fake_core,
            "llama_index.core.callbacks": fake_callbacks_mod,
            "llama_index.core.callbacks.base_handler": fake_base_handler_mod,
        }):
            sys.modules.pop("mima_governance.integrations.llamaindex_handler", None)
            from mima_governance.integrations.llamaindex_handler import MimaLlamaIndexHandler
            handler = MimaLlamaIndexHandler(client, model_id="test-model")

        enqueued = []
        with patch.object(client, "_enqueue", side_effect=enqueued.append):
            event_type = MagicMock()
            event_type.value = "llm"
            handler.on_event_end(
                event_type,
                payload={"input": "hello", "output": "world"},
                event_id="evt-1",
            )

        assert len(enqueued) == 1
        assert enqueued[0].tool_name == "llamaindex:llm"
        assert len(enqueued[0].output_hash) == 64

    def test_on_event_end_with_none_payload_does_not_enqueue(self, client):
        """on_event_end with None payload must not enqueue anything."""
        import sys
        from unittest.mock import MagicMock

        fake_base_handler = type("BaseCallbackHandler", (), {
            "__init__": lambda self, **kw: None,
        })
        fake_callbacks_mod = MagicMock()
        fake_base_handler_mod = MagicMock()
        fake_base_handler_mod.BaseCallbackHandler = fake_base_handler
        fake_core = MagicMock()
        fake_core.callbacks = fake_callbacks_mod

        with patch.dict("sys.modules", {
            "llama_index": MagicMock(),
            "llama_index.core": fake_core,
            "llama_index.core.callbacks": fake_callbacks_mod,
            "llama_index.core.callbacks.base_handler": fake_base_handler_mod,
        }):
            sys.modules.pop("mima_governance.integrations.llamaindex_handler", None)
            from mima_governance.integrations.llamaindex_handler import MimaLlamaIndexHandler
            handler = MimaLlamaIndexHandler(client)

        enqueued = []
        with patch.object(client, "_enqueue", side_effect=enqueued.append):
            handler.on_event_end(MagicMock(), payload=None, event_id="evt-2")

        assert enqueued == []

    def test_end_trace_flushes_batch(self, client):
        """end_trace calls _flush_batch on the client."""
        import sys
        from unittest.mock import MagicMock

        fake_base_handler = type("BaseCallbackHandler", (), {
            "__init__": lambda self, **kw: None,
        })
        fake_callbacks_mod = MagicMock()
        fake_base_handler_mod = MagicMock()
        fake_base_handler_mod.BaseCallbackHandler = fake_base_handler
        fake_core = MagicMock()
        fake_core.callbacks = fake_callbacks_mod

        with patch.dict("sys.modules", {
            "llama_index": MagicMock(),
            "llama_index.core": fake_core,
            "llama_index.core.callbacks": fake_callbacks_mod,
            "llama_index.core.callbacks.base_handler": fake_base_handler_mod,
        }):
            sys.modules.pop("mima_governance.integrations.llamaindex_handler", None)
            from mima_governance.integrations.llamaindex_handler import MimaLlamaIndexHandler
            handler = MimaLlamaIndexHandler(client)

        with patch.object(client, "_flush_batch") as mock_flush:
            handler.end_trace(trace_id="trace-1")

        mock_flush.assert_called_once()
