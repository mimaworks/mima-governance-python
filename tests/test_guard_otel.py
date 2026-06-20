"""Tests for the OTEL emit path in mima_governance.guard."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import mima_governance.guard as guard_mod


@pytest.fixture(autouse=True)
def reset_otel_cache():
    """Reset the cached OTEL state before each test."""
    guard_mod._otel_checked = False
    guard_mod._otel_available = False
    yield
    guard_mod._otel_checked = False
    guard_mod._otel_available = False


class TestHasConfiguredOtel:
    """Tests for _has_configured_otel() detection and caching."""

    def test_returns_true_when_real_provider_configured(self):
        """When a real (non-proxy, non-NoOp) TracerProvider is set, returns True."""
        from opentelemetry import trace
        from opentelemetry.trace import NoOpTracerProvider, ProxyTracerProvider

        # Create a provider that is NOT NoOp or Proxy
        class RealTracerProvider(trace.TracerProvider):
            def get_tracer(self, *a, **kw):
                return MagicMock()

        real_provider = RealTracerProvider()
        assert not isinstance(real_provider, NoOpTracerProvider)
        assert not isinstance(real_provider, ProxyTracerProvider)

        with patch("opentelemetry.trace.get_tracer_provider", return_value=real_provider):
            result = guard_mod._has_configured_otel()

        assert result is True

    def test_returns_false_when_proxy_provider(self):
        """When the default ProxyTracerProvider is active, returns False."""
        from opentelemetry.trace import ProxyTracerProvider

        proxy = ProxyTracerProvider()
        with patch("opentelemetry.trace.get_tracer_provider", return_value=proxy):
            result = guard_mod._has_configured_otel()

        assert result is False

    def test_returns_false_when_noop_provider(self):
        """When NoOpTracerProvider is explicitly set, returns False."""
        from opentelemetry.trace import NoOpTracerProvider

        noop = NoOpTracerProvider()
        with patch("opentelemetry.trace.get_tracer_provider", return_value=noop):
            result = guard_mod._has_configured_otel()

        assert result is False

    def test_returns_false_when_import_fails(self):
        """When opentelemetry is not importable, returns False (fail-safe)."""
        # Temporarily hide the real opentelemetry modules
        real_modules = {}
        otel_keys = [k for k in sys.modules if k.startswith("opentelemetry")]
        for k in otel_keys:
            real_modules[k] = sys.modules.pop(k)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def failing_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=failing_import):
                result = guard_mod._has_configured_otel()
            assert result is False
        finally:
            sys.modules.update(real_modules)

    def test_caches_after_first_call(self):
        """Detection only runs once; subsequent calls use cached value."""
        from opentelemetry.trace import ProxyTracerProvider

        proxy = ProxyTracerProvider()
        with patch("opentelemetry.trace.get_tracer_provider", return_value=proxy):
            result1 = guard_mod._has_configured_otel()

        assert result1 is False
        assert guard_mod._otel_checked is True

        # Second call: even with a different provider, cached value remains
        from opentelemetry import trace

        class FakeReal(trace.TracerProvider):
            def get_tracer(self, *a, **kw):
                return MagicMock()

        with patch("opentelemetry.trace.get_tracer_provider", return_value=FakeReal()):
            result2 = guard_mod._has_configured_otel()

        assert result2 is False  # Still cached as False

    def test_caches_true_value(self):
        """Once detected as available, stays True on subsequent calls."""
        guard_mod._otel_checked = True
        guard_mod._otel_available = True

        result = guard_mod._has_configured_otel()
        assert result is True

    def test_exception_falls_safe_to_false(self):
        """Any unexpected exception in detection results in False."""
        with patch("opentelemetry.trace.get_tracer_provider", side_effect=RuntimeError("boom")):
            result = guard_mod._has_configured_otel()

        assert result is False


class TestEmitOtelSpan:
    """Tests for _emit_otel_span() span emission."""

    def test_emits_span_with_correct_attributes(self):
        """Span is created with mima.call_site, mima.attested, mima.workspace_id."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
            with patch("mima_governance.config.get_workspace_id", return_value="ws-123"):
                result = guard_mod._emit_otel_span("openai.chat")

        assert result is True
        mock_tracer.start_as_current_span.assert_called_once_with("mima.ai_call")

        calls = mock_span.set_attribute.call_args_list
        attrs = {c[0][0]: c[0][1] for c in calls}
        assert attrs["mima.call_site"] == "openai.chat"
        assert attrs["mima.attested"] is False
        assert attrs["mima.workspace_id"] == "ws-123"

    def test_omits_workspace_id_when_none(self):
        """When workspace_id is not configured, attribute is not set."""
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with patch("opentelemetry.trace.get_tracer", return_value=mock_tracer):
            with patch("mima_governance.config.get_workspace_id", return_value=None):
                result = guard_mod._emit_otel_span("anthropic.messages")

        assert result is True
        attr_keys = [c[0][0] for c in mock_span.set_attribute.call_args_list]
        assert "mima.workspace_id" not in attr_keys

    def test_returns_false_on_exception(self):
        """Any exception in span creation returns False, never raises."""
        with patch("opentelemetry.trace.get_tracer", side_effect=RuntimeError("tracer broken")):
            result = guard_mod._emit_otel_span("litellm.completion")

        assert result is False


class TestAppendReportOtelIntegration:
    """Tests for _append_report using the OTEL path."""

    def test_uses_otel_when_configured(self):
        """When OTEL is configured, _append_report emits via OTEL and skips socket."""
        guard_mod._otel_checked = True
        guard_mod._otel_available = True

        with patch.object(guard_mod, "_emit_otel_span", return_value=True) as mock_emit:
            with patch.object(guard_mod, "_send_via_socket") as mock_socket:
                guard_mod._append_report("openai.chat")

        mock_emit.assert_called_once_with("openai.chat")
        mock_socket.assert_not_called()

    def test_falls_back_to_socket_when_otel_not_configured(self):
        """When OTEL is not configured, falls back to daemon socket."""
        guard_mod._otel_checked = True
        guard_mod._otel_available = False

        with patch.object(guard_mod, "_emit_otel_span") as mock_emit:
            with patch.object(guard_mod, "_send_via_socket", return_value=True) as mock_socket:
                guard_mod._append_report("openai.chat")

        mock_emit.assert_not_called()
        mock_socket.assert_called_once()

    def test_falls_back_to_socket_when_otel_emit_fails(self):
        """If _emit_otel_span returns False, falls back to socket."""
        guard_mod._otel_checked = True
        guard_mod._otel_available = True

        with patch.object(guard_mod, "_emit_otel_span", return_value=False):
            with patch.object(guard_mod, "_send_via_socket", return_value=True) as mock_socket:
                guard_mod._append_report("openai.chat")

        mock_socket.assert_called_once()

    def test_falls_back_to_queue_when_socket_fails(self):
        """Full fallback chain: OTEL fails -> socket fails -> in-process queue."""
        guard_mod._otel_checked = True
        guard_mod._otel_available = True

        with patch.object(guard_mod, "_emit_otel_span", return_value=False):
            with patch.object(guard_mod, "_send_via_socket", return_value=False):
                with patch.object(guard_mod, "_ensure_report_thread") as mock_queue:
                    mock_q = MagicMock()
                    mock_queue.return_value = mock_q
                    guard_mod._append_report("openai.chat")

        mock_q.put_nowait.assert_called_once()

    def test_never_raises_on_otel_exception(self):
        """Even if _emit_otel_span raises, _append_report swallows it."""
        guard_mod._otel_checked = True
        guard_mod._otel_available = True

        with patch.object(guard_mod, "_emit_otel_span", side_effect=RuntimeError("boom")):
            with patch.object(guard_mod, "_send_via_socket", return_value=True):
                # Must not raise
                guard_mod._append_report("openai.chat")
