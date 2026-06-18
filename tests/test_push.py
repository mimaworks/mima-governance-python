"""Tests for `mima push` — GRC evidence push from CLI/CI."""

import json
import sys
from unittest.mock import patch, MagicMock

import pytest


class TestCmdPush:
    """mima push subcommand."""

    def test_push_help(self, capsys):
        with patch("sys.argv", ["mima", "push", "--help"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "change_event" in out
        assert "record_type" in out

    def test_push_not_logged_in(self, capsys, monkeypatch):
        monkeypatch.delenv("MIMA_API_KEY", raising=False)
        monkeypatch.delenv("MIMA_WORKSPACE_ID", raising=False)
        with patch("mima_governance.config.get_api_key", return_value=None):
            with patch("mima_governance.config.get_workspace_id", return_value=None):
                with patch("sys.argv", ["mima", "push", "change_event"]):
                    with pytest.raises(SystemExit) as exc:
                        from mima_governance.cli import main
                        main()
                    assert exc.value.code == 1
        assert "not logged in" in capsys.readouterr().err

    def test_push_unknown_record_type(self, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "key")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")
        with patch("sys.argv", ["mima", "push", "bogus_type"]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1
        assert "unknown record_type" in capsys.readouterr().err

    @patch("httpx.post")
    def test_push_change_event_success(self, mock_post, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "mima_ext_test")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws-123")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "record_id": "rec-abc-123",
            "record_type": "change_event",
            "mapped_controls": ["SOC2_CC8.1", "ISO27001_2022_8.32"],
        }
        mock_post.return_value = mock_resp

        with patch("sys.argv", [
            "mima", "push", "change_event",
            "--by", "ci-bot@company.com",
            "--description", "Deploy v1.2.3",
            "--environment", "production",
            "--system", "api-service",
            "--change-id", "JIRA-99",
        ]):
            from mima_governance.cli import main
            main()

        out = capsys.readouterr().out
        assert "rec-abc-123" in out
        assert "SOC2_CC8.1" in out

        # Verify payload sent to API
        call_kwargs = mock_post.call_args
        sent = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
        assert sent["record_type"] == "change_event"
        assert sent["payload"]["by"] == "ci-bot@company.com"
        assert sent["payload"]["change_id"] == "JIRA-99"
        assert sent["environment"] == "production"

    @patch("httpx.post")
    def test_push_change_event_json_output(self, mock_post, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "mima_ext_test")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws-123")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "record_id": "rec-xyz",
            "record_type": "change_event",
            "mapped_controls": [],
        }
        mock_post.return_value = mock_resp

        with patch("sys.argv", [
            "mima", "push", "change_event",
            "--by", "ci", "--description", "deploy", "--environment", "prod", "--system", "api",
            "--json",
        ]):
            from mima_governance.cli import main
            main()

        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["record_id"] == "rec-xyz"

    def test_push_change_event_missing_required(self, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "key")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        with patch("sys.argv", [
            "mima", "push", "change_event",
            "--by", "ci",
            # missing --description, --environment, --system
        ]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "missing" in err.lower()

    @patch("httpx.post")
    def test_push_vendor_risk_success(self, mock_post, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "key")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "record_id": "rec-vr-1",
            "record_type": "vendor_risk",
            "mapped_controls": ["SOC2_CC9.2"],
        }
        mock_post.return_value = mock_resp

        with patch("sys.argv", [
            "mima", "push", "vendor_risk",
            "--vendor", "OpenAI",
            "--tier", "high",
            "--last-reviewed", "2026-06-01",
            "--findings", "2",
        ]):
            from mima_governance.cli import main
            main()

        out = capsys.readouterr().out
        assert "rec-vr-1" in out
        assert "SOC2_CC9.2" in out

    def test_push_vendor_risk_invalid_tier(self, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "key")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        with patch("sys.argv", [
            "mima", "push", "vendor_risk",
            "--vendor", "OpenAI", "--tier", "extreme",
            "--last-reviewed", "2026-06-01",
        ]):
            with pytest.raises(SystemExit) as exc:
                from mima_governance.cli import main
                main()
            assert exc.value.code == 1
        assert "tier" in capsys.readouterr().err

    @patch("httpx.post")
    def test_push_stdin_mode(self, mock_post, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "key")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "record_id": "rec-stdin",
            "record_type": "incident_report",
            "mapped_controls": ["SOC2_CC7.3"],
        }
        mock_post.return_value = mock_resp

        stdin_payload = json.dumps({
            "record_type": "incident_report",
            "payload": {"title": "PII leak", "severity": "high"},
            "system_name": "my-service",
        })

        with patch("sys.argv", ["mima", "push", "--stdin"]):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.read.return_value = stdin_payload
                from mima_governance.cli import main
                main()

        out = capsys.readouterr().out
        assert "rec-stdin" in out

    def test_push_stdin_invalid_json(self, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "key")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        with patch("sys.argv", ["mima", "push", "--stdin"]):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.read.return_value = "not json {"
                with pytest.raises(SystemExit) as exc:
                    from mima_governance.cli import main
                    main()
                assert exc.value.code == 1
        assert "invalid JSON" in capsys.readouterr().err

    @patch("httpx.post")
    def test_push_incident_report_success(self, mock_post, capsys, monkeypatch):
        monkeypatch.setenv("MIMA_API_KEY", "key")
        monkeypatch.setenv("MIMA_WORKSPACE_ID", "ws")
        monkeypatch.setenv("MIMA_BASE_URL", "http://localhost:8081")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "record_id": "rec-ir-1",
            "record_type": "incident_report",
            "mapped_controls": ["SOC2_CC7.3", "SOC2_CC7.4"],
        }
        mock_post.return_value = mock_resp

        with patch("sys.argv", [
            "mima", "push", "incident_report",
            "--title", "LLM returned PII",
            "--severity", "medium",
            "--description", "Model leaked email address",
            "--affected-systems", "ai-chat,api-gateway",
        ]):
            from mima_governance.cli import main
            main()

        sent = mock_post.call_args[1]["json"]
        assert sent["payload"]["affected_systems"] == ["ai-chat", "api-gateway"]
        assert "rec-ir-1" in capsys.readouterr().out
